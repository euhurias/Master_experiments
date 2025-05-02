import sys
import os
import argparse
import logging
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torchcrf import CRF
from seqeval.metrics import classification_report, f1_score
from sklearn.model_selection import KFold

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Arquivo único com dados no formato IOB')
    parser.add_argument('--output_dir', required=True, help='Diretório para salvar resultados')
    parser.add_argument('--k_folds', type=int, default=5, help='Número de folds (padrão: 5)')
    parser.add_argument('--batch_size', type=int, default=16, help='Tamanho do batch (padrão: 16)')
    parser.add_argument('--epochs', type=int, default=15, help='Número de épocas (padrão: 15)')
    parser.add_argument('--patience', type=int, default=5, help='Paciência para early stopping')
    return parser.parse_args()

class Config:
    model_name = "neuralmind/bert-base-portuguese-cased"
    max_seq_length = 180
    learning_rate = 5e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42

class BertCRF(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)
        
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return loss
        return self.crf.decode(emissions, mask=attention_mask.bool())

class InputExample:
    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels

class DataProcessor:
    def __init__(self):
        self.label_map = {}
        self.labels = []
    
    def get_examples(self, data_file):
        examples = []
        all_labels = set()
        
        with open(data_file, 'r', encoding='utf-8') as f:
            entries = f.read().strip().split("\n\n")
            
            for i, entry in enumerate(entries):
                words, labels = [], []
                for line in entry.split('\n'):
                    if line.strip():
                        parts = line.strip().split()
                        words.append(parts[0])
                        label = parts[-1].upper()
                        labels.append(label)
                        all_labels.add(label)
                if words:
                    examples.append(InputExample(i, words, labels))
        
        self._create_label_maps(all_labels)
        return examples
    
    def _create_label_maps(self, labels):
        special_labels = ['[PAD]', '[CLS]', '[SEP]', 'X']
        self.labels = special_labels + sorted([l for l in labels if l not in special_labels])
        self.label_map = {label: i for i, label in enumerate(self.labels)}

def convert_examples_to_features(examples, tokenizer, processor):
    features = []
    for ex in examples:
        tokens = ['[CLS]']
        label_ids = [processor.label_map['[CLS]']]
        predict_mask = [0]
        
        for word, label in zip(ex.words, ex.labels):
            word_tokens = tokenizer.tokenize(word) or [tokenizer.unk_token]
            tokens.extend(word_tokens)
            main_label = processor.label_map.get(label, processor.label_map['X'])
            label_ids.extend([main_label] + [processor.label_map['X']]*(len(word_tokens)-1))
            predict_mask.extend([1] + [0]*(len(word_tokens)-1))
        
        tokens = tokens[:Config.max_seq_length-1]
        label_ids = label_ids[:Config.max_seq_length-1]
        predict_mask = predict_mask[:Config.max_seq_length-1]
        
        tokens.append('[SEP]')
        label_ids.append(processor.label_map['[SEP]'])
        predict_mask.append(0)
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        padding = Config.max_seq_length - len(input_ids)
        
        features.append({
            'input_ids': input_ids + [0]*padding,
            'attention_mask': [1]*len(input_ids) + [0]*padding,
            'labels': label_ids + [processor.label_map['[PAD]']]*padding,
            'predict_mask': predict_mask + [0]*padding
        })
    return features

class NerDataset(data.Dataset):
    def __init__(self, examples, tokenizer, processor):
        self.features = convert_examples_to_features(examples, tokenizer, processor)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feat = self.features[idx]
        return {
            'input_ids': torch.tensor(feat['input_ids']),
            'attention_mask': torch.tensor(feat['attention_mask']),
            'labels': torch.tensor(feat['labels']),
            'predict_mask': torch.tensor(feat['predict_mask'], dtype=torch.bool)
        }

def evaluate(model, dataloader, processor):
    model.eval()
    true_labels, pred_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(Config.device)
            attention_mask = batch['attention_mask'].to(Config.device)
            labels_batch = batch['labels'].cpu().numpy()
            masks_batch = batch['predict_mask'].cpu().numpy()
            
            preds = model(input_ids, attention_mask)
            
            for i in range(len(preds)):
                current_true = []
                current_pred = []
                for j in range(len(masks_batch[i])):
                    if masks_batch[i][j]:
                        true_label = processor.labels[labels_batch[i][j]]
                        pred_label = processor.labels[preds[i][j]]
                        current_true.append(true_label)
                        current_pred.append(pred_label)
                if current_true:
                    true_labels.append(current_true)
                    pred_labels.append(current_pred)
    
    return {
        'f1': f1_score(true_labels, pred_labels) if true_labels else 0.0,
        'report': classification_report(true_labels, pred_labels, zero_division=0)
    }

def train_fold(fold, train_loader, val_loader, processor, args):
    model = BertCRF(Config.model_name, len(processor.labels)).to(Config.device)
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader)*args.epochs
    )
    
    best_f1 = 0
    patience_counter = 0
    best_model_path = os.path.join(args.output_dir, f"fold_BERT_CRF_{fold}_best.pt")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(Config.device),
                'attention_mask': batch['attention_mask'].to(Config.device),
                'labels': batch['labels'].to(Config.device)
            }
            
            loss = model(**inputs)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        
        results = evaluate(model, val_loader, processor)
        
        if results['f1'] > best_f1:
            best_f1 = results['f1']
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break
    
    model.load_state_dict(torch.load(best_model_path))
    return evaluate(model, val_loader, processor)

def main():
    args = parse_args()
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'resultados_bert_crf.txt')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    try:
        processor = DataProcessor()
        examples = processor.get_examples(args.data)
        tokenizer = BertTokenizer.from_pretrained(Config.model_name)
        
        kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=Config.seed)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(examples)):
            train_examples = [examples[i] for i in train_idx]
            val_examples = [examples[i] for i in val_idx]
            
            train_dataset = NerDataset(train_examples, tokenizer, processor)
            val_dataset = NerDataset(val_examples, tokenizer, processor)
            
            train_loader = data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True
            )
            val_loader = data.DataLoader(
                val_dataset,
                batch_size=args.batch_size
            )
            
            results = train_fold(fold+1, train_loader, val_loader, processor, args)
            
            logging.info(f"\n=== Fold {fold+1} ===")
            logging.info(f"F1-Score: {results['f1']:.4f}")
            logging.info("Relatório de Classificação:")
            logging.info(results['report'])
            
            fold_results.append(results['f1'])
        
        logging.info("\n=== Resultado Final ===")
        logging.info(f"F1 Médio: {np.mean(fold_results):.4f} (±{np.std(fold_results):.4f})")
        logging.info(f"Valores por Fold: {[round(f, 4) for f in fold_results]}")
    
    except Exception as e:
        logging.error(f"Erro: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()