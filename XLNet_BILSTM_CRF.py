import sys
import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.optim import AdamW
from transformers import XLNetTokenizer, XLNetModel, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score
from torchcrf import CRF
from sklearn.model_selection import KFold

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Arquivo único com dados no formato IOB')
    parser.add_argument('--output_dir', required=True, help='Diretório para salvar resultados')
    parser.add_argument('--k_folds', type=int, default=5, help='Número de folds (padrão: 5)')
    parser.add_argument('--batch_size', type=int, default=16, help='Tamanho do batch (padrão: 16)')
    parser.add_argument('--epochs', type=int, default=15, help='Número de épocas (padrão: 15)')
    parser.add_argument('--patience', type=int, default=3, help='Paciência para early stopping (padrão: 3)')
    return parser.parse_args()

class Config:
    model_name = "xlnet-base-cased"
    max_seq_length = 128
    learning_rate = 2e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42

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
    for ex_idx, ex in enumerate(examples):
        try:
            tokens = []
            label_ids = []
            
            for word, label in zip(ex.words, ex.labels):
                if label not in processor.label_map:
                    raise ValueError(f"Label '{label}' não encontrado no mapeamento")
                
                word_tokens = tokenizer.tokenize(word) or [tokenizer.unk_token]
                tokens.extend(word_tokens)
                
                main_label = processor.label_map[label]
                label_ids.append(main_label)
                if len(word_tokens) > 1:
                    label_ids.extend([-100] * (len(word_tokens) - 1))

            tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
            label_ids = [-100] + label_ids + [-100]

            max_len = Config.max_seq_length
            tokens = tokens[:max_len]
            label_ids = label_ids[:max_len]

            pad_len = max_len - len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.pad_token_id] * pad_len
            attention_mask = [1] * len(tokens) + [0] * pad_len
            label_ids += [-100] * pad_len

            features.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': label_ids
            })

        except Exception as e:
            logging.error(f"Erro no exemplo {ex_idx}: {str(e)}")
            logging.error(f"Texto: {' '.join(ex.words)}")
            logging.error(f"Labels: {ex.labels}")
            raise

    return features

class NERDataset(data.Dataset):
    def __init__(self, examples, tokenizer, processor):
        self.features = convert_examples_to_features(examples, tokenizer, processor)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feat = self.features[idx]
        return {
            'input_ids': torch.tensor(feat['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(feat['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(feat['labels'], dtype=torch.long)
        }

class XLNetBiLSTMCRF(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.xlnet = XLNetModel.from_pretrained(Config.model_name)
        self.bilstm = nn.LSTM(
            input_size=self.xlnet.config.d_model,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=True
        )
        self.classifier = nn.Linear(512, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.bilstm.named_parameters():
            if 'weight_ih' in name: nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name: nn.init.orthogonal_(param)
            elif 'bias' in name: nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        nn.init.normal_(self.crf.start_transitions, 0, 0.1)
        nn.init.normal_(self.crf.end_transitions, 0, 0.1)
        nn.init.normal_(self.crf.transitions, 0, 0.1)
    
    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.xlnet(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_output, _ = self.bilstm(sequence_output)
        logits = self.classifier(lstm_output)
        
        if labels is not None:
            logits = logits[:, 1:-1, :]
            labels = labels[:, 1:-1]
            crf_mask = (labels != -100) & (attention_mask[:, 1:-1] == 1)
            crf_labels = labels.clone().masked_fill(labels == -100, 0)
            loss = -self.crf(logits, crf_labels, mask=crf_mask, reduction='mean')
            return loss, logits
        return None, logits
    
    def decode(self, logits, attention_mask):
        return self.crf.decode(logits[:, 1:-1, :], mask=attention_mask[:, 1:-1].bool())

def evaluate(model, dataloader, processor):
    model.eval()
    true_labels, pred_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                'input_ids': batch['input_ids'].to(Config.device),
                'attention_mask': batch['attention_mask'].to(Config.device)
            }
            _, logits = model(**inputs)
            preds = model.decode(logits, inputs['attention_mask'])
            
            labels = batch['labels'].cpu().numpy()[:, 1:-1]
            masks = (labels != -100) & (batch['attention_mask'].cpu().numpy()[:, 1:-1] == 1)
            
            for i in range(len(preds)):
                seq_true = [processor.labels[idx] for idx, valid in zip(labels[i], masks[i]) if valid]
                seq_pred = [processor.labels[idx] for idx, valid in zip(preds[i], masks[i]) if valid]
                if seq_true:
                    true_labels.append(seq_true)
                    pred_labels.append(seq_pred)
    
    return {
        'f1': f1_score(true_labels, pred_labels) if true_labels else 0.0,
        'report': classification_report(true_labels, pred_labels, zero_division=0) if true_labels else ""
    }

def train_fold(fold, train_loader, val_loader, processor, args):
    model = XLNetBiLSTMCRF(len(processor.labels)).to(Config.device)
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader)*args.epochs
    )
    
    best_f1 = 0
    best_model_path = os.path.join(args.output_dir, f"fold_XLNet_BILSTM_CRF_{fold}_best.pt")
    
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {
                'input_ids': batch['input_ids'].to(Config.device),
                'attention_mask': batch['attention_mask'].to(Config.device),
                'labels': batch['labels'].to(Config.device)
            }
            loss, _ = model(**inputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        results = evaluate(model, val_loader, processor)
        if results['f1'] > best_f1:
            best_f1 = results['f1']
            torch.save(model.state_dict(), best_model_path)
    
    model.load_state_dict(torch.load(best_model_path))
    final_results = evaluate(model, val_loader, processor)
    return final_results

def main():
    args = parse_args()
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'resultados_xlnet_bilstm_crf.txt'), mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    try:
        processor = DataProcessor()
        examples = processor.get_examples(args.data)
        tokenizer = XLNetTokenizer.from_pretrained(Config.model_name)
        kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=Config.seed)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(examples)):
            train_examples = [examples[i] for i in train_idx]
            val_examples = [examples[i] for i in val_idx]
            
            train_dataset = NERDataset(train_examples, tokenizer, processor)
            val_dataset = NERDataset(val_examples, tokenizer, processor)
            
            train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size)
            
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