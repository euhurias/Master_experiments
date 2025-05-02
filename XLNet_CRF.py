import sys
import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils import data
from transformers import XLNetModel, XLNetTokenizer
from torchcrf import CRF
from seqeval.metrics import classification_report, f1_score
from sklearn.model_selection import KFold

class Config:
    max_seq_length = 128
    batch_size = 16
    learning_rate = 2e-5
    epochs = 15
    model_name = 'xlnet-base-cased'
    dropout_rate = 0.3
    max_grad_norm = 1.0
    seed = 42
    patience = 3
    k_folds = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InputExample:
    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels

class DataProcessor:
    def __init__(self):
        self.special_labels = ['[PAD]', '[CLS]', '[SEP]', 'X']  # Ordem modificada
        self.label_map = {}
        self.id_to_label = {}

    def get_examples(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            entries = f.read().strip().split("\n\n")
        
        all_labels = set()
        examples = []
        for i, entry in enumerate(entries):
            words, labels = [], []
            for line in entry.split('\n'):
                if line.strip():
                    parts = line.strip().split()
                    words.append(parts[0])
                    label = parts[-1].upper() if len(parts) > 1 else 'O'
                    labels.append(label)
                    all_labels.add(label)
            examples.append(InputExample(i, words, labels))
        
        self._create_label_maps(all_labels)
        return examples

    def _create_label_maps(self, all_labels):
        # Primeiro mapeia labels especiais
        self.label_map = {label: i for i, label in enumerate(self.special_labels)}
        
        # Depois adiciona labels reais
        real_labels = sorted(all_labels.difference(set(self.special_labels)))
        for label in real_labels:
            self.label_map[label] = len(self.label_map)
        
        # Cria mapeamento inverso
        self.id_to_label = {v: k for k, v in self.label_map.items()}

def convert_example_to_feature(example, tokenizer, processor):
    tokens, labels, predict_mask = [], [], []
    for word, label in zip(example.words, example.labels):
        word_tokens = tokenizer.tokenize(word) or [tokenizer.unk_token]
        tokens.extend(word_tokens)
        
        # Usar 'X' apenas para tokens adicionais
        main_label = processor.label_map.get(label, processor.label_map['X'])
        labels.append(main_label)
        predict_mask.append(1)
        
        if len(word_tokens) > 1:
            labels.extend([processor.label_map['X']] * (len(word_tokens)-1))
            predict_mask.extend([0] * (len(word_tokens)-1))

    max_len = Config.max_seq_length - 2
    tokens = tokens[:max_len]
    labels = labels[:max_len]
    predict_mask = predict_mask[:max_len]

    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    labels = [processor.label_map['[CLS]']] + labels + [processor.label_map['[SEP]']]
    predict_mask = [0] + predict_mask + [0]

    pad_len = Config.max_seq_length - len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.pad_token_id]*pad_len
    attention_mask = [1]*len(tokens) + [0]*pad_len
    labels += [processor.label_map['[PAD]']]*pad_len
    predict_mask += [0]*pad_len

    return {
        'input_ids': input_ids[:Config.max_seq_length],
        'attention_mask': attention_mask[:Config.max_seq_length],
        'labels': labels[:Config.max_seq_length],
        'predict_mask': predict_mask[:Config.max_seq_length]
    }

class NERDataset(data.Dataset):
    def __init__(self, examples, tokenizer, processor):
        self.features = [convert_example_to_feature(ex, tokenizer, processor) for ex in examples]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

def collate_fn(batch):
    return {
        'input_ids': torch.LongTensor([x['input_ids'] for x in batch]),
        'attention_mask': torch.LongTensor([x['attention_mask'] for x in batch]),
        'labels': torch.LongTensor([x['labels'] for x in batch]),
        'predict_mask': torch.BoolTensor([x['predict_mask'] for x in batch])
    }

class XLNetCRF(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.xlnet = XLNetModel.from_pretrained(Config.model_name)
        self.dropout = nn.Dropout(Config.dropout_rate)
        self.classifier = nn.Linear(self.xlnet.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None, predict_mask=None):
        outputs = self.xlnet(input_ids, attention_mask=attention_mask)
        sequence = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence)[:, 1:-1, :]  # Remove [CLS]/[SEP]
        
        if labels is not None:
            labels = labels[:, 1:-1]
            mask = predict_mask[:, 1:-1].bool()
            
            # Verificação de segurança
            assert emissions.size(1) == labels.size(1), f"Emissions: {emissions.shape}, Labels: {labels.shape}"
            assert emissions.size(1) == mask.size(1), f"Emissions: {emissions.shape}, Mask: {mask.shape}"
            
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        return emissions

    def decode(self, emissions, mask):
        return self.crf.decode(emissions, mask=mask[:, 1:-1].bool())

def evaluate(model, dataloader, processor):
    model.eval()
    true_labels, pred_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                'input_ids': batch['input_ids'].to(Config.device),
                'attention_mask': batch['attention_mask'].to(Config.device),
                'predict_mask': batch['predict_mask'].to(Config.device)
            }
            emissions = model(**inputs)
            preds = model.decode(emissions, inputs['predict_mask'])
            
            labels = batch['labels'].cpu().numpy()[:, 1:-1]
            mask = batch['predict_mask'].cpu().numpy()[:, 1:-1]
            
            for i in range(len(preds)):
                seq_true = []
                seq_pred = []
                valid_length = min(len(preds[i]), len(mask[i]))  # Corrige o limite
                for j in range(valid_length):
                    if mask[i][j]:
                        true_label = processor.id_to_label.get(labels[i][j], 'X')
                        pred_label = processor.id_to_label.get(preds[i][j], 'X')
                        seq_true.append(true_label)
                        seq_pred.append(pred_label)
                if seq_true:
                    true_labels.append(seq_true)
                    pred_labels.append(seq_pred)
    
    return {
        'f1': f1_score(true_labels, pred_labels) if true_labels else 0.0,
        'report': classification_report(true_labels, pred_labels, zero_division=0)
    }

def train_fold(fold, train_loader, val_loader, processor, args):
    model = XLNetCRF(len(processor.label_map)).to(Config.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    best_f1 = 0
    patience = 0
    best_model_path = os.path.join(args.output_dir, f"fold_XLNet_CRF_{fold}_best.pt")
    
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(Config.device),
                'attention_mask': batch['attention_mask'].to(Config.device),
                'labels': batch['labels'].to(Config.device),
                'predict_mask': batch['predict_mask'].to(Config.device)
            }
            
            loss = model(**inputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        results = evaluate(model, val_loader, processor)
        if results['f1'] > best_f1:
            best_f1 = results['f1']
            patience = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience += 1
            if patience >= args.patience:
                break
    
    model.load_state_dict(torch.load(best_model_path))
    return evaluate(model, val_loader, processor)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=Config.learning_rate)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'resultados_xlnet_crf.txt')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    processor = DataProcessor()
    examples = processor.get_examples(args.data)
    tokenizer = XLNetTokenizer.from_pretrained(Config.model_name)
    
    # Debug: mostrar mapeamento de labels
    logging.info(f"Label mapping: {processor.label_map}")
    logging.info(f"Number of classes: {len(processor.label_map)}")
    
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=Config.seed)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(examples)):
        train_examples = [examples[i] for i in train_idx]
        val_examples = [examples[i] for i in val_idx]
        
        train_dataset = NERDataset(train_examples, tokenizer, processor)
        val_dataset = NERDataset(val_examples, tokenizer, processor)
        
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        val_loader = data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn
        )
        
        results = train_fold(fold+1, train_loader, val_loader, processor, args)
        
        logging.info(f"\n=== Fold {fold+1} ===")
        logging.info(f"F1-Score: {results['f1']:.4f}")
        logging.info("Classification Report:")
        logging.info(results['report'])
        
        fold_results.append(results['f1'])
    
    logging.info("\n=== Final Results ===")
    logging.info(f"F1 Médio: {np.mean(fold_results):.4f} (±{np.std(fold_results):.4f})")
    logging.info(f"Valores por Fold: {[round(f, 4) for f in fold_results]}")

if __name__ == "__main__":
    main()