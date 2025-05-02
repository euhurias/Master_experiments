import sys
import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.optim import AdamW
from transformers import XLNetForTokenClassification, XLNetTokenizer, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

# Configurações determinísticas
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

class Config:
    model_name = "xlnet-base-cased"
    max_seq_length = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    batch_size = 16
    epochs = 15  # Aumentado para dar mais tempo de treino
    patience = 5  # Aumentado para evitar parada prematura

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
                        label = parts[-1].upper().replace('X', 'O')
                        labels.append(label)
                        all_labels.add(label)
                if words:
                    examples.append(InputExample(i, words, labels))
        
        self._create_label_maps(all_labels)
        return examples
    
    def _create_label_maps(self, labels):
        self.labels = ['O']
        for label in sorted(labels):
            if label != 'O' and label not in self.labels:
                self.labels.append(label)
        self.label_map = {label: i for i, label in enumerate(self.labels)}

def convert_examples_to_features(examples, tokenizer, processor):
    features = []
    for ex in examples:
        tokens, labels = [], []
        for word, label in zip(ex.words, ex.labels):
            word_tokens = tokenizer.tokenize(word) or [tokenizer.unk_token]
            tokens.extend(word_tokens)
            
            if label not in processor.label_map:
                label = 'O'
            labels.extend([processor.label_map[label]] + [-100]*(len(word_tokens)-1))
        
        tokens = tokens[:Config.max_seq_length-2]
        labels = labels[:Config.max_seq_length-2]
        
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        labels = [-100] + labels + [-100]
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        padding = Config.max_seq_length - len(input_ids)
        
        predict_mask = [0]*len(input_ids)
        for i in range(1, len(labels)-1):
            if labels[i] != -100:
                predict_mask[i] = 1
                
        features.append({
            'input_ids': input_ids + [0]*padding,
            'attention_mask': [1]*len(input_ids) + [0]*padding,
            'labels': labels + [-100]*padding,
            'predict_mask': predict_mask + [0]*padding
        })
    return features

class NERDataset(data.Dataset):
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

class XLNetBiLSTM(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.xlnet = XLNetForTokenClassification.from_pretrained(
            Config.model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        self.lstm = nn.LSTM(
            input_size=self.xlnet.config.hidden_size,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        self.classifier = nn.Linear(512, num_labels)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, input_ids, attention_mask, labels=None, predict_mask=None):
        outputs = self.xlnet(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        sequence = outputs.hidden_states[-1]
        
        lstm_out, _ = self.lstm(sequence)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)
        
        if labels is not None:
            active_loss = predict_mask.view(-1) & (labels.view(-1) != -100)
            active_logits = logits.view(-1, logits.size(-1))[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            if active_labels.numel() == 0:
                return torch.tensor(0.0, requires_grad=True), logits
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(active_logits, active_labels)
            return loss, logits
        
        return None, logits

def evaluate(model, dataloader, processor):
    model.eval()
    true_labels, pred_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(Config.device) for k, v in batch.items()}
            _, logits = model(**inputs)
            
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            masks = batch['predict_mask'].cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            
            for i in range(len(preds)):
                current_true = []
                current_pred = []
                for j in range(len(masks[i])):
                    if masks[i][j] and labels[i][j] != -100:
                        current_true.append(processor.labels[labels[i][j]])
                        current_pred.append(processor.labels[preds[i][j]])
                
                if current_true:
                    true_labels.append(current_true)
                    pred_labels.append(current_pred)
    
    return {
        'f1': f1_score(true_labels, pred_labels) if true_labels else 0.0,
        'report': classification_report(true_labels, pred_labels, zero_division=0) if true_labels else ""
    }

def train_fold(fold, train_loader, val_loader, processor, args):
    set_seed(Config.seed + fold)  # Semente diferente por fold
    
    model = XLNetBiLSTM(len(processor.labels)).to(Config.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train_loader)*0.1),
        num_training_steps=len(train_loader)*args.epochs
    )
    
    best_f1 = -float('inf')
    best_model_path = os.path.join(args.output_dir, f"fold_XLNet_BILSTM_{fold}_best.pt")
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {k: v.to(Config.device) for k, v in batch.items()}
            loss, _ = model(**inputs)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        
        results = evaluate(model, val_loader, processor)
        
        # Early stopping com tolerância
        if results['f1'] > best_f1 + 1e-4:
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    args = parser.parse_args()

    set_seed(Config.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'resultados_xlnet_bilstm.txt')),
            logging.StreamHandler()
        ]
    )
    
    try:
        processor = DataProcessor()
        examples = processor.get_examples(args.data)
        tokenizer = XLNetTokenizer.from_pretrained(Config.model_name)
        
        kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=Config.seed)
        fold_results = []
        fold_reports = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(examples)):
            logging.info(f"\n{'='*40}")
            logging.info(f"Fold {fold+1}")
            logging.info(f"{'='*40}")
            
            train_examples = [examples[i] for i in train_idx]
            val_examples = [examples[i] for i in val_idx]
            
            # Balanceamento por fold
            all_labels = [lb for ex in train_examples for lb in ex.labels]
            class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
            class_weights = torch.FloatTensor(class_weights).to(Config.device)
            
            train_dataset = NERDataset(train_examples, tokenizer, processor)
            val_dataset = NERDataset(val_examples, tokenizer, processor)
            
            train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size)
            
            results = train_fold(fold+1, train_loader, val_loader, processor, args)
            
            fold_results.append(results['f1'])
            fold_reports.append(results['report'])
            
            logging.info(f"\nFold {fold+1} Resultados:")
            logging.info(f"F1: {results['f1']:.4f}")
            logging.info("Relatório Detalhado:")
            logging.info(results['report'])
        
        logging.info("\n\n" + "="*40)
        logging.info("Resultado Final da Validação Cruzada")
        logging.info("="*40)
        for i, (f1, report) in enumerate(zip(fold_results, fold_reports)):
            logging.info(f"\nFold {i+1}: F1 = {f1:.4f}")
            logging.info(report)
        
        logging.info("\nResumo Final:")
        logging.info(f"Média F1: {np.mean(fold_results):.4f}")
        logging.info(f"Desvio Padrão: {np.std(fold_results):.4f}")
        logging.info(f"Valores por Fold: {[round(f, 4) for f in fold_results]}")
    
    except Exception as e:
        logging.error(f"Erro: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()