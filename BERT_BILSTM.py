import sys
import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

class Config:
    model_name = "neuralmind/bert-base-portuguese-cased"
    max_seq_length = 180
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    batch_size = 16
    epochs = 15
    patience = 5
    learning_rate = 2e-5

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
        all_labels = []
        
        with open(data_file, 'r', encoding='utf-8') as f:
            entries = f.read().strip().split("\n\n")
            for i, entry in enumerate(entries):
                words, labels = [], []
                for line in entry.split('\n'):
                    if line.strip():
                        parts = line.strip().split()
                        words.append(parts[0])
                        label = parts[-1].upper().replace('X', 'O')  # Corrigir labels X
                        labels.append(label)
                        all_labels.append(label)
                examples.append(InputExample(i, words, labels))
        
        self._create_label_maps(all_labels)
        return examples
    
    def _create_label_maps(self, all_labels):
        unique_labels = sorted(list(set(all_labels)))
        self.labels = ['O']
        for lbl in unique_labels:
            if lbl != 'O' and lbl not in self.labels:
                self.labels.append(lbl)
        self.label_map = {label: i for i, label in enumerate(self.labels)}

def convert_examples_to_features(examples, tokenizer, processor):
    features = []
    for ex in examples:
        tokens = ['[CLS]']
        label_ids = [-100]  # [CLS]
        predict_mask = [0]
        
        for word, label in zip(ex.words, ex.labels):
            word_tokens = tokenizer.tokenize(word) or [tokenizer.unk_token]
            tokens.extend(word_tokens)
            
            if label not in processor.label_map:
                label = 'O'
            
            main_label = processor.label_map[label]
            label_ids.append(main_label)
            predict_mask.append(1)
            
            if len(word_tokens) > 1:
                label_ids.extend([-100] * (len(word_tokens) - 1))
                predict_mask.extend([0] * (len(word_tokens) - 1))
        
        max_length = Config.max_seq_length - 1
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            label_ids = label_ids[:max_length]
            predict_mask = predict_mask[:max_length]
        
        tokens.append('[SEP]')
        label_ids.append(-100)
        predict_mask.append(0)
        
        padding = Config.max_seq_length - len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens) + [0] * padding
        attention_mask = [1] * len(tokens) + [0] * padding
        label_ids += [-100] * padding
        predict_mask += [0] * padding
        
        features.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_ids,
            'predict_mask': predict_mask
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

class BERTBiLSTM(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(Config.model_name)
        self.lstm = nn.LSTM(
            768,
            256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        self.classifier = nn.Linear(512, num_labels)
        self.dropout = nn.Dropout(0.3)
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.classifier.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.01)
    
    def forward(self, input_ids, attention_mask, labels=None, predict_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        
        lstm_out, _ = self.lstm(sequence_output)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)
        
        loss = None
        if labels is not None:
            active_loss = (labels != -100) & (attention_mask == 1) & predict_mask
            active_logits = logits.view(-1, logits.size(-1))[active_loss.view(-1)]
            active_labels = labels.view(-1)[active_loss.view(-1)]
            
            if active_labels.numel() == 0:
                return torch.tensor(0.0, requires_grad=True), logits
            
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(active_logits, active_labels)
        
        return loss, logits

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
        'report': classification_report(true_labels, pred_labels, zero_division=0)
    }

def train_fold(fold, train_loader, val_loader, processor, args, class_weights):
    set_seed(Config.seed + fold)
    
    model = BERTBiLSTM(len(processor.labels)).to(Config.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train_loader)*0.1),
        num_training_steps=len(train_loader)*args.epochs
    )
    
    best_f1 = -float('inf')
    patience_counter = 0
    best_model_path = os.path.join(args.output_dir, f"fold_BERT_BILSTM_{fold}_best.pt")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {k: v.to(Config.device) for k, v in batch.items()}
            loss, _ = model(**inputs)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        results = evaluate(model, val_loader, processor)
        val_f1 = results['f1']
        
        logging.info(f"Fold {fold} - Epoch {epoch+1}")
        logging.info(f"Train Loss: {total_loss/len(train_loader):.4f}")
        logging.info(f"Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break
    
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    else:
        logging.warning(f"Modelo não encontrado em {best_model_path}")
    
    return evaluate(model, val_loader, processor)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
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
            logging.FileHandler(os.path.join(args.output_dir, 'resultados_bert_bilstm.txt')),
            logging.StreamHandler()
        ]
    )
    
    try:
        processor = DataProcessor()
        examples = processor.get_examples(args.data)
        tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
        
        all_labels = [lb for ex in examples for lb in ex.labels]
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(all_labels), 
            y=all_labels
        )
        class_weights = torch.FloatTensor(class_weights).to(Config.device)
        
        kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=Config.seed)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(examples)):
            logging.info(f"\n{'='*40}")
            logging.info(f"Fold {fold+1}")
            logging.info(f"{'='*40}")
            
            train_examples = [examples[i] for i in train_idx]
            val_examples = [examples[i] for i in val_idx]
            
            train_dataset = NERDataset(train_examples, tokenizer, processor)
            val_dataset = NERDataset(val_examples, tokenizer, processor)
            
            train_loader = data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True
            )
            val_loader = data.DataLoader(
                val_dataset,
                batch_size=args.batch_size
            )
            
            results = train_fold(fold+1, train_loader, val_loader, processor, args, class_weights)
            
            fold_results.append(results['f1'])
            logging.info(f"\nFold {fold+1} F1: {results['f1']:.4f}")
            logging.info("Relatório:")
            logging.info(results['report'])
        
        logging.info("\nResultado Final:")
        logging.info(f"F1 Médio: {np.mean(fold_results):.4f} (±{np.std(fold_results):.4f})")
        logging.info(f"Folds: {[round(f, 4) for f in fold_results]}")
    
    except Exception as e:
        logging.error(f"Erro crítico: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()