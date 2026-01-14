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

class Config:
    model_name = 'xlnet-base-cased'
    max_seq_length = 180
    batch_size = 16
    learning_rate = 3e-5
    total_train_epochs = 15
    k_folds = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 44
    patience = 3
    dropout_rate = 0.1

class InputExample:
    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels

class DataProcessor:
    @classmethod
    def _read_data(cls, input_file):
        with open(input_file) as f:
            return f.read().strip().split("\n\n")
    
    def get_examples(self, data_file):
        examples = []
        entries = self._read_data(data_file)
        for i, entry in enumerate(entries):
            words, labels = [], []
            for line in entry.split('\n'):
                if line.strip():
                    parts = line.strip().split()
                    words.append(parts[0])
                    labels.append(parts[-1])
            examples.append(InputExample(i, words, labels))
        return examples

def create_label_map(examples):
    all_labels = set()
    for ex in examples:
        all_labels.update(ex.labels)
    special_labels = ['X']
    labels = special_labels + sorted([l for l in all_labels if l not in special_labels])
    return {label: i for i, label in enumerate(labels)}

class NerDataset(data.Dataset):
    def __init__(self, examples, tokenizer, label_map, max_len):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = ex.words
        labels = ex.labels
        
        input_ids = [self.tokenizer.cls_token_id]
        label_ids = [self.label_map['X']]
        predict_mask = [0]

        for word, label in zip(tokens, labels):
            word_tokens = self.tokenizer.tokenize(word) or [self.tokenizer.unk_token]
            input_ids.extend(self.tokenizer.convert_tokens_to_ids(word_tokens))
            
            label_ids.append(self.label_map[label])
            predict_mask.append(1)
            
            for _ in range(1, len(word_tokens)):
                label_ids.append(self.label_map['X'])
                predict_mask.append(0)

        input_ids.append(self.tokenizer.sep_token_id)
        label_ids.append(self.label_map['X'])
        predict_mask.append(0)

        input_ids = input_ids[:self.max_len]
        label_ids = label_ids[:self.max_len]
        predict_mask = predict_mask[:self.max_len]

        padding = self.max_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding
        label_ids += [self.label_map['X']] * padding
        predict_mask += [0] * padding
        
        attention_mask = [1] * (self.max_len - padding) + [0] * padding

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(label_ids),
            'predict_mask': torch.tensor(predict_mask)
        }

class XLNetCNN(nn.Module):
    def __init__(self, base_model, num_labels, dropout_rate=0.1, cnn_filters=128, cnn_kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.xlnet = base_model
        hidden_size = self.xlnet.config.hidden_size
        
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, cnn_filters, ks, padding=ks//2)
            for ks in cnn_kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(cnn_filters * len(cnn_kernel_sizes), num_labels)
        self.num_labels = num_labels
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.xlnet(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        sequence_output = outputs.hidden_states[-1]
        
        x = sequence_output.transpose(1, 2)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))
            conv_out = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(conv_out)
        
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            
        return {'loss': loss, 'logits': logits}

def evaluate(model, dataloader, label_map):
    model.eval()
    true_labels, pred_labels = [], []
    id2label = {v: k for k, v in label_map.items()}
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(Config.device) for k, v in batch.items() if k != 'predict_mask'}
            outputs = model(**inputs)
            logits = outputs.logits
            preds = logits.argmax(-1).cpu().numpy()
            
            for i in range(len(preds)):
                mask = batch['predict_mask'][i].numpy().astype(bool)
                
                pairs = []
                for t, p, m in zip(batch['labels'][i].tolist(), preds[i], mask):
                    if m:
                        true_label = id2label[t]
                        pred_label = id2label[p]
                        if true_label != 'X' and pred_label != 'X':
                            pairs.append((true_label, pred_label))
                
                if pairs:
                    t, p = zip(*pairs)
                    true_labels.append(list(t))
                    pred_labels.append(list(p))
    
    return {
        'f1': f1_score(true_labels, pred_labels),
        'report': classification_report(true_labels, pred_labels, zero_division=0)
    }

def train_fold(fold, train_loader, val_loader, label_map, args):
    base_model = XLNetForTokenClassification.from_pretrained(
        Config.model_name,
        num_labels=len(label_map),
        output_hidden_states=True
    )
    model = XLNetCNN(base_model.xlnet, len(label_map))
    model.to(Config.device)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader)*args.epochs
    )
    
    best_f1 = 0
    patience = 0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            inputs = {k: v.to(Config.device) for k, v in batch.items() if k != 'predict_mask'}
            outputs = model(**inputs)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        results = evaluate(model, val_loader, label_map)
        
        print(f"Fold {fold}, Epoch {epoch+1}: Loss={avg_loss:.4f}, F1={results['f1']:.4f}")
        
        if results['f1'] > best_f1:
            best_f1 = results['f1']
            patience = 0
            torch.save(model.state_dict(), 
                      os.path.join(args.output_dir, f'fold_xlnet_cnn_{fold}_best.pt'))
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(torch.load(
        os.path.join(args.output_dir, f'fold_xlnet_cnn_{fold}_best.pt')))
    return evaluate(model, val_loader, label_map)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--max_seq_length', type=int, default=180)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'resultados_xlnet_cnn.txt')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    Config.batch_size = args.batch_size
    Config.max_seq_length = args.max_seq_length
    
    processor = DataProcessor()
    examples = processor.get_examples(args.data)
    label_map = create_label_map(examples)
    tokenizer = XLNetTokenizer.from_pretrained(Config.model_name)
    
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        predict_mask = [item['predict_mask'] for item in batch]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, 
                                                   padding_value=tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=label_map['X'])
        predict_mask = torch.nn.utils.rnn.pad_sequence(predict_mask, batch_first=True, padding_value=0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'predict_mask': predict_mask
        }
    
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=Config.seed)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(examples)):
        train_examples = [examples[i] for i in train_idx]
        val_examples = [examples[i] for i in val_idx]
        
        train_dataset = NerDataset(train_examples, tokenizer, label_map, Config.max_seq_length)
        val_dataset = NerDataset(val_examples, tokenizer, label_map, Config.max_seq_length)
        
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                      shuffle=True, collate_fn=collate_fn)
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, 
                                    collate_fn=collate_fn)
        
        print(f"\n=== Treinando Fold {fold+1}/{args.k_folds} ===")
        print(f"Modelo: XLNet + CNN")
        print(f"Exemplos treino: {len(train_examples)}, validação: {len(val_examples)}")
        
        results = train_fold(fold+1, train_loader, val_loader, label_map, args)
        
        logging.info(f"\n=== Fold {fold+1} ===")
        logging.info(f"F1-Score: {results['f1']:.4f}")
        logging.info("Classification Report:")
        logging.info(results['report'])
        
        fold_results.append(results['f1'])
    
    logging.info("\n=== Resultados Finais ===")
    logging.info(f"Modelo: XLNet + CNN")
    logging.info(f"F1 Médio: {np.mean(fold_results):.4f} (±{np.std(fold_results):.4f})")
    logging.info(f"Valores por Fold: {[round(f, 4) for f in fold_results]}")
    
    with open(os.path.join(args.output_dir, 'config.txt'), 'w') as f:
        f.write(str(vars(args)))

if __name__ == "__main__":
    main()