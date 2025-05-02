import sys
import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torchcrf import CRF
from seqeval.metrics import classification_report, f1_score
from sklearn.model_selection import KFold

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

class Config:
    model_name = "neuralmind/bert-base-portuguese-cased"
    max_seq_length = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    batch_size = 8
    learning_rate = 3e-5
    epochs = 15
    patience = 3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--k_folds', type=int, default=5)
    return parser.parse_args()

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
                if words and any(lbl != 'O' for lbl in labels):
                    examples.append(InputExample(i, words, labels))
        
        self._create_label_maps(all_labels)
        return examples
    
    def _create_label_maps(self, labels):
        unique_labels = sorted(list(labels))
        self.labels = ['O'] + [lbl for lbl in unique_labels if lbl != 'O']
        self.label_map = {label: i for i, label in enumerate(self.labels)}

def convert_examples_to_features(examples, tokenizer, processor):
    features = []
    for ex in examples:
        tokens = []
        label_ids = []
        
        for word, label in zip(ex.words, ex.labels):
            word_tokens = tokenizer.tokenize(word) or [tokenizer.unk_token]
            tokens.extend(word_tokens)
            
            clean_label = label if label in processor.label_map else 'O'
            main_label = processor.label_map[clean_label]
            label_ids.append(main_label)
            
            if len(word_tokens) > 1:
                label_ids.extend([-100] * (len(word_tokens) - 1))

        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        label_ids = [-100] + label_ids + [-100]

        max_length = Config.max_seq_length
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            label_ids = label_ids[:max_length]
        else:
            pad_len = max_length - len(tokens)
            tokens += [tokenizer.pad_token] * pad_len
            label_ids += [-100] * pad_len

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1 if token != tokenizer.pad_token else 0 for token in tokens]
        
        features.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_ids
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
            'input_ids': torch.tensor(feat['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(feat['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(feat['labels'], dtype=torch.long)
        }

class BertBiLSTMCRF(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(Config.model_name)
        self.lstm = nn.LSTM(
            self.bert.config.hidden_size,
            256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        self.classifier = nn.Linear(512, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name: nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name: nn.init.orthogonal_(param)
            elif 'bias' in name: nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        nn.init.normal_(self.crf.start_transitions, 0, 0.1)
        nn.init.normal_(self.crf.end_transitions, 0, 0.1)
        nn.init.normal_(self.crf.transitions, 0, 0.1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        lstm_out, _ = self.lstm(sequence_output)
        logits = self.classifier(lstm_out)
        
        logits = logits[:, 1:-1, :]  # Remove [CLS] e [SEP]
        mask = attention_mask[:, 1:-1].bool()
        
        if labels is not None:
            labels = labels[:, 1:-1]
            labels = labels.masked_fill(labels == -100, 0)
            
            # Verificação final de segurança
            assert torch.all(labels < len(self.crf.transitions)), "Índice de label inválido detectado"
            
            loss = -self.crf(logits, labels, mask=mask, reduction='mean')
            return loss
        
        return self.crf.decode(logits, mask=mask)

def evaluate(model, dataloader, processor):
    model.eval()
    true_labels, pred_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                'input_ids': batch['input_ids'].to(Config.device),
                'attention_mask': batch['attention_mask'].to(Config.device)
            }
            preds = model(**inputs)
            
            labels = batch['labels'].cpu().numpy()[:, 1:-1]
            masks = (labels != -100) & (batch['attention_mask'].cpu().numpy()[:, 1:-1] == 1)
            
            for i in range(len(preds)):
                current_true = []
                current_pred = []
                for j in range(len(masks[i])):
                    if masks[i][j] and labels[i][j] != -100:
                        true_label = processor.labels[labels[i][j]]
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
    set_seed(Config.seed + fold)
    
    model = BertBiLSTMCRF(len(processor.labels)).to(Config.device)
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader)*Config.epochs
    )
    
    best_f1 = 0
    best_model_path = os.path.join(args.output_dir, f"fold_BERT_BILSTM_CRF_{fold}_best.pt")
    
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
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
            
            total_loss += loss.item()
        
        results = evaluate(model, val_loader, processor)
        val_f1 = results['f1']
        
        logging.info(f"Fold {fold} - Epoch {epoch+1}")
        logging.info(f"Train Loss: {total_loss/len(train_loader):.4f}")
        logging.info(f"Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
    
    model.load_state_dict(torch.load(best_model_path))
    return evaluate(model, val_loader, processor)

def main():
    args = parse_args()
    set_seed(Config.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'resultados_bert_bilstm_crf.txt')),
            logging.StreamHandler()
        ]
    )
    
    try:
        processor = DataProcessor()
        examples = processor.get_examples(args.data)
        tokenizer = BertTokenizer.from_pretrained(Config.model_name)
        
        # Debug: Verificar mapeamento de labels
        logging.info("\nLabels mapeadas:")
        for label, idx in processor.label_map.items():
            logging.info(f"{label}: {idx}")
        
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
                batch_size=Config.batch_size,
                shuffle=True,
                drop_last=True
            )
            val_loader = data.DataLoader(
                val_dataset,
                batch_size=Config.batch_size
            )
            
            results = train_fold(fold+1, train_loader, val_loader, processor, args)
            
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