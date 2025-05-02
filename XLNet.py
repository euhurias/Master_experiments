import sys
import os
import argparse
import logging
import numpy as np
import torch
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
    # Removemos [CLS] e [SEP] das labels especiais
    special_labels = ['X']  # Apenas padding
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
        
        input_ids = [self.tokenizer.cls_token_id]  # Inicia com [CLS]
        label_ids = [self.label_map['X']]          # Label X para [CLS]
        predict_mask = [0]                         # Não prever [CLS]

        # Processa cada palavra e suas labels
        for word, label in zip(tokens, labels):
            word_tokens = self.tokenizer.tokenize(word) or [self.tokenizer.unk_token]
            input_ids.extend(self.tokenizer.convert_tokens_to_ids(word_tokens))
            
            # Primeiro subtoken recebe a label original
            label_ids.append(self.label_map[label])
            predict_mask.append(1)
            
            # Subtokens adicionais recebem X
            for _ in range(1, len(word_tokens)):
                label_ids.append(self.label_map['X'])
                predict_mask.append(0)

        # Adiciona [SEP] no final
        input_ids.append(self.tokenizer.sep_token_id)
        label_ids.append(self.label_map['X'])  # Label X para [SEP]
        predict_mask.append(0)                 # Não prever [SEP]

        # Trunca para o máximo permitido
        input_ids = input_ids[:self.max_len]
        label_ids = label_ids[:self.max_len]
        predict_mask = predict_mask[:self.max_len]

        # Padding
        padding = self.max_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding
        label_ids += [self.label_map['X']] * padding
        predict_mask += [0] * padding
        
        # Cria máscara de atenção
        attention_mask = [1] * (self.max_len - padding) + [0] * padding

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(label_ids),
            'predict_mask': torch.tensor(predict_mask)
        }

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
                
                # Processamento conjunto para manter alinhamento
                pairs = []
                for t, p, m in zip(batch['labels'][i].tolist(), preds[i], mask):
                    if m:  # Só considera posições marcadas para previsão
                        true_label = id2label[t]
                        pred_label = id2label[p]
                        if true_label != 'X' and pred_label != 'X':
                            pairs.append((true_label, pred_label))
                
                # Separa as labels apenas se houver pares válidos
                if pairs:
                    t, p = zip(*pairs)
                    true_labels.append(list(t))
                    pred_labels.append(list(p))
    
    # Verificação final de consistência
    assert len(true_labels) == len(pred_labels), "Número de exemplos diferente!"
    
    return {
        'f1': f1_score(true_labels, pred_labels),
        'report': classification_report(true_labels, pred_labels, zero_division=0)
    }

def train_fold(fold, train_loader, val_loader, label_map, args):
    model = XLNetForTokenClassification.from_pretrained(
        Config.model_name,
        num_labels=len(label_map)
    ).to(Config.device)
    
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader)*args.epochs
    )
    
    best_f1 = 0
    patience = 0
    
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            inputs = {k: v.to(Config.device) for k, v in batch.items() if k != 'predict_mask'}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        results = evaluate(model, val_loader, label_map)
        if results['f1'] > best_f1:
            best_f1 = results['f1']
            patience = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'fold_XLNet_{fold}_best.pt'))
        else:
            patience += 1
            if patience >= args.patience:
                break
    
    model.load_state_dict(torch.load(os.path.join(args.output_dir, f'fold_XLNet_{fold}_best.pt')))
    return evaluate(model, val_loader, label_map)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--patience', type=int, default=3)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'resultados_xlnet.txt')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    processor = DataProcessor()
    examples = processor.get_examples(args.data)
    label_map = create_label_map(examples)
    tokenizer = XLNetTokenizer.from_pretrained(Config.model_name)
    
    # Define collate function
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        predict_mask = [item['predict_mask'] for item in batch]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
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
        
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
        
        results = train_fold(fold+1, train_loader, val_loader, label_map, args)
        
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