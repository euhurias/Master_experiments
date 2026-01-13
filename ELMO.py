import sys
import os
import argparse
import logging
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torch.optim import AdamW
from allennlp.modules.elmo import Elmo, batch_to_ids
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Arquivo único com dados no formato IOB')
    parser.add_argument('--output_dir', required=True, help='Diretório para salvar resultados')
    parser.add_argument('--k_folds', type=int, default=5, help='Número de folds (padrão: 5)')
    parser.add_argument('--batch_size', type=int, default=16, help='Tamanho do batch (padrão: 16)')
    parser.add_argument('--epochs', type=int, default=15, help='Número de épocas (padrão: 15)')
    parser.add_argument('--patience', type=int, default=5, help='Paciência para early stopping')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Taxa de aprendizado')
    return parser.parse_args()

class Config:
    # Você precisa baixar os arquivos do ELMo para português
    # Exemplo: https://github.com/allenai/bilm-tf/tree/master/bin
    options_file = "elmo_pt_options.json"  # Ajuste o caminho
    weight_file = "elmo_pt_weights.hdf5"   # Ajuste o caminho
    max_seq_length = 180
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42

class ElmoNER(nn.Module):
    def __init__(self, num_labels, dropout=0.1):
        super().__init__()
        self.elmo = Elmo(Config.options_file, Config.weight_file, num_output_representations=1, dropout=0)
        self.embedding_dim = self.elmo.get_output_dim()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embedding_dim, num_labels)
        
    def forward(self, char_ids, attention_mask, labels=None):
        elmo_output = self.elmo(char_ids)
        embeddings = elmo_output['elmo_representations'][0]
        logits = self.classifier(self.dropout(embeddings))
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.size(-1))[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss
        
        return logits

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

def convert_examples_to_features(examples, processor):
    features = []
    for ex in examples:
        words = ex.words
        labels = ex.labels
        
        words = words[:Config.max_seq_length]
        labels = labels[:Config.max_seq_length]
        
        attention_mask = [1] * len(words)
        padding = Config.max_seq_length - len(words)
        
        if padding > 0:
            words = words + [''] * padding
            labels = labels + ['[PAD]'] * padding
            attention_mask = attention_mask + [0] * padding
        
        label_ids = [processor.label_map.get(label, processor.label_map['X']) for label in labels]
        
        features.append({
            'words': words,
            'attention_mask': attention_mask,
            'labels': label_ids,
            'predict_mask': [1 if word != '' else 0 for word in words]
        })
    return features

class NerDataset(data.Dataset):
    def __init__(self, examples, processor):
        self.features = convert_examples_to_features(examples, processor)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feat = self.features[idx]
        return {
            'words': feat['words'],
            'attention_mask': torch.tensor(feat['attention_mask']),
            'labels': torch.tensor(feat['labels']),
            'predict_mask': torch.tensor(feat['predict_mask'], dtype=torch.bool)
        }

def evaluate(model, dataloader, processor):
    model.eval()
    true_labels, pred_labels = [], []
    total_loss = 0
    loss_fct = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            words = batch['words']
            attention_mask = batch['attention_mask'].to(Config.device)
            labels = batch['labels'].to(Config.device)
            labels_batch = labels.cpu().numpy()
            masks_batch = batch['predict_mask'].cpu().numpy()
            
            char_ids = batch_to_ids(words).to(Config.device)
            logits = model(char_ids, attention_mask)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
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
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(true_labels, pred_labels) if true_labels else 0.0
    precision = precision_score(true_labels, pred_labels) if true_labels else 0.0
    recall = recall_score(true_labels, pred_labels) if true_labels else 0.0
    
    return {
        'loss': avg_loss,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'report': classification_report(true_labels, pred_labels, zero_division=0)
    }

def train_fold(fold, train_loader, val_loader, processor, args):
    model = ElmoNER(num_labels=len(processor.labels)).to(Config.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    best_f1 = -1
    patience_counter = 0
    best_model_path = os.path.join(args.output_dir, f"fold_ELMo_{fold}_best.pt")
    
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            words = batch['words']
            attention_mask = batch['attention_mask'].to(Config.device)
            labels = batch['labels'].to(Config.device)
            
            char_ids = batch_to_ids(words).to(Config.device)
            loss = model(char_ids, attention_mask, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        val_results = evaluate(model, val_loader, processor)
        epoch_time = time.time() - start_time
        
        logging.info(f"\n=== Fold {fold}, Época {epoch+1} ===")
        logging.info(f"⏱️  Tempo: {epoch_time:.2f}s")
        logging.info(f"📉 Loss Treino: {avg_train_loss:.4f}")
        logging.info(f"🎯 F1 Validação: {val_results['f1']:.4f}")
        
        if val_results['f1'] > best_f1:
            best_f1 = val_results['f1']
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"✅ NOVO MELHOR F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logging.info(f"🛑 Early stopping na época {epoch+1}")
                break
    
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    
    return evaluate(model, val_loader, processor)

def main():
    args = parse_args()
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'resultados_elmo.txt')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    try:
        logging.info("🚀 INICIANDO TREINAMENTO ELMo")
        
        processor = DataProcessor()
        examples = processor.get_examples(args.data)
        
        kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=Config.seed)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(examples)):
            logging.info(f"\n🔄 INICIANDO FOLD {fold+1}/{args.k_folds}")
            
            train_examples = [examples[i] for i in train_idx]
            val_examples = [examples[i] for i in val_idx]
            
            train_dataset = NerDataset(train_examples, processor)
            val_dataset = NerDataset(val_examples, processor)
            
            train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size)
            
            results = train_fold(fold+1, train_loader, val_loader, processor, args)
            fold_results.append(results['f1'])
        
        logging.info(f"\n🎯 RESULTADO FINAL")
        logging.info(f"📊 F1 Médio: {np.mean(fold_results):.4f} (±{np.std(fold_results):.4f})")
        
        with open(os.path.join(args.output_dir, 'metricas_finais_elmo.txt'), 'w') as f:
            f.write(f"F1 Médio: {np.mean(fold_results):.4f} (±{np.std(fold_results):.4f})\n")
            f.write(f"Valores por Fold: {[round(f, 4) for f in fold_results]}\n")
    
    except Exception as e:
        logging.error(f"❌ Erro: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()