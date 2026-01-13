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
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import torch.nn.functional as F
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
    model_name = "neuralmind/bert-base-portuguese-cased"
    max_seq_length = 180
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42

class BertIDCNN(nn.Module):
    def __init__(self, model_name, num_filters, kernel_size, num_blocks, num_labels, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.embedding_dim = self.bert.config.hidden_size
        
        self.conv_blocks = nn.ModuleList()
        dilation = 1
        for _ in range(num_blocks):
            self.conv_blocks.append(
                nn.Conv1d(num_filters, num_filters, kernel_size, 
                         padding=(kernel_size-1)*dilation//2, dilation=dilation)
            )
            dilation *= 2
        
        self.initial_conv = nn.Conv1d(self.embedding_dim, num_filters, kernel_size, padding=(kernel_size-1)//2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        sequence_output = sequence_output.transpose(1, 2)
        x = F.relu(self.initial_conv(sequence_output))
        
        for conv in self.conv_blocks:
            x = F.relu(conv(x))
            
        x = x.transpose(1, 2)
        logits = self.classifier(self.dropout(x))
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.size(-1))[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss
        
        return logits

# As classes auxiliares (InputExample, DataProcessor, etc.) são as mesmas do BERT+CNN
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
    total_loss = 0
    loss_fct = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(Config.device)
            attention_mask = batch['attention_mask'].to(Config.device)
            labels = batch['labels'].to(Config.device)
            labels_batch = labels.cpu().numpy()
            masks_batch = batch['predict_mask'].cpu().numpy()
            
            logits = model(input_ids, attention_mask)
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
    model = BertIDCNN(
        Config.model_name,
        num_filters=100,
        kernel_size=3,
        num_blocks=4,
        num_labels=len(processor.labels)
    ).to(Config.device)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) // 2,
        num_training_steps=len(train_loader) * args.epochs
    )
    
    best_f1 = -1
    patience_counter = 0
    best_model_path = os.path.join(args.output_dir, f"fold_BERT_IDCNN_{fold}_best.pt")
    
    logging.info(f"🔧 Configuração BERT+IDCNN:")
    logging.info(f"   - Model: {Config.model_name}")
    logging.info(f"   - Num Filters: 100")
    logging.info(f"   - Kernel Size: 3")
    logging.info(f"   - Num Blocks: 4")
    logging.info(f"   - Num Labels: {len(processor.labels)}")
    
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
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
            batch_count += 1
            
            if (batch_idx + 1) % max(1, len(train_loader) // 10) == 0:
                current_loss = total_loss / batch_count
                progress = (batch_idx + 1) / len(train_loader) * 100
                logging.info(f"Fold {fold}, Época {epoch+1}, Progresso: {progress:.0f}% | Loss = {current_loss:.4f}")
        
        avg_train_loss = total_loss / len(train_loader)
        
        val_results = evaluate(model, val_loader, processor)
        epoch_time = time.time() - start_time
        
        logging.info(f"\n=== Fold {fold}, Época {epoch+1} ===")
        logging.info(f"⏱️  Tempo: {epoch_time:.2f}s")
        logging.info(f"📉 Loss Treino: {avg_train_loss:.4f}")
        logging.info(f"📊 Loss Validação: {val_results['loss']:.4f}")
        logging.info(f"🎯 F1 Validação: {val_results['f1']:.4f}")
        
        if val_results['f1'] > best_f1:
            best_f1 = val_results['f1']
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"✅ NOVO MELHOR F1: {best_f1:.4f} (modelo salvo)")
        else:
            patience_counter += 1
            logging.info(f"⏳ F1 não melhorou. Paciência: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                logging.info(f"🛑 Early stopping na época {epoch+1}")
                break
        
        logging.info("-" * 80)
    
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logging.info(f"✅ Fold {fold}: Melhor modelo carregado com F1 = {best_f1:.4f}")
    
    final_results = evaluate(model, val_loader, processor)
    
    logging.info(f"\n📊 RESUMO FOLD {fold}")
    logging.info(f"🎯 Melhor F1: {best_f1:.4f}")
    logging.info(f"📈 F1 Final: {final_results['f1']:.4f}")
    
    return final_results

def main():
    args = parse_args()
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'resultados_bert_idcnn.txt')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    try:
        logging.info("🚀 INICIANDO TREINAMENTO BERT+IDCNN")
        logging.info(f"📁 Dados: {args.data}")
        logging.info(f"📂 Output: {args.output_dir}")
        
        processor = DataProcessor()
        examples = processor.get_examples(args.data)
        tokenizer = BertTokenizer.from_pretrained(Config.model_name)
        
        kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=Config.seed)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(examples)):
            logging.info(f"\n🔄 INICIANDO FOLD {fold+1}/{args.k_folds}")
            
            train_examples = [examples[i] for i in train_idx]
            val_examples = [examples[i] for i in val_idx]
            
            train_dataset = NerDataset(train_examples, tokenizer, processor)
            val_dataset = NerDataset(val_examples, tokenizer, processor)
            
            train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size)
            
            results = train_fold(fold+1, train_loader, val_loader, processor, args)
            fold_results.append(results['f1'])
        
        logging.info(f"\n🎯 RESULTADO FINAL")
        logging.info(f"📊 F1 Médio: {np.mean(fold_results):.4f} (±{np.std(fold_results):.4f})")
        
        with open(os.path.join(args.output_dir, 'metricas_finais_bert_idcnn.txt'), 'w') as f:
            f.write(f"F1 Médio: {np.mean(fold_results):.4f} (±{np.std(fold_results):.4f})\n")
            f.write(f"Valores por Fold: {[round(f, 4) for f in fold_results]}\n")
    
    except Exception as e:
        logging.error(f"❌ Erro: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()