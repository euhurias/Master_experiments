import sys
import os
import argparse
import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torch.optim import AdamW
from transformers import ElectraModel, ElectraTokenizer, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import time
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Arquivo com dados no formato IOB')
    parser.add_argument('--output_dir', required=True, help='Diretório para salvar resultados')
    parser.add_argument('--model_name', type=str, default='google/electra-base-discriminator')
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--cnn_num_filters', type=int, default=256)
    parser.add_argument('--cnn_kernel_sizes', type=str, default='3,5,7', help='Tamanhos de kernel para CNN separados por vírgula')
    return parser.parse_args()

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    logging_steps = 50
    max_grad_norm = 1.0

class ElectraCNN(nn.Module):
    def __init__(self, model_name, num_labels, num_filters=256, kernel_sizes=[3,5,7], dropout=0.1):
        super().__init__()
        self.electra = ElectraModel.from_pretrained(model_name)
        self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
        hidden_size = self.electra.config.hidden_size
        
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=num_filters,
                kernel_size=k,
                padding=k//2
            ) for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.layer_norm = nn.LayerNorm(num_filters * len(kernel_sizes))
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        conv_input = sequence_output.transpose(1, 2)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(conv_input))
            conv_outputs.append(conv_out)
        
        conv_output = torch.cat(conv_outputs, dim=1)
        conv_output = conv_output.transpose(1, 2)
        
        conv_output = self.layer_norm(conv_output)
        conv_output = self.dropout(conv_output)
        logits = self.classifier(conv_output)
        
        if labels is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.shape[-1])[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            if active_labels.numel() > 0:
                loss = self.loss_fct(active_logits, active_labels)
                return loss, logits
            else:
                return torch.tensor(0.0, device=logits.device, requires_grad=True), logits
        
        return None, logits

class InputExample:
    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels

class DataProcessor:
    def __init__(self):
        self.label_map = {}
        self.labels = []
        self.reverse_label_map = {}
    
    def get_examples(self, data_file):
        examples = []
        all_labels = set()
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Arquivo de dados não encontrado: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            entries = f.read().strip().split("\n\n")
            
            for i, entry in enumerate(entries):
                words, labels = [], []
                for line in entry.split('\n'):
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            words.append(parts[0])
                            label = parts[-1].upper()
                            labels.append(label)
                            all_labels.add(label)
                if words:
                    examples.append(InputExample(i, words, labels))
        
        self._create_label_maps(all_labels)
        return examples
    
    def _create_label_maps(self, labels):
        special_labels = ['[PAD]', '[CLS]', '[SEP]', 'X', 'O']
        non_special = sorted([l for l in labels if l not in special_labels])
        self.labels = special_labels + non_special
        self.label_map = {label: i for i, label in enumerate(self.labels)}
        self.reverse_label_map = {i: label for i, label in enumerate(self.labels)}
        
        logging.info(f"Labels mapeados ({len(self.labels)}): {list(self.label_map.keys())[:10]}...")

def convert_examples_to_features(examples, processor, tokenizer, max_seq_length):
    features = []
    
    for ex_idx, example in enumerate(examples):
        words = example.words
        labels = example.labels
        
        tokens = []
        label_ids = []
        
        for word, label in zip(words, labels):
            word_tokens = tokenizer.tokenize(word)
            
            if not word_tokens:
                word_tokens = [tokenizer.unk_token]
            
            tokens.extend(word_tokens)
            
            if label in processor.label_map:
                main_label = processor.label_map[label]
            else:
                main_label = processor.label_map['O']
            
            label_ids.append(main_label)
            label_ids.extend([processor.label_map['X']] * (len(word_tokens) - 1))
        
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:max_seq_length - 2]
            label_ids = label_ids[:max_seq_length - 2]
        
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        label_ids = [processor.label_map['[CLS]']] + label_ids + [processor.label_map['[SEP]']]
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        
        padding_length = max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            token_type_ids = token_type_ids + [0] * padding_length
            label_ids = label_ids + [processor.label_map['[PAD]']] * padding_length
        
        predict_mask = []
        for i, (token_id, label_id) in enumerate(zip(input_ids, label_ids)):
            label_name = processor.reverse_label_map[label_id]
            if (token_id == tokenizer.pad_token_id or 
                token_id == tokenizer.cls_token_id or 
                token_id == tokenizer.sep_token_id or
                label_name in ['[PAD]', '[CLS]', '[SEP]', 'X']):
                predict_mask.append(0)
            else:
                predict_mask.append(1)
        
        features.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label_ids': label_ids,
            'predict_mask': predict_mask
        })
    
    return features

class NerDataset(data.Dataset):
    def __init__(self, examples, processor, tokenizer, max_seq_length):
        self.features = convert_examples_to_features(
            examples, processor, tokenizer, max_seq_length
        )
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feat = self.features[idx]
        return {
            'input_ids': torch.tensor(feat['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(feat['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(feat['token_type_ids'], dtype=torch.long),
            'labels': torch.tensor(feat['label_ids'], dtype=torch.long),
            'predict_mask': torch.tensor(feat['predict_mask'], dtype=torch.bool)
        }

def evaluate(model, dataloader, processor, tokenizer):
    model.eval()
    true_labels, pred_labels = [], []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Avaliando", leave=False):
            input_ids = batch['input_ids'].to(Config.device)
            attention_mask = batch['attention_mask'].to(Config.device)
            token_type_ids = batch['token_type_ids'].to(Config.device)
            labels = batch['labels'].to(Config.device)
            predict_mask = batch['predict_mask'].cpu().numpy()
            
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            if loss is not None:
                total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            for i in range(len(preds)):
                current_true = []
                current_pred = []
                
                for j in range(len(preds[i])):
                    if predict_mask[i][j]:
                        true_label = processor.reverse_label_map[labels_np[i][j]]
                        pred_label = processor.reverse_label_map[preds[i][j]]
                        
                        if true_label not in ['[PAD]', '[CLS]', '[SEP]', 'X']:
                            current_true.append(true_label)
                            current_pred.append(pred_label)
                
                if current_true:
                    true_labels.append(current_true)
                    pred_labels.append(current_pred)
    
    avg_loss = total_loss / max(len(dataloader), 1)
    
    if true_labels and pred_labels:
        try:
            f1 = f1_score(true_labels, pred_labels, zero_division=0)
            precision = precision_score(true_labels, pred_labels, zero_division=0)
            recall = recall_score(true_labels, pred_labels, zero_division=0)
            report = classification_report(true_labels, pred_labels, zero_division=0)
        except:
            f1 = 0.0
            precision = 0.0
            recall = 0.0
            report = "Erro ao calcular métricas"
    else:
        f1 = 0.0
        precision = 0.0
        recall = 0.0
        report = "Nenhuma predição válida"
    
    return {
        'loss': avg_loss,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'report': report
    }

def train_fold(fold, train_loader, val_loader, processor, tokenizer, args):
    kernel_sizes = [int(k) for k in args.cnn_kernel_sizes.split(',')]
    model = ElectraCNN(
        args.model_name,
        len(processor.labels),
        num_filters=args.cnn_num_filters,
        kernel_sizes=kernel_sizes,
        dropout=args.dropout
    ).to(Config.device)
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    best_f1 = -1
    patience_counter = 0
    best_model_path = os.path.join(args.output_dir, f"electra_cnn_fold_{fold}_best.pt")
    
    logging.info(f"🔧 Configuração Fold {fold}:")
    logging.info(f"  Labels: {len(processor.labels)}")
    logging.info(f"  Batch size: {args.batch_size}")
    logging.info(f"  CNN filters: {args.cnn_num_filters}")
    logging.info(f"  Kernel sizes: {args.cnn_kernel_sizes}")
    
    global_step = 0
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        batch_count = 0
        
        progress_bar = tqdm(train_loader, desc=f"Fold {fold} - Época {epoch+1}", leave=False)
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(Config.device)
            attention_mask = batch['attention_mask'].to(Config.device)
            token_type_ids = batch['token_type_ids'].to(Config.device)
            labels = batch['labels'].to(Config.device)
            
            loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()
            total_loss += loss.item()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            batch_count += 1
            
            if global_step % Config.logging_steps == 0:
                progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / max(batch_count, 1)
        val_results = evaluate(model, val_loader, processor, tokenizer)
        epoch_time = time.time() - epoch_start_time
        
        logging.info(f"\n=== Fold {fold}, Época {epoch+1} ===")
        logging.info(f"⏱️  Tempo: {epoch_time:.2f}s")
        logging.info(f"📉 Loss Treino: {avg_train_loss:.4f}")
        logging.info(f"📊 Loss Validação: {val_results['loss']:.4f}")
        logging.info(f"🎯 F1 Validação: {val_results['f1']:.4f}")
        logging.info(f"🎯 Precisão: {val_results['precision']:.4f}")
        logging.info(f"🎯 Recall: {val_results['recall']:.4f}")
        
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
        model.load_state_dict(torch.load(best_model_path, map_location=Config.device))
        logging.info(f"📦 Melhor modelo carregado (F1: {best_f1:.4f})")
    
    return evaluate(model, val_loader, processor, tokenizer)

def main():
    args = parse_args()
    
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'electra_cnn_training.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    with open(os.path.join(args.output_dir, 'electra_cnn_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logging.info("=" * 70)
    logging.info("🚀 INICIANDO TREINAMENTO - ELECTRA CNN")
    logging.info(f"📁 Dados: {args.data}")
    logging.info(f"📂 Saída: {args.output_dir}")
    logging.info(f"🤖 Modelo: {args.model_name}")
    logging.info(f"🎯 Dispositivo: {Config.device}")
    logging.info(f"🧮 Folds: {args.k_folds}")
    logging.info(f"🔧 CNN filters: {args.cnn_num_filters}")
    logging.info(f"🔧 Kernel sizes: {args.cnn_kernel_sizes}")
    logging.info("=" * 70)
    
    try:
        tokenizer = ElectraTokenizer.from_pretrained(args.model_name)
        processor = DataProcessor()
        examples = processor.get_examples(args.data)
        logging.info(f"📊 Total de exemplos: {len(examples)}")
        logging.info(f"🏷️  Labels únicos: {len(processor.labels)}")
        
        kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=Config.seed)
        fold_results = []
        all_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(examples)):
            fold_start_time = time.time()
            
            logging.info(f"\n{'='*60}")
            logging.info(f"🔄 INICIANDO FOLD {fold+1}/{args.k_folds}")
            logging.info(f"{'='*60}")
            
            train_examples = [examples[i] for i in train_idx]
            val_examples = [examples[i] for i in val_idx]
            
            train_dataset = NerDataset(train_examples, processor, tokenizer, args.max_seq_length)
            val_dataset = NerDataset(val_examples, processor, tokenizer, args.max_seq_length)
            
            train_loader = data.DataLoader(
                train_dataset, 
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0
            )
            val_loader = data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                num_workers=0
            )
            
            results = train_fold(fold+1, train_loader, val_loader, processor, tokenizer, args)
            
            fold_time = time.time() - fold_start_time
            fold_results.append(results['f1'])
            all_metrics.append({
                'fold': fold+1,
                'f1': results['f1'],
                'precision': results['precision'],
                'recall': results['recall'],
                'loss': results['loss'],
                'time': fold_time
            })
            
            logging.info(f"\n📊 RESULTADOS FOLD {fold+1}:")
            logging.info(f"  F1: {results['f1']:.4f}")
            logging.info(f"  Precisão: {results['precision']:.4f}")
            logging.info(f"  Recall: {results['recall']:.4f}")
            logging.info(f"  Loss: {results['loss']:.4f}")
            logging.info(f"  Tempo: {fold_time:.2f}s")
            
            with open(os.path.join(args.output_dir, f'electra_cnn_fold_{fold+1}_report.txt'), 'w') as f:
                f.write(f"FOLD {fold+1} - ELECTRA CNN\n")
                f.write("="*50 + "\n")
                f.write(f"F1: {results['f1']:.4f}\n")
                f.write(f"Precision: {results['precision']:.4f}\n")
                f.write(f"Recall: {results['recall']:.4f}\n")
                f.write(f"Loss: {results['loss']:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(results['report'])
        
        final_f1 = np.mean(fold_results)
        final_std = np.std(fold_results)
        
        logging.info(f"\n{'='*70}")
        logging.info("🎯 RESULTADOS FINAIS - ELECTRA CNN")
        logging.info(f"{'='*70}")
        logging.info(f"📊 F1 Médio: {final_f1:.4f} (±{final_std:.4f})")
        logging.info(f"📈 F1 por fold: {[round(f, 4) for f in fold_results]}")
        logging.info(f"🏆 Melhor F1: {max(fold_results):.4f}")
        logging.info(f"📉 Pior F1: {min(fold_results):.4f}")
        
        final_metrics = {
            'architecture': 'ELECTRA_cnn',
            'model': args.model_name,
            'final_f1_mean': float(final_f1),
            'final_f1_std': float(final_std),
            'folds': all_metrics,
            'args': vars(args)
        }
        
        with open(os.path.join(args.output_dir, 'electra_cnn_final_metrics.json'), 'w') as f:
            json.dump(final_metrics, f, indent=2, ensure_ascii=False)
        
        with open(os.path.join(args.output_dir, 'electra_cnn_final_results.txt'), 'w') as f:
            f.write("RESULTADOS FINAIS - ELECTRA CNN\n")
            f.write("="*60 + "\n")
            f.write(f"Modelo: {args.model_name}\n")
            f.write(f"F1 Médio: {final_f1:.4f} (±{final_std:.4f})\n\n")
            f.write(f"Folds detalhados:\n")
            for i, metrics in enumerate(all_metrics, 1):
                f.write(f"\n  Fold {i}:\n")
                f.write(f"    F1: {metrics['f1']:.4f}\n")
                f.write(f"    Precision: {metrics['precision']:.4f}\n")
                f.write(f"    Recall: {metrics['recall']:.4f}\n")
                f.write(f"    Loss: {metrics['loss']:.4f}\n")
                f.write(f"    Tempo: {metrics['time']:.2f}s\n")
        
        logging.info(f"\n✅ Treinamento concluído!")
        logging.info(f"📁 Resultados salvos em: {args.output_dir}")
        
    except Exception as e:
        logging.error(f"\n❌ ERRO CRÍTICO: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()