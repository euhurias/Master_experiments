import sys
import os
import argparse
import logging
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from collections import Counter
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Arquivo com dados no formato IOB')
    parser.add_argument('--output_dir', required=True, help='Diretório para salvar resultados')
    parser.add_argument('--model_name', type=str, default='neuralmind/bert-base-portuguese-cased',
                        help='Necessário apenas para o tokenizer')
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Taxa de aprendizado (padrão 1e-3 para treinar embeddings do zero)')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_seq_length', type=int, default=180)
    parser.add_argument('--embedding_dim', type=int, default=300, help='Dimensão dos embeddings')
    parser.add_argument('--num_filters', type=int, default=300, help='Número de filtros por tamanho')
    parser.add_argument('--filter_sizes', type=str, default='3,4,5',
                        help='Tamanhos dos filtros CNN separados por vírgula')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Número de camadas convolucionais empilhadas')
    parser.add_argument('--dropout', type=float, default=0.3, help='Taxa de dropout')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Usar pesos de classe na loss')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Salvar predições detalhadas de cada fold')
    parser.add_argument('--debug', action='store_true', help='Ativa logs detalhados')
    return parser.parse_args()

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    logging_steps = 50
    max_grad_norm = 1.0

# ==================== CLASSE CNN_NER CORRIGIDA ====================
class CNN_NER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_layers, num_labels, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Camadas convolucionais com BatchNorm
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        in_channels = embedding_dim
        for _ in range(num_layers):
            convs_for_layer = nn.ModuleList()
            bns_for_layer = nn.ModuleList()
            for fs in filter_sizes:
                # Padding = fs // 2 garante que a saída tenha comprimento = entrada ou entrada+1
                conv = nn.Conv1d(in_channels, num_filters, kernel_size=fs, padding=fs // 2)
                nn.init.xavier_uniform_(conv.weight)
                nn.init.zeros_(conv.bias)
                convs_for_layer.append(conv)
                bn = nn.BatchNorm1d(num_filters)
                bns_for_layer.append(bn)
            self.conv_layers.append(convs_for_layer)
            self.bn_layers.append(bns_for_layer)
            in_channels = num_filters * len(filter_sizes)  # para a próxima camada

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_channels, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask=None, labels=None, class_weights=None):
        # input_ids: [batch_size, seq_len]
        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        x = embeddings.transpose(1, 2)          # [batch_size, in_channels, seq_len]

        for convs, bns in zip(self.conv_layers, self.bn_layers):
            conv_outputs = []
            for conv, bn in zip(convs, bns):
                conv_out = conv(x)               # [batch_size, num_filters, seq_len + (padding extra)]
                # Garantir que o comprimento seja exatamente o de x (cortar se necessário)
                if conv_out.size(2) > x.size(2):
                    conv_out = conv_out[:, :, :x.size(2)]
                conv_out = bn(conv_out)
                conv_out = F.relu(conv_out)
                conv_outputs.append(conv_out)
            # Concatena na dimensão dos canais
            x = torch.cat(conv_outputs, dim=1)   # [batch_size, num_filters * len(filter_sizes), seq_len]
            x = self.dropout(x)

        x = x.transpose(1, 2)                    # [batch_size, seq_len, features]
        logits = self.classifier(x)               # [batch_size, seq_len, num_labels]

        loss = None
        if labels is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.size(-1))[active_loss]
            active_labels = labels.view(-1)[active_loss]

            if active_labels.numel() > 0:
                if class_weights is not None:
                    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
                else:
                    loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(active_logits, active_labels)
            else:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        return loss, logits
# ===================================================================

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
        special_labels = ['[PAD]', '[CLS]', '[SEP]', 'X']
        self.labels = special_labels + sorted([l for l in labels if l not in special_labels])
        self.label_map = {label: i for i, label in enumerate(self.labels)}
        self.reverse_label_map = {i: label for i, label in enumerate(self.labels)}
        logging.info(f"Labels mapeados ({len(self.labels)}): {list(self.label_map.keys())[:10]}...")

def convert_examples_to_features(examples, tokenizer, processor, max_seq_length):
    features = []
    for ex_idx, example in enumerate(examples):
        tokens = ['[CLS]']
        label_ids = [processor.label_map['[CLS]']]
        predict_mask = [0]

        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [tokenizer.unk_token]

            tokens.extend(word_tokens)
            main_label = processor.label_map.get(label, processor.label_map['X'])
            label_ids.extend([main_label] + [processor.label_map['X']] * (len(word_tokens) - 1))
            predict_mask.extend([1] + [0] * (len(word_tokens) - 1))

        if len(tokens) > max_seq_length - 1:
            tokens = tokens[:max_seq_length - 1]
            label_ids = label_ids[:max_seq_length - 1]
            predict_mask = predict_mask[:max_seq_length - 1]

        tokens.append('[SEP]')
        label_ids.append(processor.label_map['[SEP]'])
        predict_mask.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids += [tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            label_ids += [processor.label_map['[PAD]']] * padding_length
            predict_mask += [0] * padding_length

        features.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_ids,
            'predict_mask': predict_mask
        })

    return features

class NerDataset(data.Dataset):
    def __init__(self, examples, tokenizer, processor, max_seq_length):
        self.features = convert_examples_to_features(examples, tokenizer, processor, max_seq_length)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        return {
            'input_ids': torch.tensor(feat['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(feat['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(feat['labels'], dtype=torch.long),
            'predict_mask': torch.tensor(feat['predict_mask'], dtype=torch.bool)
        }

def evaluate(model, dataloader, processor, class_weights=None, debug=False, fold=None, output_dir=None, save_predictions=False):
    model.eval()
    true_labels, pred_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Avaliando", leave=False):
            input_ids = batch['input_ids'].to(Config.device)
            attention_mask = batch['attention_mask'].to(Config.device)
            labels = batch['labels'].to(Config.device)
            predict_mask = batch['predict_mask'].cpu().numpy()

            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                class_weights=class_weights
            )
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(len(preds)):
                current_true = []
                current_pred = []
                for j in range(len(predict_mask[i])):
                    if predict_mask[i][j]:
                        true_label_id = labels_np[i][j]
                        pred_label_id = preds[i][j]
                        true_label = processor.reverse_label_map[true_label_id]
                        pred_label = processor.reverse_label_map[pred_label_id]
                        if true_label not in ['[PAD]', '[CLS]', '[SEP]', 'X']:
                            current_true.append(true_label)
                            current_pred.append(pred_label)
                if current_true:
                    true_labels.append(current_true)
                    pred_labels.append(current_pred)

    avg_loss = total_loss / max(len(dataloader), 1)

    if debug:
        all_true = [lbl for seq in true_labels for lbl in seq]
        all_pred = [lbl for seq in pred_labels for lbl in seq]
        true_counter = Counter(all_true)
        pred_counter = Counter(all_pred)
        logging.debug(f"Distribuição true labels (total {len(all_true)}): {dict(true_counter.most_common(10))}")
        logging.debug(f"Distribuição pred labels (total {len(all_pred)}): {dict(pred_counter.most_common(10))}")
        entity_preds = [lbl for lbl in all_pred if lbl != 'O']
        logging.info(f"Total de entidades preditas: {len(entity_preds)}")

    if true_labels and pred_labels:
        try:
            f1 = f1_score(true_labels, pred_labels, zero_division=0)
            precision = precision_score(true_labels, pred_labels, zero_division=0)
            recall = recall_score(true_labels, pred_labels, zero_division=0)
            report = classification_report(true_labels, pred_labels, zero_division=0)
            report_dict = classification_report(true_labels, pred_labels, zero_division=0, output_dict=True)
        except:
            f1 = precision = recall = 0.0
            report = "Erro ao calcular métricas"
            report_dict = {}
    else:
        f1 = precision = recall = 0.0
        report = "Nenhuma predição válida"
        report_dict = {}

    # Salvar predições se solicitado
    if save_predictions and fold is not None and output_dir is not None:
        pred_file = os.path.join(output_dir, f'cnn_fold_{fold}_predictions.txt')
        with open(pred_file, 'w', encoding='utf-8') as f:
            f.write("TRUE_LABELS\tPRED_LABELS\n")
            for true_seq, pred_seq in zip(true_labels, pred_labels):
                f.write(" ".join(true_seq) + "\t" + " ".join(pred_seq) + "\n")
        logging.info(f"Predições salvas em: {pred_file}")

    if report_dict and fold is not None and output_dir is not None:
        csv_file = os.path.join(output_dir, f'cnn_fold_{fold}_class_report.csv')
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['class', 'precision', 'recall', 'f1-score', 'support']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for cls, metrics in report_dict.items():
                if cls not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(metrics, dict):
                    writer.writerow({
                        'class': cls,
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1-score': metrics['f1-score'],
                        'support': metrics['support']
                    })
        logging.info(f"Relatório por classe salvo em: {csv_file}")

    return {
        'loss': avg_loss,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'report': report
    }

def train_fold(fold, train_loader, val_loader, processor, args, tokenizer):
    # Contar labels no treino para pesos e diagnóstico
    label_counter = Counter()
    for batch in train_loader:
        labels = batch['labels'].numpy()
        mask = batch['predict_mask'].numpy()
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if mask[i][j]:
                    label = processor.reverse_label_map[labels[i][j]]
                    if label not in ['[PAD]', '[CLS]', '[SEP]', 'X']:
                        label_counter[label] += 1
    logging.info(f"Distribuição de labels no treino (Fold {fold}): {dict(label_counter)}")

    filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
    vocab_size = tokenizer.vocab_size

    model = CNN_NER(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        num_filters=args.num_filters,
        filter_sizes=filter_sizes,
        num_layers=args.num_layers,
        num_labels=len(processor.labels),
        dropout=args.dropout
    ).to(Config.device)

    # Calcular pesos de classe
    class_weights = None
    if args.use_class_weights:
        total = sum(label_counter.values())
        num_classes = len(processor.labels)
        weights = torch.ones(num_classes)
        for lbl, idx in processor.label_map.items():
            if lbl not in ['[PAD]', '[CLS]', '[SEP]', 'X']:
                count = label_counter.get(lbl, 0)
                if count > 0:
                    weights[idx] = total / (num_classes * count)
                else:
                    weights[idx] = 1.0
        weights = weights / weights.mean()  # normalizar
        class_weights = weights.to(Config.device)
        logging.info(f"Pesos de classe calculados (média: {weights.mean():.4f})")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_f1 = -1
    patience_counter = 0
    best_model_path = os.path.join(args.output_dir, f"cnn_fold_{fold}_best.pt")

    logging.info(f"🔧 Configuração Fold {fold}:")
    logging.info(f"  Labels: {len(processor.labels)}")
    logging.info(f"  Vocab size: {vocab_size}")
    logging.info(f"  Embedding dim: {args.embedding_dim}")
    logging.info(f"  Num filters: {args.num_filters}")
    logging.info(f"  Filter sizes: {filter_sizes}")
    logging.info(f"  Num layers: {args.num_layers}")
    logging.info(f"  Dropout: {args.dropout}")

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
            labels = batch['labels'].to(Config.device)

            loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                class_weights=class_weights
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
        val_results = evaluate(
            model, val_loader, processor, class_weights,
            debug=args.debug, fold=fold, output_dir=args.output_dir,
            save_predictions=args.save_predictions
        )
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

    # Avaliação final com o melhor modelo
    final_results = evaluate(
        model, val_loader, processor, class_weights,
        debug=args.debug, fold=fold, output_dir=args.output_dir,
        save_predictions=args.save_predictions
    )
    return final_results

def main():
    args = parse_args()

    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'cnn_training.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

    with open(os.path.join(args.output_dir, 'cnn_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    logging.info("=" * 70)
    logging.info("🚀 INICIANDO TREINAMENTO - CNN (MELHORADA)")
    logging.info(f"📁 Dados: {args.data}")
    logging.info(f"📂 Saída: {args.output_dir}")
    logging.info(f"🎯 Dispositivo: {Config.device}")
    logging.info(f"🧮 Folds: {args.k_folds}")
    logging.info("=" * 70)

    try:
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
        processor = DataProcessor()
        examples = processor.get_examples(args.data)
        logging.info(f"📊 Total de exemplos: {len(examples)}")
        logging.info(f"🏷️  Labels únicos: {len(processor.labels)}")

        kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=Config.seed)
        fold_results = []
        all_metrics = []
        total_start_time = time.time()

        for fold, (train_idx, val_idx) in enumerate(kf.split(examples)):
            fold_start_time = time.time()

            logging.info(f"\n{'='*60}")
            logging.info(f"🔄 INICIANDO FOLD {fold+1}/{args.k_folds}")
            logging.info(f"{'='*60}")

            train_examples = [examples[i] for i in train_idx]
            val_examples = [examples[i] for i in val_idx]

            train_dataset = NerDataset(train_examples, tokenizer, processor, args.max_seq_length)
            val_dataset = NerDataset(val_examples, tokenizer, processor, args.max_seq_length)

            train_loader = data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0
            )
            val_loader = data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                num_workers=0
            )

            results = train_fold(fold+1, train_loader, val_loader, processor, args, tokenizer)

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

            with open(os.path.join(args.output_dir, f'cnn_fold_{fold+1}_report.txt'), 'w') as f:
                f.write(f"FOLD {fold+1} - CNN\n")
                f.write("="*50 + "\n")
                f.write(f"F1: {results['f1']:.4f}\n")
                f.write(f"Precision: {results['precision']:.4f}\n")
                f.write(f"Recall: {results['recall']:.4f}\n")
                f.write(f"Loss: {results['loss']:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(results['report'])

        total_time = time.time() - total_start_time
        final_f1 = np.mean(fold_results)
        final_std = np.std(fold_results)

        logging.info(f"\n{'='*70}")
        logging.info("🎯 RESULTADOS FINAIS - CNN")
        logging.info(f"{'='*70}")
        logging.info(f"📊 F1 Médio: {final_f1:.4f} (±{final_std:.4f})")
        logging.info(f"📈 F1 por fold: {[round(f, 4) for f in fold_results]}")
        logging.info(f"🏆 Melhor F1: {max(fold_results):.4f}")
        logging.info(f"📉 Pior F1: {min(fold_results):.4f}")
        logging.info(f"⏱️  Tempo total: {total_time:.2f}s")

        final_metrics = {
            'architecture': 'CNN',
            'final_f1_mean': float(final_f1),
            'final_f1_std': float(final_std),
            'folds': all_metrics,
            'total_time_seconds': total_time,
            'args': vars(args)
        }

        with open(os.path.join(args.output_dir, 'cnn_final_metrics.json'), 'w') as f:
            json.dump(final_metrics, f, indent=2, ensure_ascii=False)

        with open(os.path.join(args.output_dir, 'cnn_final_results.txt'), 'w') as f:
            f.write("RESULTADOS FINAIS - CNN\n")
            f.write("="*60 + "\n")
            f.write(f"F1 Médio: {final_f1:.4f} (±{final_std:.4f})\n\n")
            f.write("Folds detalhados:\n")
            for m in all_metrics:
                f.write(f"\n  Fold {m['fold']}:\n")
                f.write(f"    F1: {m['f1']:.4f}\n")
                f.write(f"    Precision: {m['precision']:.4f}\n")
                f.write(f"    Recall: {m['recall']:.4f}\n")
                f.write(f"    Loss: {m['loss']:.4f}\n")
                f.write(f"    Tempo: {m['time']:.2f}s\n")
            f.write(f"\nTempo total: {total_time:.2f}s\n")

        logging.info(f"\n✅ Treinamento concluído!")
        logging.info(f"📁 Resultados salvos em: {args.output_dir}")

    except Exception as e:
        logging.error(f"\n❌ ERRO CRÍTICO: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()