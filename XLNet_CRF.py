import sys
import os
import argparse
import logging
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils import data
from transformers import XLNetModel, XLNetTokenizer, get_linear_schedule_with_warmup
from torchcrf import CRF
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Arquivo com dados no formato IOB')
    parser.add_argument('--output_dir', required=True, help='Diretório para salvar resultados')
    parser.add_argument('--model_name', type=str, default='xlnet-base-cased')
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--learning_rate', type=str, default='2e-5')
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Ignorado (CRF não suporta pesos de classe)')
    return parser.parse_args()

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    logging_steps = 50
    max_grad_norm = 1.0

class InputExample:
    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels

class DataProcessor:
    def __init__(self):
        self.special_labels = ['[PAD]', '[CLS]', '[SEP]', 'X']
        self.label_map = {}
        self.id_to_label = {}
        self.labels = []  # lista de labels para referência

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

    def _create_label_maps(self, all_labels):
        # Primeiro mapeia labels especiais
        self.label_map = {label: i for i, label in enumerate(self.special_labels)}

        # Depois adiciona labels reais
        real_labels = sorted(all_labels.difference(set(self.special_labels)))
        for label in real_labels:
            self.label_map[label] = len(self.label_map)

        # Cria mapeamento inverso
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        self.labels = list(self.label_map.keys())
        logging.info(f"Labels mapeados ({len(self.labels)}): {list(self.label_map.keys())[:10]}...")

def convert_example_to_feature(example, tokenizer, processor, max_seq_length):
    tokens, labels, predict_mask = [], [], []
    for word, label in zip(example.words, example.labels):
        word_tokens = tokenizer.tokenize(word) or [tokenizer.unk_token]
        tokens.extend(word_tokens)

        # Label principal
        main_label = processor.label_map.get(label, processor.label_map['X'])
        labels.append(main_label)
        predict_mask.append(1)

        # Subpalavras adicionais: X e não preditas
        if len(word_tokens) > 1:
            labels.extend([processor.label_map['X']] * (len(word_tokens) - 1))
            predict_mask.extend([0] * (len(word_tokens) - 1))

    max_len = max_seq_length - 2
    tokens = tokens[:max_len]
    labels = labels[:max_len]
    predict_mask = predict_mask[:max_len]

    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    labels = [processor.label_map['[CLS]']] + labels + [processor.label_map['[SEP]']]
    predict_mask = [0] + predict_mask + [0]

    pad_len = max_seq_length - len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.pad_token_id] * pad_len
    attention_mask = [1] * len(tokens) + [0] * pad_len
    labels += [processor.label_map['[PAD]']] * pad_len
    predict_mask += [0] * pad_len

    return {
        'input_ids': input_ids[:max_seq_length],
        'attention_mask': attention_mask[:max_seq_length],
        'labels': labels[:max_seq_length],
        'predict_mask': predict_mask[:max_seq_length]
    }

class NERDataset(data.Dataset):
    def __init__(self, examples, tokenizer, processor, max_seq_length):
        self.features = [convert_example_to_feature(ex, tokenizer, processor, max_seq_length) for ex in examples]

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

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    predict_mask = torch.stack([item['predict_mask'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'predict_mask': predict_mask
    }

class XLNetCRF(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.3):
        super().__init__()
        self.xlnet = XLNetModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.xlnet.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        nn.init.normal_(self.crf.start_transitions, 0, 0.1)
        nn.init.normal_(self.crf.end_transitions, 0, 0.1)
        nn.init.normal_(self.crf.transitions, 0, 0.1)

    def forward(self, input_ids, attention_mask, labels=None, predict_mask=None):
        outputs = self.xlnet(input_ids, attention_mask=attention_mask)
        sequence = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence)  # [batch, seq_len, num_labels]

        # Remove [CLS] e [SEP] para a CRF
        emissions = emissions[:, 1:-1, :]

        if labels is not None:
            labels = labels[:, 1:-1]
            # Predict_mask já considera [CLS]/[SEP] removidos, então usamos diretamente
            mask = predict_mask[:, 1:-1].bool()

            # Verificação de segurança
            assert emissions.size(1) == labels.size(1), f"Emissions: {emissions.shape}, Labels: {labels.shape}"
            assert emissions.size(1) == mask.size(1), f"Emissions: {emissions.shape}, Mask: {mask.shape}"

            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss, emissions
        else:
            return None, emissions

    def decode(self, emissions, mask):
        # mask já sem [CLS]/[SEP]
        return self.crf.decode(emissions, mask=mask)

def evaluate(model, dataloader, processor):
    model.eval()
    true_labels, pred_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Avaliando", leave=False):
            input_ids = batch['input_ids'].to(Config.device)
            attention_mask = batch['attention_mask'].to(Config.device)
            labels = batch['labels'].to(Config.device)
            predict_mask = batch['predict_mask'].to(Config.device)

            loss, emissions = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                predict_mask=predict_mask
            )
            total_loss += loss.item()

            # Decodificação
            mask = predict_mask[:, 1:-1].bool()
            preds_list = model.decode(emissions, mask)

            # Converte para array
            max_len = emissions.size(1)
            preds = torch.full((len(preds_list), max_len), -1, dtype=torch.long)
            for i, seq in enumerate(preds_list):
                seq_len = min(len(seq), max_len)
                preds[i, :seq_len] = torch.tensor(seq[:seq_len])

            labels_np = labels[:, 1:-1].cpu().numpy()
            mask_np = mask.cpu().numpy()
            preds_np = preds.numpy()

            for i in range(len(preds_np)):
                current_true = []
                current_pred = []
                for j in range(mask_np.shape[1]):
                    if mask_np[i][j] and labels_np[i][j] != -100:
                        true_label = processor.id_to_label[labels_np[i][j]]
                        pred_label = processor.id_to_label[preds_np[i][j]]
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
            f1 = precision = recall = 0.0
            report = "Erro ao calcular métricas"
    else:
        f1 = precision = recall = 0.0
        report = "Nenhuma predição válida"

    return {
        'loss': avg_loss,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'report': report
    }

def train_fold(fold, train_loader, val_loader, processor, args):
    # Verificar distribuição de labels no treino (apenas tokens preditos)
    label_counter = Counter()
    for batch in train_loader:
        labels = batch['labels'].numpy()
        mask = batch['predict_mask'].numpy()
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if mask[i][j] and labels[i][j] != -100:
                    label = processor.id_to_label[labels[i][j]]
                    if label not in ['[PAD]', '[CLS]', '[SEP]', 'X']:
                        label_counter[label] += 1
    logging.info(f"Distribuição de labels no treino (Fold {fold}): {dict(label_counter)}")

    if args.use_class_weights:
        logging.warning("Pesos de classe não são suportados pela CRF. Ignorando.")

    # Criar modelo
    model = XLNetCRF(
        args.model_name,
        len(processor.label_map),
        dropout=args.dropout
    ).to(Config.device)

    # Otimizador
    base_lr = float(args.learning_rate)
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_f1 = -1
    patience_counter = 0
    best_model_path = os.path.join(args.output_dir, f"xlnet_crf_fold_{fold}_best.pt")

    logging.info(f"🔧 Configuração Fold {fold}:")
    logging.info(f"  Labels: {len(processor.labels)}")
    logging.info(f"  Batch size: {args.batch_size}")
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
            predict_mask = batch['predict_mask'].to(Config.device)

            loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                predict_mask=predict_mask
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
        val_results = evaluate(model, val_loader, processor)
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

    # Carrega o melhor modelo e avalia novamente
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=Config.device))
        logging.info(f"📦 Melhor modelo carregado (F1: {best_f1:.4f})")

    return evaluate(model, val_loader, processor)

def main():
    args = parse_args()
    args.learning_rate = float(args.learning_rate)

    # Seeds
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Configura logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'xlnet_crf_training.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Salva argumentos
    with open(os.path.join(args.output_dir, 'xlnet_crf_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    logging.info("=" * 70)
    logging.info("🚀 INICIANDO TREINAMENTO - XLNet + CRF")
    logging.info(f"📁 Dados: {args.data}")
    logging.info(f"📂 Saída: {args.output_dir}")
    logging.info(f"🤖 Modelo: {args.model_name}")
    logging.info(f"🎯 Dispositivo: {Config.device}")
    logging.info(f"🧮 Folds: {args.k_folds}")
    logging.info("=" * 70)

    try:
        tokenizer = XLNetTokenizer.from_pretrained(args.model_name)
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

            train_dataset = NERDataset(train_examples, tokenizer, processor, args.max_seq_length)
            val_dataset = NERDataset(val_examples, tokenizer, processor, args.max_seq_length)

            train_loader = data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                drop_last=True,
                num_workers=0
            )
            val_loader = data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                collate_fn=collate_fn,
                num_workers=0
            )

            results = train_fold(fold+1, train_loader, val_loader, processor, args)

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

            # Salva relatório do fold
            with open(os.path.join(args.output_dir, f'xlnet_crf_fold_{fold+1}_report.txt'), 'w') as f:
                f.write(f"FOLD {fold+1} - XLNet + CRF\n")
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
        logging.info("🎯 RESULTADOS FINAIS - XLNet + CRF")
        logging.info(f"{'='*70}")
        logging.info(f"📊 F1 Médio: {final_f1:.4f} (±{final_std:.4f})")
        logging.info(f"📈 F1 por fold: {[round(f, 4) for f in fold_results]}")
        logging.info(f"🏆 Melhor F1: {max(fold_results):.4f}")
        logging.info(f"📉 Pior F1: {min(fold_results):.4f}")

        final_metrics = {
            'architecture': 'XLNet_CRF',
            'model': args.model_name,
            'final_f1_mean': float(final_f1),
            'final_f1_std': float(final_std),
            'folds': all_metrics,
            'args': vars(args)
        }

        with open(os.path.join(args.output_dir, 'xlnet_crf_final_metrics.json'), 'w') as f:
            json.dump(final_metrics, f, indent=2, ensure_ascii=False)

        with open(os.path.join(args.output_dir, 'xlnet_crf_final_results.txt'), 'w') as f:
            f.write("RESULTADOS FINAIS - XLNet + CRF\n")
            f.write("="*60 + "\n")
            f.write(f"Modelo: {args.model_name}\n")
            f.write(f"F1 Médio: {final_f1:.4f} (±{final_std:.4f})\n\n")
            f.write("Folds detalhados:\n")
            for m in all_metrics:
                f.write(f"\n  Fold {m['fold']}:\n")
                f.write(f"    F1: {m['f1']:.4f}\n")
                f.write(f"    Precision: {m['precision']:.4f}\n")
                f.write(f"    Recall: {m['recall']:.4f}\n")
                f.write(f"    Loss: {m['loss']:.4f}\n")
                f.write(f"    Tempo: {m['time']:.2f}s\n")

        logging.info(f"\n✅ Treinamento concluído!")
        logging.info(f"📁 Resultados salvos em: {args.output_dir}")

    except Exception as e:
        logging.error(f"\n❌ ERRO CRÍTICO: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()