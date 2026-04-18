import sys
import os
import argparse
import logging
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Arquivo com dados no formato IOB')
    parser.add_argument('--output_dir', required=True, help='Diretório para salvar resultados')
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--learning_rate', type=str, default='0.01')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Usar pesos de classe na loss')
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
        self.label_map = {}
        self.labels = []
        self.reverse_label_map = {}
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.words = set()

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
                            self.words.add(parts[0])
                if words:
                    examples.append(InputExample(i, words, labels))

        self._create_label_maps(all_labels)
        # Build word vocabulary
        self.word_to_idx.update({w: i+2 for i, w in enumerate(sorted(self.words))})
        return examples

    def _create_label_maps(self, labels):
        # Ordena mantendo 'O' primeiro se desejado, mas não é obrigatório
        self.labels = sorted(list(labels))
        self.label_map = {label: i for i, label in enumerate(self.labels)}
        self.reverse_label_map = {i: label for i, label in enumerate(self.labels)}
        logging.info(f"Labels mapeados ({len(self.labels)}): {list(self.label_map.keys())[:10]}...")

def convert_examples_to_features(examples, processor):
    """Converte exemplos para índices de palavras e labels."""
    features = []
    for ex_idx, example in enumerate(examples):
        word_ids = [processor.word_to_idx.get(w, processor.word_to_idx["<UNK>"]) for w in example.words]
        label_ids = [processor.label_map[l] for l in example.labels]
        features.append({
            'word_ids': word_ids,
            'label_ids': label_ids
        })
    return features

class NERDataset(Dataset):
    def __init__(self, examples, processor):
        self.features = convert_examples_to_features(examples, processor)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        return torch.tensor(feat['word_ids'], dtype=torch.long), torch.tensor(feat['label_ids'], dtype=torch.long)

def collate_fn(batch):
    sentences, labels = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)
    # A máscara não é estritamente necessária para a loss com ignore_index,
    # mas podemos usá-la para filtrar tokens válidos nas métricas.
    mask = (sentences_padded != 0)
    return sentences_padded, labels_padded, mask

class NERBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, num_labels):
        super(NERBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_labels)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # forget gate bias = 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        return logits

def evaluate(model, dataloader, processor, criterion):
    model.eval()
    true_labels, pred_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for sentences, labels, mask in tqdm(dataloader, desc="Avaliando", leave=False):
            sentences = sentences.to(Config.device)
            labels = labels.to(Config.device)
            mask = mask.to(Config.device)

            logits = model(sentences)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            mask_np = mask.cpu().numpy()

            for i in range(len(preds)):
                current_true = []
                current_pred = []
                for j in range(mask_np.shape[1]):
                    if mask_np[i][j] and labels_np[i][j] != -1:
                        true_label = processor.reverse_label_map[labels_np[i][j]]
                        pred_label = processor.reverse_label_map[preds[i][j]]
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

def train_fold(fold, train_loader, val_loader, processor, args, vocab_size):
    # Verificar distribuição de labels no treino
    label_counter = Counter()
    for sentences, labels, mask in train_loader:
        labels_np = labels.numpy()
        mask_np = mask.numpy()
        for i in range(len(labels_np)):
            for j in range(len(labels_np[i])):
                if mask_np[i][j] and labels_np[i][j] != -1:
                    label = processor.reverse_label_map[labels_np[i][j]]
                    label_counter[label] += 1
    logging.info(f"Distribuição de labels no treino (Fold {fold}): {dict(label_counter)}")

    # Criar modelo
    model = NERBiLSTM(
        vocab_size=vocab_size,
        embed_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.lstm_layers,
        dropout=args.dropout,
        num_labels=len(processor.labels)
    ).to(Config.device)

    # Configurar pesos de classe se solicitado
    class_weights = None
    if args.use_class_weights:
        # Calcular pesos baseados na frequência das labels
        total = sum(label_counter.values())
        num_classes = len(processor.labels)
        weights = []
        for lbl in processor.labels:
            count = label_counter.get(lbl, 0)
            if count > 0:
                weight = total / (num_classes * count)
            else:
                weight = 1.0
            weights.append(weight)
        class_weights = torch.FloatTensor(weights).to(Config.device)
        logging.info(f"Pesos de classe calculados: {class_weights}")

    criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=args.weight_decay)

    best_f1 = -1
    patience_counter = 0
    best_model_path = os.path.join(args.output_dir, f"bilstm_fold_{fold}_best.pt")

    logging.info(f"🔧 Configuração Fold {fold}:")
    logging.info(f"  Labels: {len(processor.labels)}")
    logging.info(f"  Vocab size: {vocab_size}")
    logging.info(f"  Embedding dim: {args.embedding_dim}")
    logging.info(f"  Hidden dim: {args.hidden_dim}")
    logging.info(f"  LSTM layers: {args.lstm_layers}")
    logging.info(f"  Dropout: {args.dropout}")

    global_step = 0
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        batch_count = 0

        progress_bar = tqdm(train_loader, desc=f"Fold {fold} - Época {epoch+1}", leave=False)

        for step, (sentences, labels, mask) in enumerate(progress_bar):
            sentences = sentences.to(Config.device)
            labels = labels.to(Config.device)

            optimizer.zero_grad()
            logits = model(sentences)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            global_step += 1

            if global_step % Config.logging_steps == 0:
                progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_loss / max(batch_count, 1)
        val_results = evaluate(model, val_loader, processor, criterion)
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

    return evaluate(model, val_loader, processor, criterion)

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
            logging.FileHandler(os.path.join(args.output_dir, 'bilstm_training.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Salva argumentos
    with open(os.path.join(args.output_dir, 'bilstm_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    logging.info("=" * 70)
    logging.info("🚀 INICIANDO TREINAMENTO - BiLSTM (sem CRF)")
    logging.info(f"📁 Dados: {args.data}")
    logging.info(f"📂 Saída: {args.output_dir}")
    logging.info(f"🎯 Dispositivo: {Config.device}")
    logging.info(f"🧮 Folds: {args.k_folds}")
    logging.info("=" * 70)

    try:
        processor = DataProcessor()
        examples = processor.get_examples(args.data)
        vocab_size = len(processor.word_to_idx)
        logging.info(f"📊 Total de exemplos: {len(examples)}")
        logging.info(f"🏷️  Labels únicos: {len(processor.labels)}")
        logging.info(f"📚 Tamanho do vocabulário: {vocab_size}")

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

            train_dataset = NERDataset(train_examples, processor)
            val_dataset = NERDataset(val_examples, processor)

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                drop_last=True,
                num_workers=0
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                collate_fn=collate_fn,
                num_workers=0
            )

            results = train_fold(fold+1, train_loader, val_loader, processor, args, vocab_size)

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
            with open(os.path.join(args.output_dir, f'bilstm_fold_{fold+1}_report.txt'), 'w') as f:
                f.write(f"FOLD {fold+1} - BiLSTM\n")
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
        logging.info("🎯 RESULTADOS FINAIS - BiLSTM")
        logging.info(f"{'='*70}")
        logging.info(f"📊 F1 Médio: {final_f1:.4f} (±{final_std:.4f})")
        logging.info(f"📈 F1 por fold: {[round(f, 4) for f in fold_results]}")
        logging.info(f"🏆 Melhor F1: {max(fold_results):.4f}")
        logging.info(f"📉 Pior F1: {min(fold_results):.4f}")

        final_metrics = {
            'architecture': 'BiLSTM',
            'final_f1_mean': float(final_f1),
            'final_f1_std': float(final_std),
            'folds': all_metrics,
            'args': vars(args)
        }

        with open(os.path.join(args.output_dir, 'bilstm_final_metrics.json'), 'w') as f:
            json.dump(final_metrics, f, indent=2, ensure_ascii=False)

        with open(os.path.join(args.output_dir, 'bilstm_final_results.txt'), 'w') as f:
            f.write("RESULTADOS FINAIS - BiLSTM\n")
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

        logging.info(f"\n✅ Treinamento concluído!")
        logging.info(f"📁 Resultados salvos em: {args.output_dir}")

    except Exception as e:
        logging.error(f"\n❌ ERRO CRÍTICO: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()