import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from seqeval.metrics import classification_report, f1_score
from sklearn.model_selection import KFold
from torchcrf import CRF

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, text):
        for f in self.files:
            f.write(text)
    def flush(self):
        for f in self.files:
            if hasattr(f, 'flush'):
                f.flush()

def read_data(file_path):
    sentences, labels, sentence, label = [], [], [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                word, tag = line.strip().split()
                sentence.append(word)
                label.append(tag)
            else:
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence, label = [], []
    if sentence:
        sentences.append(sentence)
        labels.append(label)
    return sentences, labels

def collate_fn(batch):
    sentences, labels = zip(*batch)
    sentences_padded = pad_sequence([torch.tensor(s) for s in sentences], batch_first=True, padding_value=0)
    labels_padded = pad_sequence([torch.tensor(l) for l in labels], batch_first=True, padding_value=-1)
    mask = (sentences_padded != 0)
    return sentences_padded, labels_padded, mask

class NERDataset(Dataset):
    def __init__(self, sentences, labels, word_to_idx, label_to_idx):
        self.sentences = [[word_to_idx.get(w, word_to_idx["<UNK>"]) for w in s] for s in sentences]
        self.labels = [[label_to_idx[l] for l in label_seq] for label_seq in labels]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx]), torch.tensor(self.labels[idx])

class BiLSTMCRFModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, label_count):
        super(BiLSTMCRFModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, label_count)
        self.crf = CRF(label_count, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        emissions = self.fc(lstm_out)
        return emissions

    def loss(self, emissions, tags, mask):
        return -self.crf(emissions, tags, mask=mask)

    def predict(self, emissions, mask):
        return self.crf.decode(emissions, mask=mask)

def run_fold(train_sentences, train_labels, val_sentences, val_labels, word_to_idx, label_to_idx, idx_to_label, args):
    train_data = NERDataset(train_sentences, train_labels, word_to_idx, label_to_idx)
    val_data = NERDataset(val_sentences, val_labels, word_to_idx, label_to_idx)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = BiLSTMCRFModel(len(word_to_idx), embed_dim=100, hidden_dim=128, label_count=len(label_to_idx))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for sentences, labels, mask in train_loader:
            optimizer.zero_grad()
            emissions = model(sentences)
            loss = model.loss(emissions, labels, mask)
            loss.backward()
            optimizer.step()

    # Avaliação
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sentences, labels, mask in val_loader:
            emissions = model(sentences)
            preds = model.predict(emissions, mask)
            for p, l, m in zip(preds, labels, mask):
                valid = m.cpu().numpy().astype(bool)
                all_preds.append([idx_to_label[idx] for idx, v in zip(p, valid) if v])
                all_labels.append([idx_to_label[idx.item()] for idx, v in zip(l, valid) if v])

    f1 = f1_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    return f1, report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Arquivo único com os dados (formato IOB)')
    parser.add_argument('--output', required=True, help='Arquivo de saída dos resultados')
    parser.add_argument('--k', type=int, default=5, help='Número de folds (default: 5)')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    sentences, labels = read_data(args.data)
    kf = KFold(n_splits=args.k, shuffle=True, random_state=42)

    with open(args.output, 'w', encoding='utf-8') as out:
        sys.stdout = Tee(sys.stdout, out)
        print(f"=== Validação Cruzada BiLSTM-CRF - {args.k} folds ===\n")

        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(sentences)):
            print(f"\n--- Fold {fold + 1}/{args.k} ---")

            train_sentences = [sentences[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_sentences = [sentences[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]

            words = set(w for s in train_sentences + val_sentences for w in s)
            tags = set(l for seq in train_labels + val_labels for l in seq)
            tags.add('O')

            word_to_idx = {word: i + 2 for i, word in enumerate(words)}
            word_to_idx["<PAD>"] = 0
            word_to_idx["<UNK>"] = 1
            label_to_idx = {tag: i for i, tag in enumerate(sorted(tags))}
            idx_to_label = {i: tag for tag, i in label_to_idx.items()}

            f1, report = run_fold(train_sentences, train_labels, val_sentences, val_labels, word_to_idx, label_to_idx, idx_to_label, args)
            fold_scores.append(f1)
            print(report)

        print(f"\n=== F1 por fold: {fold_scores}")
        print(f"=== F1 médio: {sum(fold_scores)/len(fold_scores):.4f}")

if __name__ == '__main__':
    main()
