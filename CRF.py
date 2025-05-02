import argparse
import sys
import sklearn_crfsuite
from seqeval.metrics import classification_report, f1_score
from sklearn.model_selection import KFold
import pandas as pd

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
    return sentences, labels

def word2features(sent, i):
    word = sent[i]
    features = {
        'word.lower()': word.lower(),
        'is_upper': word.isupper(),
        'is_title': word.istitle(),
        'is_digit': word.isdigit(),
        'prefix-3': word[:3],
        'suffix-3': word[-3:],
        'word_length': len(word),
    }
    if i > 0:
        word_prev = sent[i - 1]
        features.update({
            '-1:word.lower()': word_prev.lower(),
            '-1:is_upper': word_prev.isupper(),
            '-1:is_title': word_prev.istitle(),
        })
    else:
        features['BOS'] = True
    if i < len(sent) - 1:
        word_next = sent[i + 1]
        features.update({
            '+1:word.lower()': word_next.lower(),
            '+1:is_upper': word_next.isupper(),
            '+1:is_title': word_next.istitle(),
        })
    else:
        features['EOS'] = True
    return features

def extract_features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Caminho para o arquivo IOB completo (sem separação)')
    parser.add_argument('--output', required=True, help='Arquivo de saída para resultados')
    parser.add_argument('--model_name', required=True, help='Nome do modelo (ex: CRF)')
    parser.add_argument('--k', type=int, default=5, help='Número de folds (default=5)')
    args = parser.parse_args()

    with open(args.output, 'w', encoding='utf-8') as output_file:
        output_file.write(f"=== Modelo: {args.model_name} | K-Fold: {args.k} ===\n\n")
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, output_file)

        try:
            print("Carregando dados completos...")
            all_sentences, all_labels = read_data(args.data)

            print("Iniciando K-Fold Cross-Validation...")
            kf = KFold(n_splits=args.k, shuffle=True, random_state=42)

            f1_scores = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(all_sentences)):
                print(f"\n===== Fold {fold + 1}/{args.k} =====")

                train_sents = [all_sentences[i] for i in train_idx]
                train_labs = [all_labels[i] for i in train_idx]
                val_sents = [all_sentences[i] for i in val_idx]
                val_labs = [all_labels[i] for i in val_idx]

                X_train = [extract_features(s) for s in train_sents]
                X_val = [extract_features(s) for s in val_sents]

                print("Treinando modelo CRF...")
                crf = sklearn_crfsuite.CRF(
                    algorithm='lbfgs',
                    c1=0.1,
                    c2=0.1,
                    max_iterations=100,
                    all_possible_transitions=True
                )
                crf.fit(X_train, train_labs)

                print("Avaliando no fold atual...")
                y_pred = crf.predict(X_val)
                report = classification_report(val_labs, y_pred)
                f1 = f1_score(val_labs, y_pred)
                f1_scores.append(f1)

                print(report)
                print(f"F1-score do Fold {fold + 1}: {f1:.4f}")

            # Resultados finais
            print("\n===== Resultados Finais =====")
            for i, score in enumerate(f1_scores):
                print(f"Fold {i+1}: F1 = {score:.4f}")
            print(f"Média: {sum(f1_scores)/len(f1_scores):.4f}")

            # Salva F1 em CSV para análise posterior (ex: Friedman/Nemenyi)
            df = pd.DataFrame({args.model_name: f1_scores})
            df.to_csv(args.model_name + "_kfold_scores.csv", index=False)

        finally:
            sys.stdout = original_stdout

if __name__ == '__main__':
    main()
