import sys
import os
import argparse
import logging
import time
import json
import numpy as np
import sklearn_crfsuite
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Arquivo com dados no formato IOB')
    parser.add_argument('--output_dir', required=True, help='Diretório para salvar resultados')
    parser.add_argument('--model_name', type=str, default='CRF', help='Nome do modelo (para identificação)')
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--c1', type=float, default=0.1, help='Coeficiente L1 (regularização)')
    parser.add_argument('--c2', type=float, default=0.1, help='Coeficiente L2 (regularização)')
    parser.add_argument('--max_iterations', type=int, default=100, help='Número máximo de iterações')
    parser.add_argument('--algorithm', type=str, default='lbfgs', choices=['lbfgs', 'l2sgd', 'ap', 'pa', 'arow'],
                        help='Algoritmo de otimização')
    parser.add_argument('--all_possible_transitions', action='store_true', default=True,
                        help='Considerar todas as transições possíveis')
    return parser.parse_args()

class Config:
    seed = 42
    # CRF não usa dispositivo, mas mantemos para consistência
    device = 'cpu'

def read_data(file_path):
    """Lê arquivo IOB e retorna listas de sentenças (listas de palavras) e labels."""
    sentences, labels = [], []
    sentence, label = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 2:
                    word, tag = parts[0], parts[-1].upper()
                    sentence.append(word)
                    label.append(tag)
            else:
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence, label = [], []
    if sentence:  # última sentença se arquivo não termina com linha em branco
        sentences.append(sentence)
        labels.append(label)
    return sentences, labels

def word2features(sent, i):
    """Extrai features para a palavra na posição i da sentença."""
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
    """Retorna lista de features para cada token da sentença."""
    return [word2features(sent, i) for i in range(len(sent))]

def evaluate_model(crf, X_val, y_val):
    """Avalia o modelo CRF no conjunto de validação e retorna métricas."""
    y_pred = crf.predict(X_val)
    if y_val and y_pred:
        try:
            f1 = f1_score(y_val, y_pred, zero_division=0)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            report = classification_report(y_val, y_pred, zero_division=0)
        except:
            f1 = precision = recall = 0.0
            report = "Erro ao calcular métricas"
    else:
        f1 = precision = recall = 0.0
        report = "Nenhuma predição válida"
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'report': report
    }

def train_fold(fold, train_sents, train_labels, val_sents, val_labels, args):
    """Treina um modelo CRF para um fold específico e retorna métricas."""
    logging.info(f"Extraindo features para o fold {fold}...")
    X_train = [extract_features(s) for s in tqdm(train_sents, desc=f"Fold {fold} - Features treino", leave=False)]
    X_val = [extract_features(s) for s in tqdm(val_sents, desc=f"Fold {fold} - Features val", leave=False)]

    logging.info(f"Treinando modelo CRF (fold {fold})...")
    crf = sklearn_crfsuite.CRF(
        algorithm=args.algorithm,
        c1=args.c1,
        c2=args.c2,
        max_iterations=args.max_iterations,
        all_possible_transitions=args.all_possible_transitions,
        verbose=False  # evitar poluir o log
    )
    start_time = time.time()
    crf.fit(X_train, train_labels)
    train_time = time.time() - start_time
    logging.info(f"Treinamento concluído em {train_time:.2f}s")

    # Avaliação
    results = evaluate_model(crf, X_val, val_labels)
    results['train_time'] = train_time
    return results, crf

def main():
    args = parse_args()

    # Seeds (para reprodutibilidade do KFold)
    np.random.seed(Config.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Configura logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'crf_training.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Salva argumentos
    with open(os.path.join(args.output_dir, 'crf_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    logging.info("=" * 70)
    logging.info("🚀 INICIANDO TREINAMENTO - CRF (sklearn-crfsuite)")
    logging.info(f"📁 Dados: {args.data}")
    logging.info(f"📂 Saída: {args.output_dir}")
    logging.info(f"🧮 Folds: {args.k_folds}")
    logging.info("=" * 70)

    try:
        # Carrega dados
        logging.info("Carregando dados...")
        all_sentences, all_labels = read_data(args.data)
        logging.info(f"📊 Total de sentenças: {len(all_sentences)}")
        # Determinar labels únicas (para informação)
        unique_labels = set(l for seq in all_labels for l in seq)
        logging.info(f"🏷️  Labels únicos: {len(unique_labels)}")

        kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=Config.seed)
        fold_results = []
        all_metrics = []
        total_start_time = time.time()

        for fold, (train_idx, val_idx) in enumerate(kf.split(all_sentences)):
            fold_start_time = time.time()

            logging.info(f"\n{'='*60}")
            logging.info(f"🔄 INICIANDO FOLD {fold+1}/{args.k_folds}")
            logging.info(f"{'='*60}")

            train_sents = [all_sentences[i] for i in train_idx]
            train_labels = [all_labels[i] for i in train_idx]
            val_sents = [all_sentences[i] for i in val_idx]
            val_labels = [all_labels[i] for i in val_idx]

            logging.info(f"📚 Treino: {len(train_sents)} sentenças")
            logging.info(f"🧪 Validação: {len(val_sents)} sentenças")

            # Treinar fold
            results, _ = train_fold(fold+1, train_sents, train_labels, val_sents, val_labels, args)

            fold_time = time.time() - fold_start_time
            fold_results.append(results['f1'])
            all_metrics.append({
                'fold': fold+1,
                'f1': results['f1'],
                'precision': results['precision'],
                'recall': results['recall'],
                'train_time': results.get('train_time', 0),
                'total_time': fold_time
            })

            logging.info(f"\n📊 RESULTADOS FOLD {fold+1}:")
            logging.info(f"  F1: {results['f1']:.4f}")
            logging.info(f"  Precisão: {results['precision']:.4f}")
            logging.info(f"  Recall: {results['recall']:.4f}")
            logging.info(f"  Tempo treino: {results.get('train_time', 0):.2f}s")
            logging.info(f"  Tempo total fold: {fold_time:.2f}s")

            # Salva relatório do fold
            with open(os.path.join(args.output_dir, f'crf_fold_{fold+1}_report.txt'), 'w') as f:
                f.write(f"FOLD {fold+1} - CRF\n")
                f.write("="*50 + "\n")
                f.write(f"F1: {results['f1']:.4f}\n")
                f.write(f"Precision: {results['precision']:.4f}\n")
                f.write(f"Recall: {results['recall']:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(results['report'])

        total_time = time.time() - total_start_time
        final_f1 = np.mean(fold_results)
        final_std = np.std(fold_results)

        logging.info(f"\n{'='*70}")
        logging.info("🎯 RESULTADOS FINAIS - CRF")
        logging.info(f"{'='*70}")
        logging.info(f"⏱️  Tempo total: {total_time:.2f}s")
        logging.info(f"📊 F1 Médio: {final_f1:.4f} (±{final_std:.4f})")
        logging.info(f"📈 F1 por fold: {[round(f, 4) for f in fold_results]}")
        logging.info(f"🏆 Melhor F1: {max(fold_results):.4f}")
        logging.info(f"📉 Pior F1: {min(fold_results):.4f}")

        # Salva CSV com F1 por fold (para testes estatísticos)
        df = pd.DataFrame({args.model_name: fold_results})
        csv_path = os.path.join(args.output_dir, f'{args.model_name}_kfold_scores.csv')
        df.to_csv(csv_path, index=False)
        logging.info(f"📁 CSV com F1 por fold salvo em: {csv_path}")

        final_metrics = {
            'architecture': 'CRF',
            'model': args.model_name,
            'final_f1_mean': float(final_f1),
            'final_f1_std': float(final_std),
            'folds': all_metrics,
            'total_time_seconds': total_time,
            'args': vars(args)
        }

        with open(os.path.join(args.output_dir, 'crf_final_metrics.json'), 'w') as f:
            json.dump(final_metrics, f, indent=2, ensure_ascii=False)

        with open(os.path.join(args.output_dir, 'crf_final_results.txt'), 'w') as f:
            f.write("RESULTADOS FINAIS - CRF\n")
            f.write("="*60 + "\n")
            f.write(f"Modelo: {args.model_name}\n")
            f.write(f"F1 Médio: {final_f1:.4f} (±{final_std:.4f})\n\n")
            f.write("Folds detalhados:\n")
            for m in all_metrics:
                f.write(f"\n  Fold {m['fold']}:\n")
                f.write(f"    F1: {m['f1']:.4f}\n")
                f.write(f"    Precision: {m['precision']:.4f}\n")
                f.write(f"    Recall: {m['recall']:.4f}\n")
                f.write(f"    Tempo treino: {m.get('train_time', 0):.2f}s\n")
                f.write(f"    Tempo total fold: {m.get('total_time', 0):.2f}s\n")

        logging.info(f"\n✅ Treinamento concluído!")
        logging.info(f"📁 Resultados salvos em: {args.output_dir}")

    except Exception as e:
        logging.error(f"\n❌ ERRO CRÍTICO: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()