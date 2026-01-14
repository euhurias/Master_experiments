import sys
import os
import argparse
import logging
import numpy as np
import torch
from torch import nn
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
    parser.add_argument('--data', required=True, help='Arquivo único com dados no formato IOB')
    parser.add_argument('--output_dir', required=True, help='Diretório para salvar resultados')
    parser.add_argument('--model_name', type=str, default='google/electra-base-discriminator',
                       help='Nome do modelo ELECTRA (padrão: google/electra-base-discriminator)')
    parser.add_argument('--k_folds', type=int, default=5, help='Número de folds (padrão: 5)')
    parser.add_argument('--batch_size', type=int, default=16, help='Tamanho do batch (padrão: 16)')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas (padrão: 10)')
    parser.add_argument('--patience', type=int, default=5, help='Paciência para early stopping')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Taxa de aprendizado')
    parser.add_argument('--max_seq_length', type=int, default=128, help='Comprimento máximo da sequência')
    parser.add_argument('--freeze_bert', action='store_true', help='Congelar camadas do ELECTRA')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Decaimento de peso')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Passos de warmup')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Passos de acumulação de gradiente')
    return parser.parse_args()

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    logging_steps = 50
    max_grad_norm = 1.0

class ElectraNER(nn.Module):
    def __init__(self, model_name, num_labels, freeze_bert=False):
        super(ElectraNER, self).__init__()
        
        # Carrega modelo ELECTRA pré-treinado
        self.electra = ElectraModel.from_pretrained(model_name)
        self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
        self.hidden_size = self.electra.config.hidden_size
        
        # Congela camadas do ELECTRA se solicitado
        if freeze_bert:
            for param in self.electra.parameters():
                param.requires_grad = False
        
        # Dropout para regularização
        self.dropout = nn.Dropout(self.electra.config.hidden_dropout_prob)
        
        # Classificador para NER
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        # Perda
        self.loss_fct = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        # Obtém embeddings do ELECTRA
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            # Calcula loss apenas para tokens não especiais
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

class InputFeatures:
    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids, predict_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids
        self.predict_mask = predict_mask

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
        # Labels especiais - ESSA É A PARTE CORRIGIDA!
        # [PAD] deve ser o primeiro (índice 0) pois é usado para padding
        special_labels = ['[PAD]', '[CLS]', '[SEP]', 'X', 'O']
        
        # Remove duplicatas e adiciona labels de entidade
        non_special = sorted([l for l in labels if l not in special_labels])
        
        # A lista completa: labels especiais primeiro, depois entidades
        self.labels = special_labels + non_special
        self.label_map = {label: i for i, label in enumerate(self.labels)}
        self.reverse_label_map = {i: label for i, label in enumerate(self.labels)}
        
        logging.info(f"Labels mapeados ({len(self.labels)}):")
        for label, idx in list(self.label_map.items())[:10]:  # Mostra os primeiros 10
            logging.info(f"  {label} -> {idx}")
        if len(self.labels) > 10:
            logging.info(f"  ... e mais {len(self.labels) - 10} labels")

def convert_examples_to_features(examples, processor, tokenizer, max_seq_length):
    features = []
    
    for ex_idx, example in enumerate(examples):
        words = example.words
        labels = example.labels
        
        # Tokenização (necessário para alinhar palavras com subwords)
        tokens = []
        label_ids = []
        
        for word, label in zip(words, labels):
            word_tokens = tokenizer.tokenize(word)
            
            if not word_tokens:
                word_tokens = [tokenizer.unk_token]
            
            tokens.extend(word_tokens)
            # Usa o mesmo label para o primeiro token da palavra e "X" para subwords seguintes
            if label in processor.label_map:
                main_label = processor.label_map[label]
            else:
                main_label = processor.label_map['O']  # Fallback para 'O'
            
            label_ids.append(main_label)
            label_ids.extend([processor.label_map['X']] * (len(word_tokens) - 1))
        
        # Truncamento se necessário
        if len(tokens) > max_seq_length - 2:  # -2 para [CLS] e [SEP]
            tokens = tokens[:max_seq_length - 2]
            label_ids = label_ids[:max_seq_length - 2]
        
        # Adiciona tokens especiais
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        label_ids = [processor.label_map['[CLS]']] + label_ids + [processor.label_map['[SEP]']]
        
        # Converte tokens para IDs
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # Máscara de atenção (1 para tokens reais, 0 para padding)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        
        # Padding
        padding_length = max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            token_type_ids = token_type_ids + [0] * padding_length
            label_ids = label_ids + [processor.label_map['[PAD]']] * padding_length
        
        # Máscara para tokens reais (não [CLS], [SEP], [PAD], ou X)
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
        
        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label_ids=label_ids,
                predict_mask=predict_mask
            )
        )
    
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
            'input_ids': torch.tensor(feat.input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(feat.attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(feat.token_type_ids, dtype=torch.long),
            'labels': torch.tensor(feat.label_ids, dtype=torch.long),
            'predict_mask': torch.tensor(feat.predict_mask, dtype=torch.bool)
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
                    if predict_mask[i][j]:  # Apenas tokens para predição
                        true_label_id = labels_np[i][j]
                        pred_label_id = preds[i][j]
                        
                        # Converte IDs para nomes de labels
                        true_label = processor.reverse_label_map[true_label_id]
                        pred_label = processor.reverse_label_map[pred_label_id]
                        
                        # Ignora labels especiais nas métricas
                        if true_label not in ['[PAD]', '[CLS]', '[SEP]', 'X']:
                            current_true.append(true_label)
                            current_pred.append(pred_label)
                
                if current_true:
                    true_labels.append(current_true)
                    pred_labels.append(current_pred)
    
    avg_loss = total_loss / max(len(dataloader), 1)
    
    # Calcula métricas
    if true_labels and pred_labels:
        try:
            f1 = f1_score(true_labels, pred_labels, zero_division=0)
            precision = precision_score(true_labels, pred_labels, zero_division=0)
            recall = recall_score(true_labels, pred_labels, zero_division=0)
            report = classification_report(true_labels, pred_labels, zero_division=0)
        except Exception as e:
            logging.warning(f"Erro ao calcular métricas: {e}")
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
    # Inicializa modelo
    model = ElectraNER(
        model_name=args.model_name,
        num_labels=len(processor.labels),
        freeze_bert=args.freeze_bert
    ).to(Config.device)
    
    # Otimizador
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0,
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    
    # Agendador de learning rate
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    best_f1 = -1
    patience_counter = 0
    best_model_path = os.path.join(args.output_dir, f"fold_{fold}_best.pt")
    
    logging.info(f"🔧 Configuração Fold {fold}:")
    logging.info(f"  Labels: {len(processor.labels)}")
    logging.info(f"  Batch size: {args.batch_size}")
    logging.info(f"  Learning rate: {args.learning_rate}")
    logging.info(f"  Congelar ELECTRA: {args.freeze_bert}")
    
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
            
            # Logging
            if global_step % Config.logging_steps == 0:
                progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / max(batch_count, 1)
        
        # Avaliação
        val_results = evaluate(model, val_loader, processor, tokenizer)
        epoch_time = time.time() - epoch_start_time
        
        logging.info(f"\n=== Fold {fold}, Época {epoch+1} ===")
        logging.info(f"⏱️  Tempo: {epoch_time:.2f}s")
        logging.info(f"📉 Loss Treino: {avg_train_loss:.4f}")
        logging.info(f"📊 Loss Validação: {val_results['loss']:.4f}")
        logging.info(f"🎯 F1 Validação: {val_results['f1']:.4f}")
        logging.info(f"🎯 Precisão: {val_results['precision']:.4f}")
        logging.info(f"🎯 Recall: {val_results['recall']:.4f}")
        
        # Salva o melhor modelo
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
    
    # Carrega o melhor modelo
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=Config.device))
        logging.info(f"📦 Melhor modelo carregado (F1: {best_f1:.4f})")
    
    return evaluate(model, val_loader, processor, tokenizer)

def main():
    args = parse_args()
    
    # Configurações iniciais
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.seed)
    
    # Cria diretório de saída
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configura logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'electra_training.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Salva argumentos
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logging.info("=" * 70)
    logging.info("🚀 INICIANDO TREINAMENTO COM ELECTRA")
    logging.info(f"📁 Dados: {args.data}")
    logging.info(f"📂 Saída: {args.output_dir}")
    logging.info(f"🤖 Modelo: {args.model_name}")
    logging.info(f"🎯 Dispositivo: {Config.device}")
    logging.info(f"🧮 Folds: {args.k_folds}")
    logging.info(f"📏 Seq length: {args.max_seq_length}")
    logging.info(f"🔁 Gradient accumulation: {args.gradient_accumulation_steps}")
    logging.info("=" * 70)
    
    try:
        # Carrega tokenizer
        tokenizer = ElectraTokenizer.from_pretrained(args.model_name)
        
        # Processa dados
        processor = DataProcessor()
        examples = processor.get_examples(args.data)
        logging.info(f"📊 Total de exemplos: {len(examples)}")
        logging.info(f"🏷️  Labels únicos: {len(processor.labels)}")
        
        # Validação cruzada
        kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=Config.seed)
        fold_results = []
        all_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(examples)):
            fold_start_time = time.time()
            
            logging.info(f"\n{'='*60}")
            logging.info(f"🔄 INICIANDO FOLD {fold+1}/{args.k_folds}")
            logging.info(f"{'='*60}")
            
            # Divide dados
            train_examples = [examples[i] for i in train_idx]
            val_examples = [examples[i] for i in val_idx]
            
            # Cria datasets
            train_dataset = NerDataset(
                train_examples, processor, tokenizer, args.max_seq_length
            )
            val_dataset = NerDataset(
                val_examples, processor, tokenizer, args.max_seq_length
            )
            
            # Cria dataloaders
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
            
            # Treina fold
            results = train_fold(
                fold+1, 
                train_loader, 
                val_loader, 
                processor, 
                tokenizer,
                args
            )
            
            # Salva resultados
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
            
            # Salva relatório detalhado
            with open(os.path.join(args.output_dir, f'fold_{fold+1}_report_electra.txt'), 'w') as f:
                f.write(f"FOLD {fold+1} - ELECTRA NER\n")
                f.write("="*50 + "\n")
                f.write(f"F1: {results['f1']:.4f}\n")
                f.write(f"Precision: {results['precision']:.4f}\n")
                f.write(f"Recall: {results['recall']:.4f}\n")
                f.write(f"Loss: {results['loss']:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(results['report'])
        
        # Resultados finais
        final_f1 = np.mean(fold_results)
        final_std = np.std(fold_results)
        
        logging.info(f"\n{'='*70}")
        logging.info("🎯 RESULTADOS FINAIS - ELECTRA")
        logging.info(f"{'='*70}")
        logging.info(f"📊 F1 Médio: {final_f1:.4f} (±{final_std:.4f})")
        logging.info(f"📈 F1 por fold: {[round(f, 4) for f in fold_results]}")
        logging.info(f"🏆 Melhor F1: {max(fold_results):.4f} (Fold {fold_results.index(max(fold_results))+1})")
        logging.info(f"📉 Pior F1: {min(fold_results):.4f} (Fold {fold_results.index(min(fold_results))+1})")
        
        # Salva métricas finais
        final_metrics = {
            'model': args.model_name,
            'final_f1_mean': float(final_f1),
            'final_f1_std': float(final_std),
            'folds': all_metrics,
            'args': vars(args)
        }
        
        with open(os.path.join(args.output_dir, 'final_metrics_electra.json'), 'w') as f:
            json.dump(final_metrics, f, indent=2, ensure_ascii=False)
        
        with open(os.path.join(args.output_dir, 'final_results_electra.txt'), 'w') as f:
            f.write("RESULTADOS FINAIS - ELECTRA NER\n")
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