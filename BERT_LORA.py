import sys
import os
import argparse
import logging
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torch.optim import AdamW
from transformers import BertForTokenClassification, BertTokenizer, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from peft import LoraConfig, get_peft_model, TaskType
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Arquivo único com dados no formato IOB')
    parser.add_argument('--output_dir', required=True, help='Diretório para salvar resultados')
    parser.add_argument('--k_folds', type=int, default=5, help='Número de folds (padrão: 5)')
    parser.add_argument('--batch_size', type=int, default=16, help='Tamanho do batch (padrão: 16)')
    parser.add_argument('--epochs', type=int, default=20, help='Número de épocas (padrão: 20)')
    parser.add_argument('--patience', type=int, default=8, help='Paciência para early stopping')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Taxa de aprendizado')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA r parameter')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha parameter')
    parser.add_argument('--use_class_weights', action='store_true', help='Usar pesos de classes para lidar com desbalanceamento')
    return parser.parse_args()

class Config:
    model_name = "neuralmind/bert-base-portuguese-cased"
    max_seq_length = 180
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42

class BertLoRA(nn.Module):
    def __init__(self, model_name, num_labels, lora_r=8, lora_alpha=16, class_weights=None):
        super().__init__()
        
        self.bert = BertForTokenClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False,
            ignore_mismatched_sizes=True
        )
        
        # Aplicar pesos de classes se fornecidos
        if class_weights is not None:
            self.bert.classifier.weight = nn.Parameter(class_weights)
            logging.info("🔧 Pesos de classes aplicados à camada de classificação")
        
        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["query", "value", "key", "output.dense"]
        )
        self.bert = get_peft_model(self.bert, lora_config)
        
        # Log LoRA parameters
        trainable_params = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.bert.parameters())
        logging.info(f"🔧 LoRA - Parâmetros treináveis: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=False
        )
        
        if labels is not None:
            loss, logits = outputs
            return loss
        else:
            logits = outputs[0]
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
        # CORREÇÃO: Manter apenas labels IOB reais
        iob_labels = sorted([l for l in labels if l not in ['[PAD]', '[CLS]', '[SEP]', 'X']])
        self.labels = ['O'] + iob_labels  # O deve ser o primeiro
        self.label_map = {label: i for i, label in enumerate(self.labels)}
        logging.info(f"🏷️ Mapeamento de labels: {self.label_map}")

def convert_examples_to_features(examples, tokenizer, processor):
    features = []
    for ex_index, ex in enumerate(examples):
        tokens = []
        label_ids = []
        
        for word, label in zip(ex.words, ex.labels):
            word_tokens = tokenizer.tokenize(word) or [tokenizer.unk_token]
            tokens.extend(word_tokens)
            
            # CORREÇÃO: Usar 'O' para tokens subwords adicionais
            if label in processor.label_map:
                label_id = processor.label_map[label]
            else:
                label_id = processor.label_map['O']  # Fallback para 'O'
            
            # Primeiro token da palavra mantém o label original, demais recebem -100 (ignorar na loss)
            label_ids.append(label_id)
            for _ in range(1, len(word_tokens)):
                label_ids.append(-100)  # Ignorar na loss do BERT
        
        # Truncamento
        if len(tokens) > Config.max_seq_length - 2:
            tokens = tokens[:Config.max_seq_length - 2]
            label_ids = label_ids[:Config.max_seq_length - 2]
        
        # Adicionar [CLS] e [SEP]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        label_ids = [-100] + label_ids + [-100]  # [CLS] e [SEP] são ignorados
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # Padding
        padding_length = Config.max_seq_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + [0] * padding_length
        label_ids = label_ids + [-100] * padding_length
        
        # Máscara para avaliação (apenas primeiros tokens das palavras)
        predict_mask = [0] * Config.max_seq_length
        current_pos = 1  # Começa após [CLS]
        for word in ex.words:
            word_tokens = tokenizer.tokenize(word) or [tokenizer.unk_token]
            if current_pos < Config.max_seq_length - 1:  # -1 para [SEP]
                predict_mask[current_pos] = 1
            current_pos += len(word_tokens)
            if current_pos >= Config.max_seq_length - 1:
                break
        
        features.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_ids,
            'predict_mask': predict_mask
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

def compute_class_weights(dataloader, processor, device):
    """Calcula pesos de classes para lidar com desbalanceamento"""
    all_labels = []
    for batch in dataloader:
        labels = batch['labels'].view(-1)
        # Filtrar labels válidos (não -100)
        valid_labels = labels[labels != -100]
        all_labels.extend(valid_labels.cpu().numpy())
    
    if not all_labels:
        return None
    
    # Calcular pesos usando sklearn
    classes = np.arange(len(processor.labels))
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=all_labels
    )
    
    # Converter para tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    logging.info(f"⚖️ Pesos de classes calculados: {class_weights.cpu().numpy()}")
    return class_weights

def evaluate(model, dataloader, processor):
    model.eval()
    true_labels, pred_labels = [], []
    total_loss = 0
    loss_fct = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(Config.device)
            attention_mask = batch['attention_mask'].to(Config.device)
            labels = batch['labels'].to(Config.device)
            predict_mask = batch['predict_mask'].cpu().numpy()
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calcular loss apenas para tokens não ignorados (labels != -100)
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, logits.size(-1))[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            if len(active_labels) > 0:
                loss = loss_fct(active_logits, active_labels)
                total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels_batch = labels.cpu().numpy()
            
            for i in range(len(preds)):
                current_true = []
                current_pred = []
                for j in range(len(predict_mask[i])):
                    if predict_mask[i][j] and labels_batch[i][j] != -100:
                        true_label_idx = labels_batch[i][j]
                        pred_label_idx = preds[i][j]
                        
                        if 0 <= true_label_idx < len(processor.labels) and 0 <= pred_label_idx < len(processor.labels):
                            true_label = processor.labels[true_label_idx]
                            pred_label = processor.labels[pred_label_idx]
                            current_true.append(true_label)
                            current_pred.append(pred_label)
                
                if current_true:
                    true_labels.append(current_true)
                    pred_labels.append(current_pred)
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    # CORREÇÃO: Verificar se há predições antes de calcular métricas
    if not true_labels or not pred_labels:
        logging.warning("⚠️ Nenhuma predição válida para cálculo de métricas")
        return {
            'loss': avg_loss,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'report': "Nenhuma predição válida",
            'detailed_report': "Nenhuma predição válida",
            'true_labels': [],
            'pred_labels': []
        }
    
    try:
        f1 = f1_score(true_labels, pred_labels, zero_division=0)
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        
        # CORREÇÃO: Gerar relatório detalhado com output_dict=True
        report_dict = classification_report(true_labels, pred_labels, zero_division=0, output_dict=True)
        report_str = classification_report(true_labels, pred_labels, zero_division=0)
        
        # Criar tabela detalhada com métricas por entidade
        detailed_report = "\n📊 RELATÓRIO DETALHADO POR ENTIDADE:\n"
        detailed_report += "=" * 60 + "\n"
        detailed_report += f"{'ENTIDADE':<20} {'PRECISION':<10} {'RECALL':<10} {'F1-SCORE':<10} {'SUPPORT':<10}\n"
        detailed_report += "-" * 60 + "\n"
        
        for entity in report_dict:
            if entity not in ['micro avg', 'macro avg', 'weighted avg']:
                metrics = report_dict[entity]
                detailed_report += f"{entity:<20} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1-score']:<10.4f} {metrics['support']:<10}\n"
        
        # Adicionar médias
        if 'micro avg' in report_dict:
            metrics = report_dict['micro avg']
            detailed_report += f"{'micro avg':<20} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1-score']:<10.4f} {metrics['support']:<10}\n"
        if 'macro avg' in report_dict:
            metrics = report_dict['macro avg']
            detailed_report += f"{'macro avg':<20} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1-score']:<10.4f} {metrics['support']:<10}\n"
        if 'weighted avg' in report_dict:
            metrics = report_dict['weighted avg']
            detailed_report += f"{'weighted avg':<20} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1-score']:<10.4f} {metrics['support']:<10}\n"
        
        detailed_report += "=" * 60
        
    except Exception as e:
        logging.warning(f"⚠️ Erro no cálculo de métricas: {e}")
        f1, precision, recall, report_str, detailed_report = 0.0, 0.0, 0.0, "Erro no cálculo", f"Erro no cálculo: {e}"
    
    return {
        'loss': avg_loss,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'report': report_str,
        'detailed_report': detailed_report,
        'true_labels': true_labels,
        'pred_labels': pred_labels
    }

def train_fold(fold, train_loader, val_loader, processor, args):
    # Calcular pesos de classes se solicitado
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_loader, processor, Config.device)
    
    model = BertLoRA(
        Config.model_name, 
        len(processor.labels),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        class_weights=class_weights
    ).to(Config.device)
    
    # Ajustar learning rate baseado no fold para variação
    current_lr = args.learning_rate * (0.9 + 0.1 * (fold % 3))
    optimizer = AdamW(model.parameters(), lr=current_lr)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) // 3,
        num_training_steps=len(train_loader) * args.epochs
    )
    
    best_f1 = -1
    patience_counter = 0
    best_model_path = os.path.join(args.output_dir, f"fold_BERT_LoRA_{fold}_best.pt")
    
    # Log de informações do fold
    logging.info(f"📈 Fold {fold}: LR={current_lr:.2e}, Class Weights={args.use_class_weights}")
    
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(Config.device)
            attention_mask = batch['attention_mask'].to(Config.device)
            labels = batch['labels'].to(Config.device)
            
            loss = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Log a cada 20% do dataset
            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                current_loss = total_loss / batch_count
                logging.info(f"Fold {fold}, Época {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}: Loss = {current_loss:.4f}")
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Avaliação
        val_results = evaluate(model, val_loader, processor)
        
        epoch_time = time.time() - start_time
        
        logging.info(f"\n=== Fold {fold}, Época {epoch+1} ===")
        logging.info(f"Tempo: {epoch_time:.2f}s")
        logging.info(f"Loss Treino: {avg_train_loss:.4f}")
        logging.info(f"Loss Validação: {val_results['loss']:.4f}")
        logging.info(f"F1 Validação: {val_results['f1']:.4f}")
        logging.info(f"Precision Validação: {val_results['precision']:.4f}")
        logging.info(f"Recall Validação: {val_results['recall']:.4f}")
        
        # CORREÇÃO: Exibir relatório detalhado
        logging.info(f"\n{val_results['detailed_report']}")
        
        # Análise de progresso para entidades problemáticas
        if val_results['f1'] > 0:
            entity_progress = []
            try:
                report_dict = classification_report(val_results['true_labels'], val_results['pred_labels'], 
                                                  zero_division=0, output_dict=True)
                for entity in report_dict:
                    if entity not in ['micro avg', 'macro avg', 'weighted avg'] and report_dict[entity]['support'] > 0:
                        f1_entity = report_dict[entity]['f1-score']
                        if f1_entity == 0:
                            entity_progress.append(f"{entity}({report_dict[entity]['support']})")
                
                if entity_progress:
                    logging.info(f"🔍 Entidades com F1 zero: {', '.join(entity_progress)}")
            except:
                pass
        
        if val_results['f1'] > best_f1:
            best_f1 = val_results['f1']
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"🎯 NOVO MELHOR F1: {best_f1:.4f} (modelo salvo)")
            
            # Salvar predições do melhor modelo
            if val_results['f1'] > 0.1:  # Só salvar se for um modelo razoável
                pred_file = os.path.join(args.output_dir, f"fold_{fold}_best_predictions.txt")
                with open(pred_file, 'w', encoding='utf-8') as f:
                    f.write(f"Melhor F1: {best_f1:.4f}\n\n")
                    for i, (true, pred) in enumerate(zip(val_results['true_labels'][:10], val_results['pred_labels'][:10])):
                        f.write(f"Exemplo {i}:\n")
                        f.write(f"Verdadeiro: {true}\n")
                        f.write(f"Previsto:   {pred}\n\n")
        else:
            patience_counter += 1
            logging.info(f"⏳ F1 não melhorou. Paciência: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                logging.info(f"🛑 Early stopping na época {epoch+1}")
                break
        
        logging.info("-" * 80)
    
    # Carregar o melhor modelo para avaliação final
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logging.info(f"✅ Fold {fold}: Melhor modelo carregado com F1 = {best_f1:.4f}")
    
    final_results = evaluate(model, val_loader, processor)
    
    # Resumo do fold
    logging.info(f"\n📊 RESUMO FOLD {fold}")
    logging.info(f"Melhor F1: {best_f1:.4f}")
    logging.info(f"F1 Final: {final_results['f1']:.4f}")
    logging.info(f"\n{final_results['detailed_report']}")
    
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
            logging.FileHandler(os.path.join(args.output_dir, 'resultados_bert_lora.txt')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    try:
        logging.info("🚀 INICIANDO TREINAMENTO BERT + LoRA")
        logging.info(f"📁 Dados: {args.data}")
        logging.info(f"📂 Output: {args.output_dir}")
        logging.info(f"🎯 Folds: {args.k_folds}")
        logging.info(f"📦 Batch: {args.batch_size}")
        logging.info(f"📈 Épocas: {args.epochs}")
        logging.info(f"💪 Paciência: {args.patience}")
        logging.info(f"📚 Learning Rate: {args.learning_rate}")
        logging.info(f"🔧 LoRA r: {args.lora_r}")
        logging.info(f"🔧 LoRA alpha: {args.lora_alpha}")
        logging.info(f"⚖️ Class Weights: {args.use_class_weights}")
        logging.info(f"⚡ Device: {Config.device}")
        
        processor = DataProcessor()
        examples = processor.get_examples(args.data)
        tokenizer = BertTokenizer.from_pretrained(Config.model_name)
        
        logging.info(f"📊 Total de exemplos: {len(examples)}")
        logging.info(f"🏷️ Labels: {processor.labels}")
        
        # Verificar distribuição de labels
        label_counts = {}
        for ex in examples:
            for label in ex.labels:
                label_counts[label] = label_counts.get(label, 0) + 1
        logging.info(f"📈 Distribuição de labels: {label_counts}")
        
        # Calcular estatísticas de desbalanceamento
        total_entities = sum(count for label, count in label_counts.items() if label != 'O')
        logging.info(f"📊 Total de entidades (não-O): {total_entities}")
        
        for label, count in label_counts.items():
            if label != 'O':
                percentage = (count / total_entities) * 100
                logging.info(f"   {label}: {count} ({percentage:.1f}%)")
        
        kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=Config.seed)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(examples)):
            logging.info(f"\n{'='*60}")
            logging.info(f"🔄 INICIANDO FOLD {fold+1}/{args.k_folds}")
            logging.info(f"{'='*60}")
            
            train_examples = [examples[i] for i in train_idx]
            val_examples = [examples[i] for i in val_idx]
            
            logging.info(f"📚 Treino: {len(train_examples)} exemplos")
            logging.info(f"🧪 Validação: {len(val_examples)} exemplos")
            
            train_dataset = NerDataset(train_examples, tokenizer, processor)
            val_dataset = NerDataset(val_examples, tokenizer, processor)
            
            train_loader = data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True
            )
            val_loader = data.DataLoader(
                val_dataset,
                batch_size=args.batch_size
            )
            
            results = train_fold(fold+1, train_loader, val_loader, processor, args)
            
            logging.info(f"\n✅ FOLD {fold+1} FINALIZADO")
            logging.info(f"F1-Score: {results['f1']:.4f}")
            logging.info(f"Precision: {results['precision']:.4f}")
            logging.info(f"Recall: {results['recall']:.4f}")
            
            fold_results.append(results['f1'])
        
        logging.info(f"\n{'='*60}")
        logging.info("🎯 RESULTADO FINAL")
        logging.info(f"{'='*60}")
        logging.info(f"F1 Médio: {np.mean(fold_results):.4f} (±{np.std(fold_results):.4f})")
        logging.info(f"Valores por Fold: {[round(f, 4) for f in fold_results]}")
        
        # Salvar métricas finais
        with open(os.path.join(args.output_dir, 'metricas_finais_bert_lora.txt'), 'w') as f:
            f.write(f"F1 Médio: {np.mean(fold_results):.4f} (±{np.std(fold_results):.4f})\n")
            f.write(f"Valores por Fold: {[round(f, 4) for f in fold_results]}\n")
            f.write(f"Melhor Fold: {np.argmax(fold_results) + 1} (F1 = {max(fold_results):.4f})\n")
            f.write(f"LoRA r: {args.lora_r}\n")
            f.write(f"LoRA alpha: {args.lora_alpha}\n")
            f.write(f"Class Weights: {args.use_class_weights}\n")
    
    except Exception as e:
        logging.error(f"❌ Erro: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()