import sys
import os
import argparse
import logging
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torch.optim import AdamW
from transformers import LlamaForCausalLM, LlamaTokenizer, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Arquivo único com dados no formato IOB')
    parser.add_argument('--output_dir', required=True, help='Diretório para salvar resultados')
    parser.add_argument('--k_folds', type=int, default=5, help='Número de folds (padrão: 5)')
    parser.add_argument('--batch_size', type=int, default=8, help='Tamanho do batch (padrão: 8) - menor para LLaMA')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas (padrão: 10)')
    parser.add_argument('--patience', type=int, default=3, help='Paciência para early stopping')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Taxa de aprendizado')
    parser.add_argument('--max_length', type=int, default=256, help='Comprimento máximo da sequência')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Usar gradient checkpointing para economizar memória')
    return parser.parse_args()

class Config:
    llama_model_name = "huggyllama/llama-7b"   # Ajuste conforme seu ambiente
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42

class LlamaForNER(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        
        try:
            # Tentar carregar com configurações de economia de memória
            self.llama = LlamaForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True,
                offload_folder="./offload"
            )
        except Exception as e:
            logging.warning(f"❌ Não foi possível carregar com 8-bit: {e}")
            logging.info("🔄 Tentando carregar sem quantização...")
            self.llama = LlamaForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        # Congelar parâmetros do modelo base para fine-tuning eficiente
        for param in self.llama.parameters():
            param.requires_grad = False
        
        # Descongelar apenas as últimas camadas
        num_layers = len(self.llama.model.layers)
        layers_to_train = 4  # Número de camadas para fine-tuning
        for i in range(num_layers - layers_to_train, num_layers):
            for param in self.llama.model.layers[i].parameters():
                param.requires_grad = True
        
        # Descongelar o head do modelo
        for param in self.llama.lm_head.parameters():
            param.requires_grad = True
        
        # Classifier para NER
        self.classifier = nn.Linear(self.llama.config.hidden_size, num_labels)
        
        # Contar parâmetros treináveis
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"🔧 Llama Puro - Parâmetros treináveis: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.llama(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        
        # Pegar os hidden states da última camada
        logits = self.classifier(hidden_states)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.size(-1))[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss
        
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
        special_labels = ['[PAD]', '[CLS]', '[SEP]', 'X']
        self.labels = special_labels + sorted([l for l in labels if l not in special_labels])
        self.label_map = {label: i for i, label in enumerate(self.labels)}

def convert_examples_to_features(examples, tokenizer, processor, max_length):
    features = []
    for ex in examples:
        tokens = []
        label_ids = []
        predict_mask = []
        
        for word, label in zip(ex.words, ex.labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:  # Se não houver tokens, usar token desconhecido
                word_tokens = [tokenizer.unk_token]
            tokens.extend(word_tokens)
            main_label = processor.label_map.get(label, processor.label_map['X'])
            label_ids.extend([main_label] + [processor.label_map['X']]*(len(word_tokens)-1))
            predict_mask.extend([1] + [0]*(len(word_tokens)-1))
        
        tokens = tokens[:max_length]
        label_ids = label_ids[:max_length]
        predict_mask = predict_mask[:max_length]
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        padding = max_length - len(input_ids)
        
        features.append({
            'input_ids': input_ids + [tokenizer.pad_token_id]*padding,
            'attention_mask': [1]*len(input_ids) + [0]*padding,
            'labels': label_ids + [processor.label_map['[PAD]']]*padding,
            'predict_mask': predict_mask + [0]*padding
        })
    return features

class NerDataset(data.Dataset):
    def __init__(self, examples, tokenizer, processor, max_length):
        self.features = convert_examples_to_features(examples, tokenizer, processor, max_length)
    
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

def train_fold(fold, train_loader, val_loader, processor, args, tokenizer):
    try:
        model = LlamaForNER(
            Config.llama_model_name,
            len(processor.labels)
        )
        
        # Habilitar gradient checkpointing se solicitado
        if args.gradient_checkpointing:
            model.llama.gradient_checkpointing_enable()
            logging.info(f"✅ Gradient checkpointing habilitado para fold {fold}")
        
        # Otimizador apenas para parâmetros treináveis
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(train_loader) // 3,
            num_training_steps=len(train_loader) * args.epochs
        )
        
        best_f1 = -1
        patience_counter = 0
        best_model_path = os.path.join(args.output_dir, f"fold_LLaMA_{fold}_best.pt")
        
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
                
                # Log a cada batch devido ao tamanho menor
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
            
            if val_results['f1'] > best_f1:
                best_f1 = val_results['f1']
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"🎯 NOVO MELHOR F1: {best_f1:.4f} (modelo salvo)")
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
        
        return final_results
        
    except Exception as e:
        logging.error(f"❌ Erro no fold {fold}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        # Retornar resultados vazios em caso de erro
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'loss': 0.0, 'report': ''}

def main():
    args = parse_args()
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'resultados_llama.txt')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    try:
        logging.info("🚀 INICIANDO TREINAMENTO LLaMA PURO")
        logging.info(f"📁 Dados: {args.data}")
        logging.info(f"📂 Output: {args.output_dir}")
        logging.info(f"🎯 Folds: {args.k_folds}")
        logging.info(f"📦 Batch: {args.batch_size}")
        logging.info(f"📈 Épocas: {args.epochs}")
        logging.info(f"💪 Paciência: {args.patience}")
        logging.info(f"📚 Learning Rate: {args.learning_rate}")
        logging.info(f"📏 Max Length: {args.max_length}")
        logging.info(f"🔧 Gradient Checkpointing: {args.gradient_checkpointing}")
        logging.info(f"⚡ Device: {Config.device}")
        
        processor = DataProcessor()
        examples = processor.get_examples(args.data)
        
        try:
            tokenizer = LlamaTokenizer.from_pretrained(Config.llama_model_name)
            tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            logging.error(f"❌ Erro ao carregar tokenizer: {e}")
            logging.info("🔄 Usando tokenizer alternativo...")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(Config.llama_model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        logging.info(f"📊 Total de exemplos: {len(examples)}")
        logging.info(f"🏷️ Labels: {processor.labels}")
        
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
            
            train_dataset = NerDataset(train_examples, tokenizer, processor, args.max_length)
            val_dataset = NerDataset(val_examples, tokenizer, processor, args.max_length)
            
            train_loader = data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True
            )
            val_loader = data.DataLoader(
                val_dataset,
                batch_size=args.batch_size
            )
            
            results = train_fold(fold+1, train_loader, val_loader, processor, args, tokenizer)
            
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
        with open(os.path.join(args.output_dir, 'metricas_finais_llama.txt'), 'w') as f:
            f.write(f"F1 Médio: {np.mean(fold_results):.4f} (±{np.std(fold_results):.4f})\n")
            f.write(f"Valores por Fold: {[round(f, 4) for f in fold_results]}\n")
            f.write(f"Melhor Fold: {np.argmax(fold_results) + 1} (F1 = {max(fold_results):.4f})\n")
            f.write(f"Max Length: {args.max_length}\n")
    
    except Exception as e:
        logging.error(f"❌ Erro: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()