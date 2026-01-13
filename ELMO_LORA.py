import sys
import os
import argparse
import logging
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torch.optim import AdamW
from allennlp.modules.elmo import Elmo, batch_to_ids
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from peft import LoraConfig, get_peft_model, TaskType
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Arquivo único com dados no formato IOB')
    parser.add_argument('--output_dir', required=True, help='Diretório para salvar resultados')
    parser.add_argument('--k_folds', type=int, default=5, help='Número de folds (padrão: 5)')
    parser.add_argument('--batch_size', type=int, default=16, help='Tamanho do batch (padrão: 16)')
    parser.add_argument('--epochs', type=int, default=15, help='Número de épocas (padrão: 15)')
    parser.add_argument('--patience', type=int, default=5, help='Paciência para early stopping')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Taxa de aprendizado')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA r parameter')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha parameter')
    return parser.parse_args()

class Config:
    options_file = "elmo_pt_options.json"  # Ajuste o caminho
    weight_file = "elmo_pt_weights.hdf5"   # Ajuste o caminho
    max_seq_length = 180
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42

class ElmoLoRA(nn.Module):
    def __init__(self, num_labels, lora_r=8, lora_alpha=16, dropout=0.1):
        super().__init__()
        self.elmo = Elmo(Config.options_file, Config.weight_file, num_output_representations=1, dropout=0)
        self.embedding_dim = self.elmo.get_output_dim()
        
        # Aplicar LoRA nas camadas lineares do classificador
        self.classifier = nn.Linear(self.embedding_dim, num_labels)
        
        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["weight"]  # Aplicar LoRA nos pesos do classificador
        )
        self.classifier = get_peft_model(self.classifier, lora_config)
        
        self.dropout = nn.Dropout(dropout)
        
        # Log LoRA parameters
        trainable_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.classifier.parameters())
        logging.info(f"🔧 LoRA - Parâmetros treináveis: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
        
    def forward(self, char_ids, attention_mask, labels=None):
        elmo_output = self.elmo(char_ids)
        embeddings = elmo_output['elmo_representations'][0]
        logits = self.classifier(self.dropout(embeddings))
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.size(-1))[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss
        
        return logits

# Classes auxiliares (InputExample, DataProcessor, NerDataset, evaluate) são as mesmas do ELMo básico

def train_fold(fold, train_loader, val_loader, processor, args):
    model = ElmoLoRA(
        num_labels=len(processor.labels),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    ).to(Config.device)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    best_f1 = -1
    patience_counter = 0
    best_model_path = os.path.join(args.output_dir, f"fold_ELMo_LoRA_{fold}_best.pt")
    
    logging.info(f"🔧 Configuração ELMo+LoRA:")
    logging.info(f"   - LoRA r: {args.lora_r}")
    logging.info(f"   - LoRA alpha: {args.lora_alpha}")
    
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            words = batch['words']
            attention_mask = batch['attention_mask'].to(Config.device)
            labels = batch['labels'].to(Config.device)
            
            char_ids = batch_to_ids(words).to(Config.device)
            loss = model(char_ids, attention_mask, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % max(1, len(train_loader) // 10) == 0:
                current_loss = total_loss / (batch_idx + 1)
                progress = (batch_idx + 1) / len(train_loader) * 100
                logging.info(f"Fold {fold}, Época {epoch+1}, Progresso: {progress:.0f}% | Loss = {current_loss:.4f}")
        
        avg_train_loss = total_loss / len(train_loader)
        val_results = evaluate(model, val_loader, processor)
        epoch_time = time.time() - start_time
        
        logging.info(f"\n=== Fold {fold}, Época {epoch+1} ===")
        logging.info(f"⏱️  Tempo: {epoch_time:.2f}s")
        logging.info(f"📉 Loss Treino: {avg_train_loss:.4f}")
        logging.info(f"📊 Loss Validação: {val_results['loss']:.4f}")
        logging.info(f"🎯 F1 Validação: {val_results['f1']:.4f}")
        logging.info(f"🎯 Precision Validação: {val_results['precision']:.4f}")
        logging.info(f"🎯 Recall Validação: {val_results['recall']:.4f}")
        
        if val_results['f1'] > best_f1:
            best_f1 = val_results['f1']
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"✅ NOVO MELHOR F1: {best_f1:.4f} (modelo salvo)")
        else:
            patience_counter += 1
            logging.info(f"⏳ F1 não melhorou. Paciência: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                logging.info(f"🛑 Early stopping na época {epoch+1}")
                break
        
        logging.info("-" * 80)
    
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logging.info(f"✅ Fold {fold}: Melhor modelo carregado com F1 = {best_f1:.4f}")
    
    final_results = evaluate(model, val_loader, processor)
    
    logging.info(f"\n📊 RESUMO FOLD {fold}")
    logging.info(f"🎯 Melhor F1: {best_f1:.4f}")
    logging.info(f"📈 F1 Final: {final_results['f1']:.4f}")
    
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
            logging.FileHandler(os.path.join(args.output_dir, 'resultados_elmo_lora.txt')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    try:
        logging.info("🚀 INICIANDO TREINAMENTO ELMo+LoRA")
        logging.info(f"📁 Dados: {args.data}")
        logging.info(f"📂 Output: {args.output_dir}")
        logging.info(f"🎯 Folds: {args.k_folds}")
        logging.info(f"📦 Batch: {args.batch_size}")
        logging.info(f"📈 Épocas: {args.epochs}")
        logging.info(f"💪 Paciência: {args.patience}")
        logging.info(f"📚 Learning Rate: {args.learning_rate}")
        logging.info(f"🔧 LoRA r: {args.lora_r}")
        logging.info(f"🔧 LoRA alpha: {args.lora_alpha}")
        
        processor = DataProcessor()
        examples = processor.get_examples(args.data)
        
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
            
            train_dataset = NerDataset(train_examples, processor)
            val_dataset = NerDataset(val_examples, processor)
            
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
            logging.info(f"🎯 F1-Score: {results['f1']:.4f}")
            
            fold_results.append(results['f1'])
        
        logging.info(f"\n{'='*60}")
        logging.info("🎯 RESULTADO FINAL")
        logging.info(f"{'='*60}")
        logging.info(f"📊 F1 Médio: {np.mean(fold_results):.4f} (±{np.std(fold_results):.4f})")
        logging.info(f"📈 Valores por Fold: {[round(f, 4) for f in fold_results]}")
        
        with open(os.path.join(args.output_dir, 'metricas_finais_elmo_lora.txt'), 'w') as f:
            f.write(f"F1 Médio: {np.mean(fold_results):.4f} (±{np.std(fold_results):.4f})\n")
            f.write(f"Valores por Fold: {[round(f, 4) for f in fold_results]}\n")
            f.write(f"LoRA r: {args.lora_r}\n")
            f.write(f"LoRA alpha: {args.lora_alpha}\n")
    
    except Exception as e:
        logging.error(f"❌ Erro: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()