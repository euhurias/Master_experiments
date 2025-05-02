# Code for Legal Entity Recognition task

This directory has code to train and evaluate classical-based , transformer-based and hybrid approaches on LER task using the three datasets, LENER, UlyssesNER and CDJUR. This package implements 11 approaches divided in three approaches:

**Classical-based approaches**:

- CRF
- BiLSTM
-BiLSTM-CRF

**Transformer-based approaches**:

- BERT
- XLNet

**Hybrid approaches**:

- BERT-CRF
- BERT-BiLSTM
- BERT-BiLSTM-CRF

- XLNet-CRF
- XLNet-BiLSTM
- XLNet-BiLSTM-CRF

## Environment Setup

The code uses a Python 3.8+ environment and a GPU is desirable. The following steps used  a Python virtual environment.

## 📌 Requirements

  ```bash
  pip install -r requirements.txt
  ```
- Main libraries:
  - `transformers`
  - `torch`
  - `seqeval`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `sentencepiece`
  - `pytorch-crf`
  - `matplotlib`


## 📁 Expected data structure

The data file must be in **CoNLL** format, with the following structure:


```
Palavra1 O  
Palavra2 B-ENTIDADE  
Palavra3 I-ENTIDADE  

# Nova sentença
Palavra4 O  
Palavra5 B-ENTIDADE  
```

## ⚙️ Script Execution

To execute the training and evaluation script, use the following command:

```bash
python script.py --data data.txt --output_dir output
```

### 🔹 Executing a specific BERT or XLNet model with CUDA debugging control

```bash
CUDA_LAUNCH_BLOCKING=1 python BERT.py --data data.txt --output_dir output
```

> ⚠️ Using `CUDA_LAUNCH_BLOCKING=1` is recommended only for debugging GPU-related errors.

## 📦 Expected Results

At the end of the execution, the script will generate:

- Evaluation metrics (F1, precision, recall) for each fold.
- Average and standard deviation of the final metrics.
- Files with checkpoints of the trained models (if implemented).
- Detailed logs of the training process.
