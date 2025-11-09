# Benchmarking Lightweight Transformers on FakeNewsNet

## Research Project Overview

This repository contains the implementation for the research paper **"Towards Deployable Disinformation Defense: Benchmarking Lightweight Transformers on FakeNewsNet"** by Humberto Goncalves, Kossi Sam Affambi, and Zakaria Alomari from the Department of Computer Science, New York Institute of Technology.

### Research Objective

The primary objective of this research is to design a **scalable and efficient framework for real-time disinformation detection** using lightweight transformer models. Unlike previous studies that relied on heavy transformer architectures (e.g., BERT, RoBERTa) which are computationally expensive and unsuitable for real-time deployment, this work focuses on:

1. **Performance Evaluation**: Benchmarking lightweight transformers (ALBERT, DistilBERT, TinyBERT) on fake news detection
2. **Operational Efficiency**: Measuring inference latency, memory usage, and model size for deployment feasibility
3. **Explainability**: Integrating XAI techniques (SHAP, LIME) for transparent and trustworthy predictions

### Problem Statement

While transformer-based models like BERT achieve state-of-the-art performance (up to 98% accuracy) on fake news classification, their high computational cost and latency make them impractical for real-time deployment on social media platforms. This creates a critical gap: platforms combating viral disinformation need immediate detection, but the most effective models are too computationally expensive to scale.

This research addresses this gap by evaluating lightweight alternatives that balance detection accuracy with computational efficiency.

---

## Dataset: FakeNewsNet

### Overview

The FakeNewsNet data repository is a comprehensive benchmark developed to support research on misinformation by integrating news content with social context and spatiotemporal engagement patterns.

### Dataset Composition

- **PolitiFact**: ~1,056 labeled political news articles

  - 420 fake articles
  - 528 real articles
- **GossipCop**: ~21,000 entertainment news articles

  - 5,323 fake articles
  - 16,817 real articles

### Merged Dataset

For this research, we combined both datasets into `merged_fakenewsnet.csv` containing:

- **clean_title**: Preprocessed news article titles
- **label**: Binary classification (0 = fake, 1 = real)

### Preprocessing Pipeline

The preprocessing steps (implemented in `FakeNewsNet_Datasets_Cleaning.ipynb`) include:

1. **Duplicate Removal**: Identified and removed duplicate entries based on identical titles
2. **Missing Value Handling**: Discarded records with missing titles or labels
3. **Text Normalization**: Converted all text to lowercase
4. **URL Removal**: Removed URLs as they are irrelevant to semantic understanding
5. **Mention Removal**: Stripped Twitter-specific user mentions (@username)
6. **Punctuation Removal**: Removed punctuation to reduce tokenization noise
7. **Emoji & HTML Removal**: Cleaned non-linguistic artifacts
8. **Language Filtering**: Filtered out non-English records
9. **Class Balancing**: Applied random undersampling to GossipCop to balance fake/real distribution
10. **Label Encoding**: Converted string labels ('fake', 'real') to numeric (0, 1)

---

## Model Training Details

### Lightweight Transformer Models

We benchmark three lightweight transformer architectures:

#### 1. **ALBERT (A Lite BERT)**

- **Model**: `albert-base-v2`
- **Architecture**: 12 layers, 128 hidden dimensions, 12 attention heads
- **Key Innovation**: Parameter sharing across layers reduces model size
- **Tokenizer**: SentencePiece-based tokenizer
- **Parameters**: ~12M (vs. BERT's 110M)

#### 2. **DistilBERT**

- **Model**: `distilbert-base-uncased`
- **Architecture**: 6 layers, 768 hidden dimensions, 12 attention heads
- **Key Innovation**: Knowledge distillation from BERT (40% smaller, 60% faster)
- **Tokenizer**: WordPiece tokenizer
- **Parameters**: ~66M

#### 3. **TinyBERT**

- **Model**: `huawei-noah/TinyBERT_General_4L_312D`
- **Architecture**: 4 layers, 312 hidden dimensions, 12 attention heads
- **Key Innovation**: Two-stage knowledge distillation (7.5x smaller, 9.4x faster than BERT)
- **Tokenizer**: BERT-compatible WordPiece tokenizer
- **Parameters**: ~14.5M

---

## Training Pipeline

### 1. Data Loading and Validation

```python
# Load preprocessed dataset
df = pd.read_csv("merged_fakenewsnet.csv")

# Verify label distribution
# Ensure labels are numeric (0 = fake, 1 = real)
```

### 2. Dataset Splitting (80/10/10)

We use **stratified splitting** to maintain class balance across all sets:

- **Training Set (80%)**: 7,553 samples for model training
- **Validation Set (10%)**: 944 samples for hyperparameter tuning and monitoring
- **Test Set (10%)**: 945 samples for final evaluation

```python
train_test_split(
    df['clean_title'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']  # Maintains class distribution
)
```

**Why stratified splitting?** Ensures that the proportion of fake/real news is consistent across train, validation, and test sets, preventing bias in evaluation.

### 3. Tokenization

Each model uses its pre-trained tokenizer to convert text into token IDs:

```python
tokenizer(
    texts,
    truncation=True,        # Truncate sequences longer than max_length
    padding='max_length',   # Pad shorter sequences to max_length
    max_length=128,         # Maximum sequence length
    return_tensors='pt'     # Return PyTorch tensors
)
```

**Key Parameters:**

- **max_length=128**: Balances context capture with computational efficiency
- **truncation=True**: Handles titles longer than 128 tokens
- **padding='max_length'**: Ensures uniform batch processing

**Output:** Three components per sample:

- `input_ids`: Token IDs representing the text
- `attention_mask`: Binary mask (1 for real tokens, 0 for padding)
- `token_type_ids`: Segment IDs (not used in single-sentence classification)

### 4. Custom Dataset and DataLoaders

```python
class FakeNewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
```

**DataLoader Configuration:**

- **Batch Size**: 16 samples per batch
- **Training**: `shuffle=True` for better generalization
- **Validation/Test**: `shuffle=False` for consistent evaluation

### 5. Model Initialization

```python
# Load pre-trained model with classification head
model = ModelForSequenceClassification.from_pretrained(
    'model-name',
    num_labels=2  # Binary classification (fake vs. real)
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

**Classification Head:** The pre-trained transformer is augmented with a linear layer for binary classification:

- Input: Hidden state of [CLS] token (contextual representation of entire sequence)
- Output: 2 logits (scores for fake and real classes)

### 6. Optimizer and Learning Rate Scheduler

```python
# AdamW optimizer (Adam with weight decay regularization)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Linear learning rate decay
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_epochs * len(train_loader)
)
```

#### What is a Learning Rate?

Think of training a model like walking down a hill to find the lowest point (best solution):

- **Learning Rate** = size of your steps
- **Big steps (high learning rate)**: You move fast but might overshoot the bottom
- **Small steps (low learning rate)**: You move slowly but more precisely

**Our learning rate**: `2e-5` = `0.00002` (very small steps because the model is already pre-trained)

#### Learning Rate Scheduler?

The scheduler automatically makes your steps smaller as you get closer to the goal:

```
Start of training:  Step size = 0.00002  (bigger steps to move quickly)
Middle of training: Step size = 0.00001  (medium steps)
End of training:    Step size = 0.000001 (tiny steps for fine-tuning)
```

**Why?** Early on, you want to learn fast. Later, you want to be careful not to mess up what you already learned.

**Visual Example:**

```
Learning Rate Over Time
│
│ 0.00002 ●────╲
│              ╲
│               ╲
│                ╲
│                 ╲
│                  ●
│ 0.00000          └────────────
  Epoch 1         Epoch 2    Epoch 3
```

#### Why AdamW Optimizer?

**AdamW** is like a smart assistant that:

- Adjusts step sizes for different parts of the model automatically
- Prevents the model from memorizing training data (overfitting)
- Is proven to work best for transformer models

### 7. Training Loop (3 Epochs)

#### What is an Epoch?

An **epoch** is one complete pass through all your training data.

**Example:**

- You have 7,553 news titles to learn from
- **1 epoch** = The model sees all 7,553 titles once
- **3 epochs** = The model sees all 7,553 titles three times

**Why multiple epochs?**

- **1 epoch**: Not enough, the model hasn't learned the patterns yet
- **3 epochs**: Just right, the model learns without memorizing
- **10+ epochs**: Too much, the model starts memorizing instead of learning (overfitting)

**Think of it like studying for an exam:**

- Reading the textbook once (1 epoch) = not enough
- Reading it 3 times (3 epochs) = good understanding
- Reading it 20 times (20 epochs) = you memorize but don't understand

#### Training Timeline

```
Epoch 1: Model sees all 7,553 titles → learns basic patterns
Epoch 2: Model sees all 7,553 titles again → refines understanding
Epoch 3: Model sees all 7,553 titles again → fine-tunes predictions
```

#### Epoch Structure

Each epoch consists of two phases:

**Training Phase:**

```python
model.train()  # Enable dropout and batch normalization
for batch in train_loader:
    # 1. Forward pass
    outputs = model(**batch)
    loss = outputs.loss
  
    # 2. Backward pass
    loss.backward()  # Compute gradients
  
    # 3. Optimizer step
    optimizer.step()        # Update weights
    lr_scheduler.step()     # Update learning rate
    optimizer.zero_grad()   # Reset gradients
```

**Validation Phase:**

```python
model.eval()  # Disable dropout, use batch norm running stats
with torch.no_grad():  # Disable gradient computation
    for batch in val_loader:
        outputs = model(**batch)
        val_loss += outputs.loss.item()
```

#### How Epochs and Learning Rate Work Together

```
Epoch 1, Batch 1:   Learning rate = 0.00002  → Model learns
Epoch 1, Batch 100: Learning rate = 0.000018 → Model learns
Epoch 1, Batch 473: Learning rate = 0.000013 → Model learns
────────────────────────────────────────────────────────────
Epoch 2, Batch 1:   Learning rate = 0.000013 → Model refines
Epoch 2, Batch 100: Learning rate = 0.000011 → Model refines
Epoch 2, Batch 473: Learning rate = 0.000007 → Model refines
────────────────────────────────────────────────────────────
Epoch 3, Batch 1:   Learning rate = 0.000007 → Model fine-tunes
Epoch 3, Batch 100: Learning rate = 0.000005 → Model fine-tunes
Epoch 3, Batch 473: Learning rate ≈ 0        → Model polishes
```

**Notice:** Steps get smaller and smaller = more careful adjustments over time

#### Loss Function

**Cross-Entropy Loss** (automatically computed by the model):

```
Loss = -[y * log(p) + (1-y) * log(1-p)]
```

where:

- `y` = true label (0 or 1)
- `p` = predicted probability for class 1

### 8. Model Saving

```python
# Save model weights and configuration
model.save_pretrained("model_fakenewsnet")

# Save tokenizer for inference
tokenizer.save_pretrained("model_fakenewsnet")
```

**Saved Files:**

- `config.json`: Model architecture configuration
- `pytorch_model.bin` or `model.safetensors`: Trained weights
- `vocab.txt`: Tokenizer vocabulary
- `tokenizer_config.json`: Tokenizer settings

---

## Evaluation Metrics

### Performance Metrics

#### 1. **Accuracy**

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Overall correctness of predictions.

#### 2. **Precision**

```
Precision = TP / (TP + FP)
```

Proportion of positive predictions that are correct. High precision = low false positive rate.

#### 3. **Recall (Sensitivity)**

```
Recall = TP / (TP + FN)
```

Proportion of actual positives correctly identified. High recall = low false negative rate.

#### 4. **F1 Score**

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

Harmonic mean of precision and recall. Balances both metrics.

#### 5. **AUC-ROC**

Area under the Receiver Operating Characteristic curve. Measures model's ability to distinguish between classes across all classification thresholds.

### Operational Efficiency Metrics

#### 1. **Inference Latency**

- Measures average time to process a batch of 16 samples
- Reported in milliseconds per batch and per sample
- Critical for real-time deployment

#### 2. **Memory Usage**

- **GPU Memory**: Allocated CUDA memory during inference
- **CPU Memory**: Process memory consumption
- Determines deployment feasibility on resource-constrained devices

#### 3. **Model Size**

- Total disk space occupied by saved model files
- Smaller models enable easier deployment and distribution
- Important for edge deployment scenarios

---

## Repository Structure

```
fakenewnet_bertmodels/
├── data/
│   ├── merged_fakenewsnet.csv          # Combined preprocessed dataset
│   └── merged_fakenewsnet_numeric.csv  # Numeric labels version
├── notebooks/
│   ├── FakeNewsNet_Datasets_Cleaning.ipynb  # Preprocessing pipeline
│   ├── ALBERT.ipynb                          # ALBERT training & evaluation
│   ├── DistilBERT.ipynb                      # DistilBERT training & evaluation
│   └── TinyBERT.ipynb                        # TinyBERT training & evaluation
├── albert_fakenewsnet/              # Saved ALBERT model (after training)
├── distilbert_fakenewsnet/          # Saved DistilBERT model (after training)
├── tinybert_fakenewsnet/            # Saved TinyBERT model (after training)
└── README.md                        # This file
```

---

## Usage Instructions

### Prerequisites

```bash
pip install transformers==4.36.0 torch==2.9.0 scikit-learn==1.7.2 pandas tqdm psutil sentencepiece
```

### Running the Notebooks

1. **Data Preprocessing** (if not already done):

   ```
   Run: FakeNewsNet_Datasets_Cleaning.ipynb
   Output: merged_fakenewsnet.csv
   ```
2. **Model Training**:

   - For ALBERT: Run `ALBERT.ipynb`
   - For DistilBERT: Run `DistilBERT.ipynb`
   - For TinyBERT: Run `TinyBERT.ipynb`
3. **Expected Outputs**:

   - Trained model saved in respective directory
   - Performance metrics (Accuracy, Precision, Recall, F1, AUC)
   - Efficiency metrics (Latency, Memory, Model Size)

### Training Time Estimates

- **CPU**: 2-4 hours per model (3 epochs)
- **GPU (CUDA)**: 15-30 minutes per model (3 epochs)

---

## Research Contributions

This work makes the following contributions to the field of disinformation detection:

1. **Comprehensive Benchmarking**: First systematic comparison of ALBERT, DistilBERT, and TinyBERT on FakeNewsNet dataset
2. **Efficiency-Focused Evaluation**: Unlike prior work that focuses solely on accuracy, we measure operational efficiency metrics critical for real-time deployment
3. **Deployment Readiness**: Provides actionable insights on model selection based on accuracy-efficiency trade-offs
4. **Reproducible Framework**: Standardized training pipeline ensures fair comparison and reproducibility

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{goncalves2025lightweight,
  title={Towards Deployable Disinformation Defense: Benchmarking Lightweight Transformers on FakeNewsNet},
  author={Goncalves, Humberto and Affambi, Kossi Sam and Alomari, Zakaria},
  journal={[Conference/Journal Name]},
  year={2025},
  institution={New York Institute of Technology}
}
```

---

## Authors

- **Humberto Goncalves** - hdasilva@nyit.edu
- **Kossi Sam Affambi** - kaffambi@nyit.edu
- **Zakaria Alomari** (Corresponding Author) - zalomari@nyit.edu

Department of Computer Science
New York Institute of Technology
Vancouver, British Columbia, Canada

---

## License

This project is for academic research purposes. Please contact the authors for usage permissions.

---

## Acknowledgments

- FakeNewsNet dataset creators for providing comprehensive fake news benchmark
- Hugging Face Transformers library for pre-trained models
- Research community for prior work on lightweight transformers and fake news detection
