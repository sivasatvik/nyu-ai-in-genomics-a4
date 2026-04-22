# Assignment 4: Advanced Genomic Foundation Models (Weeks 8–9)

**Due Date**: Apr 28
**Total Points**: 100 (You will be graded to 115 for this assignment and the extra will transfer to any other assignments)

This assignment builds on Assignment 3 by exploring more advanced techniques for genomic foundation models. You will:

2. Apply **parameter-efficient fine-tuning (LoRA)** to adapt your FMs to TF classification
3. Use **interpretability** methodologies to try and derive insight from these models
4. Investigate **cross-species transfer learning** and few-shot adaptation
5. Perform **ablation studies** to understand which CDS regions drive TF predictions

**Prerequisites from Assignment 3:**
- `data/tf_multispecies_sequences.parquet` (TF dataset with CDS and protein sequences) - No need to remake teh dataset 
- `data/tf_split_indices.json` (train/val/test splits) - Minimize noise from different splits
- `data/dna_embeddings.npy` (CDS embeddings from Nucleotide Transformer or Evo2)
- `data/protein_embeddings.npy` (protein embeddings from ESM2)

If you do not have these files (you might have saved them in a different format, that is okay), feel free to use your own and re-generate them from Assignment 3 code.

---

## Required Libraries

```
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
```

Optional utilities for attribution/motif analysis (you can use different utilities):

```
import captum.attr as captum_attr
from logomaker import Logo
```

Set random seed for reproducibility:

```
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

---

# Fusion & LoRA Fine-Tuning (45 pts)

## 1.1 Setup & Data Loading from Assignment 3

Before starting, ensure you have the following artifacts from Assignment 3 in your `data/` directory. This section will load the preprocessed:

1. dataset,
2. embeddings,
3. splits.

### Load TF dataset and embeddings (5 pts)

Load the data you generated in the previous assignment and generate the same splits.

```python

```

---

## 1.2 LoRA Adapter Fine-Tuning (NT, Evo-2, ESM2)

In this section, you will fine-tune **LoRA adapters** (plus a small classifier head) and compare against the frozen-backbone baselines on assignment 3. Keep this experiment small enough to be runnable (limit sequence length, batch size, and epochs).

Ensure that **you are using identical splits** as last assignment to minimize any noise and have a good apples to apples comparison. For efficiency, evaluate on **Split 1** (human train → mouse/fly zero-shot transfer) only; the ESM2 reference implementation below follows this pattern.

A quick suggested epoch for LoRA finetuning would be 5-10 epochs, LoRA rank 8–16. Feel free to fix the code below.

```python
from dataclasses import dataclass
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data")

# Models
FAST_RUN = True
RUN_NT_LORA = True
RUN_ESM2_LORA = True
RUN_EVO2_LORA = False 

# LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Hyperparameters
EPOCHS = 5 if FAST_RUN else 10
BATCH_SIZE = 4 if FAST_RUN else 8

# Sequence length limits
DNA_LORA_MAX_LEN = 512
PROTEIN_LORA_MAX_LEN = 512

# Models
NT_MODEL_ID = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
ESM2_MODEL_ID = "facebook/esm2_t12_35M_UR50D"
EVO2_MODEL_NAME = "evo2_1b_base"
EVO2_LAYER_FOR_POOLING = "blocks.5.mlp.l3" # How do you get the sequence level embedding from the model?

# Just a helper for the metrics, feel free to not use
def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan"),
        "prc_auc": average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan"),
    }

### 1.2.1 NT LoRA (CDS DNA) (10 pts)

### 1.2.2 Evo-2 LoRA (CDS; environment-dependent - optional)

### 1.2.3 ESM2 LoRA (protein) (10 pts)


## 1.3 Frozen vs LoRA Comparison (5 pts)

Compare the frozen-backbone baseline results from Assignment 3 against your LoRA experiments from Section 1.3.

Highlight where LoRA helps most and whether it changes precision-recall tradeoffs. 

Calculate the same metrics and add them to your big table.

## 1.4 LoRA Hyperparameter Ablations

Test different LoRA hyperparameters to understand the trade-off between parameter efficiency and performance. Run on one model (your choice) for efficiency.

### 1.4.1 LoRA Rank and Alpha Grid Search (10 pts)

```python
# Ablation: test different LoRA ranks and alpha values - Simple grid search idea below
LORA_RANKS = [4, 8, 16, 32]
LORA_ALPHAS = [8, 16, 32]
```

Train at least 2 ranks and 2 alpha values. Report runtime and memory usage.

### 1.4.2 Visualize Parameter Efficiency vs Performance (5 pts)

Plot the performance metrics per LoRA configuration. Is there some sweet spot between the number of parameters, runtime, and performance?

# Interpretability & Applications (55 pts)

## 2.1 Attribution Analysis: Understanding Model Decisions

**Learning Objective:** What drives your models' predictions? Do they learn biologically "meaningful" patterns? Here you will use a few different methodologies to try and identify what the model is learning

**Why Attribution Matters:**
- Opening the black box - What do foundation models learn?
- If they actually learn useful biology, we should be able to trace residue level (nucleotide or aminoacid) to results
- Comparing multiple methods (gradients, attention, SHAP) validates findings
- Biological validation: do high-attribution regions match known TF motifs (e.g. HLH, Zinc finger, etc.)?

### 2.1.1 Integrated Gradients for Attribution (15 pts)

**What is Integrated Gradients?**

Once again we use our favourite library, captum, to help us with this task. You are free to use differnet libraries. We integrate gradients along a path from a random sequence baseline to an input sequence to identify which positions in the sequence are most important for the model's prediction.

1. Select 5-10 high-confidence TF predictions from your validation set
2. Implement Integrated Gradients using `captum.attr.IntegratedGradients` or another library of your choice
3. For CDS/AA sequences:
   - Use your trained **ESM2/NT/Evo2 model** (We suggest ESM2)
   - Set baseline as a random sequence
   - Compute token-level attributions (one score per k-mer or nucleotide position)
4. Visualize attributions as:
   - **Heatmap**: Plot attribution scores along the sequence 
   - **Sequence logo**: Use `logomaker` to show enriched motifs in high-attribution regions
5. Extract top-5 high-attribution windows (e.g., choose your BP window, 30 is a good baseline)
6. Compare against known TF motifs (optional: use the JASPAR database or literature)

- Do high-attribution regions correspond to any known domains?
- Are there species-specific attribution patterns?

---

### 2.1.2 Attention Weights Visualization (15 pts)

**What are Attention Weights?**

Transformer models (NT, Evo2, ESM2) use multi-head self-attention to weigh the importance of different positions. Attention weights show which tokens the model "attends to" when making predictions. 

Unlike gradients, attention is computed during the forward pass without needing backpropagation. There is a nuanced discussion between attention and gradients. Sometimes they coincide, but often they do not.

1. For the same 5-10 sequences from 2.1.1, extract attention weights from the same model
2. Most transformer models expose attention via `model(..., output_attentions=True)`.
3. Average attention across either:
   - Top N attention heads with highest variance
   - The last layer before you extract your embeddings (we suggest this)
4. Visualize as a heatmap: rows = query positions, columns = key positions
5. Compare attention patterns to IG attributions:
   - Do they highlight the same regions?
   - Where do they disagree, and why do you think that might occur?



- Are attention patterns sparse (focused on specific motifs) or diffuse? How does that compare with IG?


### 2.1.3 Sparse Autoencoder (SAE) Feature Extraction 30 points)

**What are Sparse Autoencoders?**

The recent breakthrough (According to some) in mechanistic interpretability of models are Sparse Autoencoders (SAEs). They aim to decompose a model's latent dimension to interpretable sparse features. 

Each SAE feature ideally corresponds to a meaningful biological concept (e.g., a specific motif, structural domain, or sequence pattern).

SAEs are a [very](https://arxiv.org/abs/2501.16615) [controversial](https://arxiv.org/abs/2501.16615) area of research and have been successfully applied to biological Foundation Models before (e.g. [ESM2](https://www.nature.com/articles/s41592-025-02836-7). SAEs purport to be able to tie foundation model latents to known and novel concepts.

SAEs project dense model activation into a **much higher** dimension "dictionary" using a linear encoder and a ReLU. You usually implement this with an nn.Module that calculates a reconstruction loss (similar to regular AE, but the bottleneck is larger rather than smaller). You pair this with an L1 penalty to map latents to only one dimension in teh dictionary to ensure "monosemanticity", where each latent corresponds to one concept. Then you map those back to the original dimension.

1. Train a sparse autoencoder on intermediate activations from your **ESM2 LoRA model** (e.g., output of layer 6) (15 pts)
2. Extract activations for 10-20 sequences (mix of TF and non-TF)
3. Train wide SAE with L1 sparsity penalty to encourage sparse feature usage
4. Analyze learned features: (15 pts)
   - Which features activate most strongly for TFs vs non-TFs? Are they meaningful?
   - Do features correspond to known protein domains (e.g., DNA-binding domains, zinc fingers)?
   - Compare SAE features to attribution results from 2.1.1-2.1.3