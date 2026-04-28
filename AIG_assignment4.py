# %% [markdown]
# # Assignment 4: Advanced Genomic Foundation Models
# 
# Siva Satvik Mandapati - sm12779

# %% [markdown]
# ## Required Libraries

# %%
import numpy as np
import pandas as pd
import torch
import sys
import requests
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
# Non-interactive plotting for batch runs / script export
plt.ioff()

# Force line-buffered stdout so print statements appear immediately in sbatch /
# Singularity logs rather than being flushed only at script exit.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# %%
# Optional utilities for attribution/motif analysis (you can use different utilities)
import captum.attr as captum_attr
from logomaker import Logo

# %%
# Set random seed for reproducibility
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# %% [markdown]
# ---
# 
# # Fusion & LoRA Fine-Tuning
# 
# ## 1.1 Setup & Data Loading from Assignment 3
# 
# Before starting, ensure you have the following artifacts from Assignment 3 in your `data/` directory. This section will load the preprocessed:
# 
# 1. dataset,
# 2. embeddings,
# 3. splits.
# 
# ### Load TF dataset and embeddings
# 
# Load the data you generated in the previous assignment and generate the same splits.

# %%
import json

DATA_DIR = Path("data")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load the TF dataset
df = pd.read_parquet(DATA_DIR / "tf_multispecies_sequences_large.parquet")
print(f"Dataset shape: {df.shape}")
print(df.head())
print(df.columns.tolist())

# %% [markdown]
# Dataset shape: (2591, 8)
# 
# |id | species | symbol | ensembl_gene_id |  ...                                        | protein_seq | chrom | strand|
# |--|--------|--------|----------------|-----|---------------------------------------------------|-------|-----|
# | 0 | human | TEAD3 | ENSG00000007866 | ... | MASNSWNASSSPGEAREDGPEGLDKGLDNDAEGVWSPDIEQSFQEA...  |   6   | -1 |
# | 1 | human | ZNF469 | ENSG00000225614 | ... | MPGERPRGAPPPTMTGDLQPRQVASSPGHPSQPPLEDNTPATRTTK...  |  16  |  1 |
# | 2 | human  |  SIX4 | ENSG00000100625 | ... | MIASAADIKQENGMESASEGQEAHREVAGGAAVGLSPPAPAPFPLE...  |  14 |  -1 |
# | 3  | human  |  RFX5 | ENSG00000143390 | ... | MAEDEPDAKSPKTGGRAPPGGAEAGEPTTLLQRLRGTISKAVQNKV...  |   1  | -1 |
# | 4 |  human  | SMAD5 | ENSG00000113658 | ... | MTSMASLFSFTSPAVKRLLGWKQGDEEEKWAEKAVDALVKKLKKKK...  |  5  |  1 |
# 
# [5 rows x 8 columns]
# 
# ['species', 'symbol', 'ensembl_gene_id', 'is_tf', 'cds_seq', 'protein_seq', 'chrom', 'strand']

# %%
# Load split indices from Assignment 3
with open(DATA_DIR / "tf_split_indices.json", "r") as f:
    split_indices = json.load(f)

print("Available splits:", list(split_indices.keys()))

# %% [markdown]
# Available splits: ['split1_train', 'split1_val', 'split1_test', 'split2_train', 'split2_val', 'split2_test', 'split3_train', 'split3_val', 'split3_test']

# %%
# Load precomputed embeddings
dna_embeddings = np.load(DATA_DIR / "dna_embeddings.npy")
protein_embeddings = np.load(DATA_DIR / "protein_embeddings.npy")

print(f"DNA embeddings shape:     {dna_embeddings.shape}")
print(f"Protein embeddings shape: {protein_embeddings.shape}")

# %% [markdown]
# DNA embeddings shape:     (2591, 1024)
# 
# Protein embeddings shape: (2591, 1280)

# %%
# Extract Split 1 indices (human train → mouse/fly zero-shot transfer)

train_idx = split_indices["split1_train"]
val_idx   = split_indices["split1_val"]
test_idx  = split_indices["split1_test"]

print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

# %% [markdown]
# Train: 700 | Val: 150 | Test: 1591

# %%
# Build is_tf arrays
labels = df["is_tf"].values

y_train = labels[train_idx]
y_val   = labels[val_idx]
y_test  = labels[test_idx]

print(f"Train class distribution: {np.bincount(y_train)}")
print(f"Val class distribution:   {np.bincount(y_val)}")
print(f"Test class distribution:  {np.bincount(y_test)}")

# %% [markdown]
# Train class distribution: [350 350]
# 
# Val class distribution:   [75 75]
# 
# Test class distribution:  [800 791]

# %% [markdown]
# ---
# 
# ## 1.2 LoRA Adapter Fine-Tuning (NT, Evo-2, ESM2)

# %%
# Download model snapshot from Hugging Face Hub (optional - removes if models already available)
from huggingface_hub import snapshot_download
local_dir = "./models/nucleotide-transformer-500m-human-ref"
if not Path(local_dir).exists():
    snapshot_download(repo_id="InstaDeepAI/nucleotide-transformer-500m-human-ref", local_dir=local_dir)
local_dir = "./models/esm2_t12_35M_UR50D"
if not Path(local_dir).exists():
    snapshot_download(repo_id="facebook/esm2_t12_35M_UR50D", local_dir=local_dir)

# %%
from dataclasses import dataclass
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = Path("data")

# Models to run
# NOTE: We must finetune at least one nucleotide model (NT) and one aminoacid model (ESM2/Evo2)
FAST_RUN      = False
RUN_NT_LORA   = True
RUN_ESM2_LORA = True
RUN_EVO2_LORA = False  # environment-dependent; set True if Evo2 is available

# LoRA hyperparameters
LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05

# Training hyperparameters
EPOCHS     = 5 if FAST_RUN else 10
BATCH_SIZE = 4 if FAST_RUN else 8

# Sequence length limits
DNA_LORA_MAX_LEN     = 512
PROTEIN_LORA_MAX_LEN = 512

# Model identifiers
NT_MODEL_NAME = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
NT_MODEL_LOCAL = Path("models/nucleotide-transformer-500m-human-ref")
NT_MODEL_ID    = str(NT_MODEL_LOCAL) if NT_MODEL_LOCAL.exists() else NT_MODEL_NAME
ESM_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
ESM_MODEL_LOCAL = Path("models/esm2_t12_35M_UR50D")
ESM2_MODEL_ID  = str(ESM_MODEL_LOCAL) if ESM_MODEL_LOCAL.exists() else ESM_MODEL_NAME
EVO2_MODEL_NAME     = "evo2_1b_base"
EVO2_LAYER_FOR_POOLING = "blocks.5.mlp.l3"  # layer used for sequence-level pooling


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute standard binary classification metrics."""
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan"),
        "prc_auc":   average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan"),
    }

def infer_lora_target_modules(model: nn.Module) -> list[str]:
    """Infer common attention projection module names for LoRA injection."""
    preferred_names = ("query", "key", "value", "q_proj", "k_proj", "v_proj")
    found = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf_name = name.split(".")[-1]
            if leaf_name in preferred_names:
                found.append(leaf_name)

    # Preserve order but remove duplicates.
    deduped = list(dict.fromkeys(found))
    if not deduped:
        raise ValueError(
            "Could not infer LoRA target modules from the model. "
            "Inspect model.named_modules() and set target_modules explicitly."
        )
    return deduped

def load_esm2_backbone(model_id: str, enable_attentions: bool = False) -> nn.Module:
    """Load ESM2 with an attention implementation that can expose weights."""
    load_kwargs = {}
    if enable_attentions:
        # SDPA/flash attention backends may skip returning attention tensors.
        load_kwargs["attn_implementation"] = "eager"

    try:
        model = AutoModel.from_pretrained(model_id, **load_kwargs)
    except TypeError:
        # Older transformers versions may not accept attn_implementation.
        model = AutoModel.from_pretrained(model_id)

    for cfg_owner in (model, getattr(model, "base_model", None)):
        cfg = getattr(cfg_owner, "config", None)
        if cfg is None:
            continue
        cfg.output_attentions = enable_attentions
        if enable_attentions and hasattr(cfg, "attn_implementation"):
            cfg.attn_implementation = "eager"

    return model

# %% [markdown]
# ### 1.2.1 NT LoRA (CDS DNA)

# %%
if RUN_NT_LORA:
    print("=== NT LoRA Fine-Tuning ===")

    # ── Tokenize sequences ──────────────────────────────────────────────────────
    nt_tokenizer = AutoTokenizer.from_pretrained(NT_MODEL_ID)

    cds_sequences = df["cds_seq"].values  # adjust column name if needed

    def tokenize_dna(seqs, tokenizer, max_len):
        return tokenizer(
            list(seqs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )

    train_enc = tokenize_dna(cds_sequences[train_idx], nt_tokenizer, DNA_LORA_MAX_LEN)
    val_enc   = tokenize_dna(cds_sequences[val_idx],   nt_tokenizer, DNA_LORA_MAX_LEN)

    train_labels_t = torch.tensor(y_train, dtype=torch.float)
    val_labels_t   = torch.tensor(y_val,   dtype=torch.float)

    train_ds_nt = TensorDataset(
        train_enc["input_ids"], train_enc["attention_mask"], train_labels_t
    )
    val_ds_nt = TensorDataset(
        val_enc["input_ids"], val_enc["attention_mask"], val_labels_t
    )

    train_loader_nt = DataLoader(train_ds_nt, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_nt   = DataLoader(val_ds_nt,   batch_size=BATCH_SIZE)

    # ── Build LoRA model ────────────────────────────────────────────────────────
    nt_base = AutoModel.from_pretrained(NT_MODEL_ID)
    nt_target_modules = infer_lora_target_modules(nt_base)
    print(f"NT LoRA target modules: {nt_target_modules}")

    lora_cfg_nt = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=nt_target_modules,
    )
    nt_lora = get_peft_model(nt_base, lora_cfg_nt)
    nt_lora.print_trainable_parameters()

    # Simple classifier head on top of [CLS] embedding
    hidden_size = nt_base.config.hidden_size

    class LoRAClassifier(nn.Module):
        def __init__(self, backbone, hidden_size):
            super().__init__()
            self.backbone = backbone
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )

        def forward(self, input_ids, attention_mask):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            return self.classifier(cls_emb).squeeze(-1)

    nt_model = LoRAClassifier(nt_lora, hidden_size).to(device)

    # ── Training loop ───────────────────────────────────────────────────────────
    optimizer_nt = AdamW(nt_model.parameters(), lr=2e-4)
    criterion    = nn.BCEWithLogitsLoss()

    nt_train_losses, nt_val_losses = [], []

    for epoch in range(EPOCHS):
        nt_model.train()
        epoch_loss = 0.0
        for input_ids, attention_mask, batch_labels in train_loader_nt:
            input_ids, attention_mask, batch_labels = (
                input_ids.to(device), attention_mask.to(device), batch_labels.to(device)
            )
            optimizer_nt.zero_grad()
            logits = nt_model(input_ids, attention_mask)
            loss   = criterion(logits, batch_labels)
            loss.backward()
            optimizer_nt.step()
            epoch_loss += loss.item()

        nt_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, attention_mask, batch_labels in val_loader_nt:
                input_ids, attention_mask, batch_labels = (
                    input_ids.to(device), attention_mask.to(device), batch_labels.to(device)
                )
                logits    = nt_model(input_ids, attention_mask)
                val_loss += criterion(logits, batch_labels).item()

        nt_train_losses.append(epoch_loss / len(train_loader_nt))
        nt_val_losses.append(val_loss / len(val_loader_nt))
        print(f"  Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {nt_train_losses[-1]:.4f} | Val Loss: {nt_val_losses[-1]:.4f}")

    # ── Evaluate ────────────────────────────────────────────────────────────────
    nt_model.eval()
    all_proba = []
    with torch.no_grad():
        for input_ids, attention_mask, _ in val_loader_nt:
            logits = nt_model(input_ids.to(device), attention_mask.to(device))
            all_proba.extend(torch.sigmoid(logits).cpu().numpy())

    nt_lora_metrics = compute_metrics(y_val, np.array(all_proba))
    print("\nNT LoRA Validation Metrics:")
    for k, v in nt_lora_metrics.items():
        print(f"  {k}: {v:.4f}")

# %% [markdown]
# === NT LoRA Fine-Tuning ===
# 
# NT LoRA target modules: ['query', 'key', 'value']
# 
# trainable params: 1,474,560 || all params: 481,912,801 || trainable%: 0.3060
# 
#   Epoch 1/10 | Train Loss: 0.6775 | Val Loss: 0.6611
# 
#   Epoch 2/10 | Train Loss: 0.5696 | Val Loss: 0.5706
# 
#   Epoch 3/10 | Train Loss: 0.4163 | Val Loss: 0.6002
# 
#   Epoch 4/10 | Train Loss: 0.2905 | Val Loss: 0.7050
# 
#   Epoch 5/10 | Train Loss: 0.1851 | Val Loss: 0.7646
# 
#   Epoch 6/10 | Train Loss: 0.1322 | Val Loss: 1.3166
# 
#   Epoch 7/10 | Train Loss: 0.1424 | Val Loss: 0.7929
# 
#   Epoch 8/10 | Train Loss: 0.0968 | Val Loss: 1.0348
# 
#   Epoch 9/10 | Train Loss: 0.0789 | Val Loss: 1.0746
# 
#   Epoch 10/10 | Train Loss: 0.0748 | Val Loss: 1.1656
# 
# 
# NT LoRA Validation Metrics: accuracy: 0.7600, precision: 0.8197, recall: 0.6667, f1: 0.7353, roc_auc: 0.8148, prc_auc: 0.8195

# %% [markdown]
# ### 1.2.3 ESM2 LoRA (protein)

# %%
if RUN_ESM2_LORA:
    print("=== ESM2 LoRA Fine-Tuning ===")

    # ── Tokenize protein sequences ───────────────────────────────────────────────
    esm2_tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_ID)

    protein_sequences = df["protein_seq"].values  # adjust column name if needed

    def tokenize_protein(seqs, tokenizer, max_len):
        return tokenizer(
            list(seqs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )

    train_enc_esm = tokenize_protein(protein_sequences[train_idx], esm2_tokenizer, PROTEIN_LORA_MAX_LEN)
    val_enc_esm   = tokenize_protein(protein_sequences[val_idx],   esm2_tokenizer, PROTEIN_LORA_MAX_LEN)

    train_ds_esm = TensorDataset(
        train_enc_esm["input_ids"], train_enc_esm["attention_mask"], train_labels_t
    )
    val_ds_esm = TensorDataset(
        val_enc_esm["input_ids"], val_enc_esm["attention_mask"], val_labels_t
    )

    train_loader_esm = DataLoader(train_ds_esm, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_esm   = DataLoader(val_ds_esm,   batch_size=BATCH_SIZE)

    # ── Build LoRA model ────────────────────────────────────────────────────────
    esm2_base = load_esm2_backbone(ESM2_MODEL_ID, enable_attentions=True)
    esm2_target_modules = infer_lora_target_modules(esm2_base)
    print(f"ESM2 LoRA target modules: {esm2_target_modules}")

    lora_cfg_esm = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=esm2_target_modules,
    )
    esm2_lora = get_peft_model(esm2_base, lora_cfg_esm)
    esm2_lora.print_trainable_parameters()

    esm2_hidden = esm2_base.config.hidden_size
    esm2_model  = LoRAClassifier(esm2_lora, esm2_hidden).to(device)

    # ── Training loop ───────────────────────────────────────────────────────────
    optimizer_esm = AdamW(esm2_model.parameters(), lr=2e-4)

    esm2_train_losses, esm2_val_losses = [], []

    for epoch in range(EPOCHS):
        esm2_model.train()
        epoch_loss = 0.0
        for input_ids, attention_mask, batch_labels in train_loader_esm:
            input_ids, attention_mask, batch_labels = (
                input_ids.to(device), attention_mask.to(device), batch_labels.to(device)
            )
            optimizer_esm.zero_grad()
            logits = esm2_model(input_ids, attention_mask)
            loss   = criterion(logits, batch_labels)
            loss.backward()
            optimizer_esm.step()
            epoch_loss += loss.item()

        esm2_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, attention_mask, batch_labels in val_loader_esm:
                input_ids, attention_mask, batch_labels = (
                    input_ids.to(device), attention_mask.to(device), batch_labels.to(device)
                )
                logits    = esm2_model(input_ids, attention_mask)
                val_loss += criterion(logits, batch_labels).item()

        esm2_train_losses.append(epoch_loss / len(train_loader_esm))
        esm2_val_losses.append(val_loss / len(val_loader_esm))
        print(f"  Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {esm2_train_losses[-1]:.4f} | Val Loss: {esm2_val_losses[-1]:.4f}")

    # ── Evaluate ────────────────────────────────────────────────────────────────
    esm2_model.eval()
    all_proba_esm = []
    with torch.no_grad():
        for input_ids, attention_mask, _ in val_loader_esm:
            logits = esm2_model(input_ids.to(device), attention_mask.to(device))
            all_proba_esm.extend(torch.sigmoid(logits).cpu().numpy())

    esm2_lora_metrics = compute_metrics(y_val, np.array(all_proba_esm))
    print("\nESM2 LoRA Validation Metrics:")
    for k, v in esm2_lora_metrics.items():
        print(f"  {k}: {v:.4f}")

# %% [markdown]
# === ESM2 LoRA Fine-Tuning ===
# 
# ESM2 LoRA target modules: ['query', 'key', 'value']
# 
# trainable params: 276,480 || all params: 33,776,881 || trainable%: 0.8185
# 
#   Epoch 1/10 | Train Loss: 0.6210 | Val Loss: 0.4670
# 
#   Epoch 2/10 | Train Loss: 0.3415 | Val Loss: 0.4005
# 
#   Epoch 3/10 | Train Loss: 0.2846 | Val Loss: 0.3491
# 
#   Epoch 4/10 | Train Loss: 0.2512 | Val Loss: 0.3192
# 
#   Epoch 5/10 | Train Loss: 0.2104 | Val Loss: 0.3087
# 
#   Epoch 6/10 | Train Loss: 0.1914 | Val Loss: 0.3187
# 
#   Epoch 7/10 | Train Loss: 0.1502 | Val Loss: 0.3807
# 
#   Epoch 8/10 | Train Loss: 0.1280 | Val Loss: 0.3989
# 
#   Epoch 9/10 | Train Loss: 0.1042 | Val Loss: 0.4087
# 
#   Epoch 10/10 | Train Loss: 0.0801 | Val Loss: 0.4426
# 
# ESM2 LoRA Validation Metrics: accuracy: 0.8533, precision: 0.8354, recall: 0.8800, f1: 0.8571, roc_auc: 0.9470, prc_auc: 0.9413

# %% [markdown]
# ---
# 
# ## 1.3 Frozen vs LoRA Comparison

# %%
# Frozen model metrics from assignment 3
frozen_metrics = {
    "NT_frozen":   {"accuracy": 0.5845, "precision": 0.5812, "recall": 0.5923,
                    "f1": 0.5832, "roc_auc": 0.6282, "prc_auc": 0.6077},
    "ESM2_frozen": {"accuracy": 0.5827, "precision": 0.5784, "recall": 0.5802,
                    "f1": 0.5822, "roc_auc": 0.6099, "prc_auc": 0.5929},
}

# Collect LoRA metrics
lora_results = {}
if RUN_NT_LORA:
    lora_results["NT_LoRA"] = nt_lora_metrics
if RUN_ESM2_LORA:
    lora_results["ESM2_LoRA"] = esm2_lora_metrics

# Build comparison DataFrame
all_results = {**frozen_metrics, **lora_results}
comparison_df = pd.DataFrame(all_results).T
print(comparison_df.to_string())

# %% [markdown]
# |   model    | accuracy | precision  |  recall |   f1   | roc_auc  | prc_auc |
# |------------|-----------|-----------|-----------|----------|----------|----------|
# | NT_frozen  |  0.584500 |  0.581200 |  0.592300 | 0.583200 | 0.628200 |  0.607700 |
# | ESM2_frozen|  0.582700 |  0.578400 |  0.580200 | 0.582200 | 0.609900 |  0.592900 |
# | NT_LoRA    |  0.760000 |  0.819672 |  0.666667 | 0.735294 | 0.814756 |  0.819454 |
# | ESM2_LoRA  |  0.853333 |  0.835443 |  0.880000 | 0.857143 | 0.947022 |  0.941252 |

# %%
# Visualize comparison
metrics_to_plot = ["accuracy", "f1", "roc_auc", "prc_auc"]

fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(16, 4))
for ax, metric in zip(axes, metrics_to_plot):
    vals = comparison_df[metric]
    colors = ["steelblue" if "frozen" in idx.lower() else "coral" for idx in vals.index]
    ax.bar(vals.index, vals.values, color=colors)
    ax.set_title(metric.upper())
    ax.set_xticklabels(vals.index, rotation=30, ha="right")
    ax.set_ylim(0, 1)

plt.suptitle("Frozen Backbone vs LoRA Fine-Tuning", fontsize=14)
plt.tight_layout()
fig_path = FIG_DIR / "frozen_backbone_vs_lora_ft.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[INFO] Saved figure: {fig_path}")

# %% [markdown]
# ![Metrics Comparision](figures/frozen_backbone_vs_lora_ft.png)
# Looking at the metrics, we can see that the metrics from LoRA fine-tuning are significantly better than the frozen backbone models. This suggests that even a small number of trainable parameters can lead to substantial performance improvements when fine-tuning large pretrained models on specific tasks. The ESM2 LoRA model, in particular, shows a marked increase in both F1 score and ROC-AUC compared to its frozen counterpart, indicating that the protein sequence information is being effectively leveraged through fine-tuning.
# 
# The precision-recall tradeoff seems to shift in favor of higher precision with LoRA fine-tuning, especially for the NT model. This could indicate that the LoRA adapters are helping the model to focus on more relevant features in the input sequences, leading to fewer false positives. The ESM2 LoRA model also shows a significant boost in recall, suggesting that it is better at identifying true positives compared to the frozen version. Overall, these results highlight the effectiveness of LoRA fine-tuning in improving model performance while maintaining parameter efficiency.

# %% [markdown]
# ---
# 
# ## 1.4 LoRA Hyperparameter Ablations
# 
# ### 1.4.1 LoRA Rank and Alpha Grid Search

# %%
import time

# Ablation: test different LoRA ranks and alpha values
LORA_RANKS  = [4, 8, 16, 32]
LORA_ALPHAS = [8, 16, 32]

# We run the ablation on ESM2 (change to NT if preferred)
ABLATION_EPOCHS = 3  # fewer epochs to keep runtime manageable

ablation_results = []

for rank in LORA_RANKS:
    for alpha in LORA_ALPHAS:
        print(f"\n--- rank={rank}, alpha={alpha} ---")
        t0 = time.time()

        # Build model
        base_model = load_esm2_backbone(ESM2_MODEL_ID, enable_attentions=False)
        ablation_target_modules = infer_lora_target_modules(base_model)
        print(f"Ablation LoRA target modules: {ablation_target_modules}")
        cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            target_modules=ablation_target_modules,
        )
        peft_model = get_peft_model(base_model, cfg)
        n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)

        model = LoRAClassifier(peft_model, base_model.config.hidden_size).to(device)
        opt   = AdamW(model.parameters(), lr=2e-4)

        # Train
        for _ in range(ABLATION_EPOCHS):
            model.train()
            for input_ids, attention_mask, batch_labels in train_loader_esm:
                input_ids, attention_mask, batch_labels = (
                    input_ids.to(device), attention_mask.to(device), batch_labels.to(device)
                )
                opt.zero_grad()
                loss = criterion(model(input_ids, attention_mask), batch_labels)
                loss.backward()
                opt.step()

        runtime = time.time() - t0

        # Evaluate
        model.eval()
        proba = []
        with torch.no_grad():
            for input_ids, attention_mask, _ in val_loader_esm:
                logits = model(input_ids.to(device), attention_mask.to(device))
                proba.extend(torch.sigmoid(logits).cpu().numpy())

        metrics = compute_metrics(y_val, np.array(proba))
        ablation_results.append({
            "rank": rank, "alpha": alpha,
            "n_trainable_params": n_trainable,
            "runtime_s": round(runtime, 1),
            **metrics,
        })
        print(f"  n_params={n_trainable:,} | runtime={runtime:.1f}s | "
              f"f1={metrics['f1']:.4f} | roc_auc={metrics['roc_auc']:.4f}")

ablation_df = pd.DataFrame(ablation_results)
print("\nAblation Results:")
print(ablation_df.to_string(index=False))

# %% [markdown]
# Ablation Results:
# 
# | rank | alpha | n_trainable_params | runtime_s | accuracy | precision | recall | f1 | roc_auc | prc_auc |
# |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
# | 4 | 8 | 138240 | 21.4 | 0.853333 | 0.835443 | 0.880000 | 0.857143 | 0.936533 | 0.932754 |
# | 4 | 16 | 138240 | 21.2 | 0.853333 | 0.827160 | 0.893333 | 0.858974 | 0.942044 | 0.939710 |
# | 4 | 32 | 138240 | 21.2 | 0.860000 | 0.837500 | 0.893333 | 0.864516 | 0.944000 | 0.941087 |
# | 8 | 8 | 276480 | 21.2 | 0.840000 | 0.831169 | 0.853333 | 0.842105 | 0.928711 | 0.923771 |
# | 8 | 16 | 276480 | 21.3 | 0.840000 | 0.840000 | 0.840000 | 0.840000 | 0.929244 | 0.922239 |
# | 8 | 32 | 276480 | 21.5 | 0.860000 | 0.846154 | 0.880000 | 0.862745 | 0.941156 | 0.930151 |
# | 16 | 8 | 552960 | 21.3 | 0.860000 | 0.837500 | 0.893333 | 0.864516 | 0.938667 | 0.934450 |
# | 16 | 16 | 552960 | 21.3 | 0.846667 | 0.833333 | 0.866667 | 0.849673 | 0.938667 | 0.932619 |
# | 16 | 32 | 552960 | 21.3 | 0.800000 | 0.835821 | 0.746667 | 0.788732 | 0.924267 | 0.916802 |
# | 32 | 8 | 1105920 | 21.4 | 0.846667 | 0.842105 | 0.853333 | 0.847682 | 0.935111 | 0.930364 |
# | 32 | 16 | 1105920 | 21.4 | 0.853333 | 0.827160 | 0.893333 | 0.858974 | 0.931733 | 0.917520 |
# | 32 | 32 | 1105920 | 21.4 | 0.813333 | 0.840580 | 0.773333 | 0.805556 | 0.932622 | 0.926138 |

# %% [markdown]
# ### 1.4.2 Visualize Parameter Efficiency vs Performance

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. F1 score heatmap over rank × alpha
pivot_f1 = ablation_df.pivot(index="rank", columns="alpha", values="f1")
sns.heatmap(pivot_f1, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[0])
axes[0].set_title("F1 Score")

# 2. ROC-AUC heatmap
pivot_auc = ablation_df.pivot(index="rank", columns="alpha", values="roc_auc")
sns.heatmap(pivot_auc, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[1])
axes[1].set_title("ROC-AUC")

# 3. Parameter efficiency scatter (trainable params vs F1)
scatter = axes[2].scatter(
    ablation_df["n_trainable_params"] / 1e3,
    ablation_df["f1"],
    c=ablation_df["runtime_s"],
    cmap="plasma",
    s=100,
)
for _, row in ablation_df.iterrows():
    axes[2].annotate(
        f"r={int(row['rank'])},a={int(row['alpha'])}",
        (row["n_trainable_params"] / 1e3, row["f1"]),
        fontsize=7,
    )
plt.colorbar(scatter, ax=axes[2], label="Runtime (s)")
axes[2].set_xlabel("Trainable Parameters (K)")
axes[2].set_ylabel("F1 Score")
axes[2].set_title("Parameter Efficiency vs Performance")

plt.suptitle("LoRA Hyperparameter Ablations (ESM2)", fontsize=14)
plt.tight_layout()
fig_path = FIG_DIR / "lora_hyp_ablations.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[INFO] Saved figure: {fig_path}")

# %% [markdown]
# ![LoRA Hyperparameters Ablations](figures/lora_hyp_ablations.png)
# Based on the above ablation results, we can see that the number of trainable parameters increases with higher rank values, as expected. The runtime remains relatively stable across different configurations, suggesting that the computational overhead of LoRA fine-tuning is not significantly affected by the choice of rank and alpha within this range. The f1 scores and ROC-AUC values show some variability, with certain configurations (e.g., rank=4, alpha=32) achieving higher performance metrics. However, there doesn't seem to be a clear linear relationship between the number of trainable parameters and performance, indicating that there may be a sweet spot in the hyperparameter space that balances parameter efficiency and model performance. This highlights the importance of hyperparameter tuning when applying LoRA fine-tuning to achieve optimal results on specific tasks.

# %% [markdown]
# ---
# 
# # Interpretability & Applications
# 
# ## 2.1 Attribution Analysis: Understanding Model Decisions
# 
# ---
# 
# ### 2.1.1 Integrated Gradients for Attribution

# %%
from captum.attr import IntegratedGradients

# We use the ESM2 LoRA model for interpretability (switch to NT if preferred)
# Ensure the model is in eval mode
esm2_model.eval()

# ── Select 5-10 high-confidence TF-positive predictions from the validation set ──
all_proba_esm_arr = np.array(all_proba_esm)
high_conf_idx = np.where(all_proba_esm_arr >= 0.9)[0][:10]  # top high-confidence positives
print(f"Selected {len(high_conf_idx)} high-confidence samples for attribution.")

# Gather tokenised inputs for these samples
selected_val_ids = val_enc_esm["input_ids"][high_conf_idx].to(device)
selected_val_mask = val_enc_esm["attention_mask"][high_conf_idx].to(device)

# %%
# ── Wrapper function for Integrated Gradients ────────────────────────────────
# IG operates on continuous input embeddings, not discrete token IDs.

def get_embedding_layer(model):
    """Return the token embedding layer from the ESM2 backbone."""
    return model.backbone.base_model.model.embeddings.word_embeddings

embedding_layer = get_embedding_layer(esm2_model)

def forward_from_embeddings(embeddings, attention_mask):
    """Forward pass that takes embeddings (not token IDs) as input."""
    outputs = esm2_model.backbone(
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
    )
    cls_emb = outputs.last_hidden_state[:, 0, :]
    return torch.sigmoid(esm2_model.classifier(cls_emb).squeeze(-1))


ig = IntegratedGradients(forward_from_embeddings)

all_attributions = []

for i in range(len(high_conf_idx)):
    input_ids_i   = selected_val_ids[i:i+1]
    att_mask_i    = selected_val_mask[i:i+1]

    # Token embeddings for the actual input
    input_embeds = embedding_layer(input_ids_i)  # (1, seq_len, hidden)

    # Baseline: embeddings of a zero / pad sequence
    baseline_ids  = torch.zeros_like(input_ids_i)
    baseline_embs = embedding_layer(baseline_ids)

    attrs, _ = ig.attribute(
        inputs=input_embeds,
        baselines=baseline_embs,
        additional_forward_args=(att_mask_i,),
        n_steps=50,
        return_convergence_delta=True,
    )
    # Summarise over the embedding dimension → one score per token position
    token_attrs = attrs.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    all_attributions.append(token_attrs)

print(f"Attribution shapes: {[a.shape for a in all_attributions]}")

# %% [markdown]
# Attribution shapes: [(512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,)]

# %%
# ── Visualise attributions as heatmaps ──────────────────────────────────────
fig, axes = plt.subplots(min(5, len(all_attributions)), 1,
                         figsize=(16, 2.5 * min(5, len(all_attributions))))
if len(all_attributions) == 1:
    axes = [axes]

for ax, attrs in zip(axes, all_attributions[:5]):
    im = ax.imshow(
        attrs[np.newaxis, :],  # (1, seq_len)
        aspect="auto",
        cmap="RdBu_r",
        vmin=-np.abs(attrs).max(),
        vmax=np.abs(attrs).max(),
    )
    ax.set_yticks([])
    ax.set_xlabel("Token Position")
    plt.colorbar(im, ax=ax, orientation="vertical", pad=0.01)

plt.suptitle("Integrated Gradients — Token-Level Attributions (ESM2)", fontsize=13)
plt.tight_layout()
fig_path = FIG_DIR / "ig_token_level_attributions.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[INFO] Saved figure: {fig_path}")

# %% [markdown]
# ![IG Token Level Attributions](figures/ig_token_level_attributions.png)

# %%
# ── Extract top-5 high-attribution windows (window size = 30) ───────────────
WINDOW_SIZE = 30
TOP_K_WINDOWS = 5

def top_k_windows(attrs, k=TOP_K_WINDOWS, w=WINDOW_SIZE):
    scores = np.convolve(np.abs(attrs), np.ones(w) / w, mode="valid")
    top_starts = np.argsort(scores)[::-1][:k]
    return [(s, s + w, scores[s]) for s in sorted(top_starts)]

print("Top attribution windows per sequence:")
for i, attrs in enumerate(all_attributions[:5]):
    windows = top_k_windows(attrs)
    print(f"  Sample {i}: {windows}")

# %% [markdown]
# Top attribution windows per sequence:
# 
#   Sample 0: [(475, 505, 0.009095029276795685), (476, 506, 0.009547541531113287), (477, 507, 0.009630925751601656), (478, 508, 0.009594724979251621), (479, 509, 0.009484857600182292)]
# 
#   Sample 1: [(0, 30, 0.007542296755127607), (1, 31, 0.00758620580503096), (2, 32, 0.004681570370060701), (3, 33, 0.004642707514964664), (4, 34, 0.00453136881502966)]
# 
#   Sample 2: [(0, 30, 0.023695754345195982), (1, 31, 0.024074877480355396), (2, 32, 0.0189634165338551), (3, 33, 0.018578170076943932), (4, 34, 0.016965208163795372)]
# 
#   Sample 3: [(0, 30, 0.009418478303511314), (1, 31, 0.009578890033299101), (2, 32, 0.005724371649557727), (3, 33, 0.005572640241977449), (5, 35, 0.005525510670850054)]
#   
#   Sample 4: [(0, 30, 0.005426976657084501), (1, 31, 0.005429855881569288), (2, 32, 0.004790489288279787), (3, 33, 0.003919008903903886), (5, 35, 0.0035609287529950956)]

# %%
# ── Sequence logo for the highest-attribution window of the first sample ─────
import logomaker

# Decode tokens for the first high-confidence sample
sample_tokens = esm2_tokenizer.convert_ids_to_tokens(
    selected_val_ids[0].cpu().tolist()
)

# Use top window of sample 0
win_start, win_end, _ = top_k_windows(all_attributions[0])[0]
window_tokens  = sample_tokens[win_start:win_end]
window_attrs   = all_attributions[0][win_start:win_end]

# Build a single-row DataFrame for logomaker (one score per position)
logo_df = pd.DataFrame(
    np.outer(np.clip(window_attrs, 0, None), np.ones(1)),
    index=range(len(window_tokens)),
    columns=["attr"],
)

fig, ax = plt.subplots(figsize=(14, 3))
ax.bar(range(len(window_tokens)), np.clip(window_attrs, 0, None), color="steelblue")
ax.set_xticks(range(len(window_tokens)))
ax.set_xticklabels(window_tokens, rotation=90, fontsize=8)
ax.set_title(f"Attribution Bar Chart — Top Window (pos {win_start}–{win_end})")
ax.set_ylabel("Attribution Score")
plt.tight_layout()
fig_path = FIG_DIR / "attribution_bar_chart.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[INFO] Saved figure: {fig_path}")

# %% [markdown]
# ### 2.1.1b Known TF Domain Comparison (InterPro)
# 
# For the high-attribution regions identified above, we compare against known protein domains using the **InterPro** API. Since we're analyzing ESM2 (protein) attributions, InterPro is more appropriate than JASPAR (which is for nucleotide motifs). This helps validate whether high-attribution regions correspond to known DNA-binding domains or other functional domains characteristic of transcription factors.
# 
# **Note**: InterPro API queries may occasionally timeout or fail due to rate limiting. If you see errors, the system will gracefully handle them and report the issue. For production use, consider installing a local InterPro database or using batch search functionality.

# %%
# ── Compare top-attribution windows against known TF domains (InterPro) ──────
# Extract actual protein sequences for the high-attribution windows
# and look them up in InterPro for known protein domains

import time

def query_interpro_domains(sequence: str, email: str, timeout: int = 600) -> dict:
    """
    Query InterProScan 5 API asynchronously.
    Requires an email address per EBI usage policy.

    Returns the raw JSON response from InterProScan.
    """
    BASE_URL = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"

    try:
        submit_res = requests.post(
            f"{BASE_URL}/run",
            data={
                "email": email,
                "sequence": sequence,
                "goterms": "false",
                "pathways": "false",
            },
            timeout=600,
        )

        if submit_res.status_code != 200:
            return {"error": f"Submission failed: {submit_res.status_code}", "msg": submit_res.text}

        job_id = submit_res.text.strip()
        print(f"Job submitted successfully. Job ID: {job_id}")

        start_time = time.time()
        while True:
            if (time.time() - start_time) > timeout:
                return {"error": "Processing timeout exceeded"}

            status_res = requests.get(f"{BASE_URL}/status/{job_id}", timeout=600)
            status = status_res.text.strip()

            if status == "FINISHED":
                break
            if status in ["ERROR", "FAILURE"]:
                return {"error": f"InterProScan job failed with status: {status}"}

            time.sleep(10)

        result_res = requests.get(f"{BASE_URL}/result/{job_id}/json", timeout=600)
        if result_res.status_code == 200:
            return result_res.json()
        return {"error": f"Failed to fetch results: {result_res.status_code}", "msg": result_res.text}

    except Exception as e:
        return {"error": str(e)}


def parse_interpro_matches(interpro_json: dict) -> list[dict]:
    """Flatten InterProScan JSON into a list of annotated matches."""
    parsed = []
    results = interpro_json.get("results", [])

    for seq_result in results:
        sequence = seq_result.get("sequence", "")
        matches = seq_result.get("matches", [])

        for match in matches:
            signature = match.get("signature", {}) or {}
            entry = match.get("entry", {}) or {}
            accession = entry.get("accession") or signature.get("accession") or match.get("model-ac") or "N/A"
            name = entry.get("name") or signature.get("name") or accession
            description = entry.get("description") or signature.get("description")
            domain_type = entry.get("type") or signature.get("type") or match.get("type") or "UNKNOWN"

            for loc in match.get("locations", []):
                parsed.append({
                    "sequence": sequence,
                    "accession": accession,
                    "name": name,
                    "description": description,
                    "type": domain_type,
                    "start": loc.get("start"),
                    "end": loc.get("end"),
                    "evalue": match.get("evalue", loc.get("evalue")),
                    "score": match.get("score", loc.get("score")),
                    "signature_library": signature.get("signatureLibraryRelease", {}).get("library"),
                })

    return parsed


# Extract sequences for top windows from first few samples
print("=== InterPro Domain Comparison for High-Attribution Regions ===\n")

interpro_results = []

for sample_idx in range(min(3, len(all_attributions))):  # analyze first 3 samples
    attrs = all_attributions[sample_idx]
    windows = top_k_windows(attrs, k=3)  # top 3 windows

    # Get original protein sequence for this sample
    orig_idx = val_idx[high_conf_idx[sample_idx]]
    protein_seq = df.loc[orig_idx, "protein_seq"]

    print(f"Sample {sample_idx} (TF confidence: {all_proba_esm_arr[high_conf_idx[sample_idx]]:.3f})")
    print(f"  Protein sequence length: {len(protein_seq)}")

    for win_start, win_end, score in windows:
        # Extract window sequence (handle tokenization)
        # Note: ESM2 uses BOS/EOS tokens, so map token positions back to original sequence
        # For simplicity, use approximate mapping (token_pos ≈ aa_pos for ESM2)
        win_start_adj = max(0, win_start - 1)  # account for BOS token
        win_end_adj = min(len(protein_seq), win_end - 1)

        if win_end_adj > win_start_adj:
            window_seq = protein_seq[win_start_adj:win_end_adj]
            print(f"    Window [{win_start_adj}-{win_end_adj}] (attr={score:.6f}): {window_seq[:20]}...")

            # Query InterPro and parse the raw JSON response correctly
            interpro_json = query_interpro_domains(window_seq, email="abcd" + str(sample_idx) + "@abc.com")

            if "error" in interpro_json:
                print(f"      InterPro query failed: {interpro_json['error']}")
                continue

            parsed_matches = parse_interpro_matches(interpro_json)
            if parsed_matches:
                print(f"      Found {len(parsed_matches)} match(es):")
                for match in parsed_matches[:3]:  # show top 3 matches
                    print(
                        f"        - {match['name']} ({match['accession']}), "
                        f"type={match['type']}, region={match['start']}-{match['end']}"
                    )
                    interpro_results.append({
                        "sample": sample_idx,
                        "window_pos": f"{win_start_adj}-{win_end_adj}",
                        "domain": match["name"],
                        "accession": match["accession"],
                        "type": match["type"],
                        "domain_start": match["start"],
                        "domain_end": match["end"],
                        "attribution": score,
                    })
            else:
                print("      No domains found (but sequence is valid)")

    print()

# Summary table of findings
if interpro_results:
    interpro_df = pd.DataFrame(interpro_results)
    print("\nSummary of InterPro Domain Annotations:")
    print(interpro_df.to_string(index=False))
else:
    print("\nNo InterPro domains found in the queried sequences.")
    print("(This may be expected for short windows, rate limits, or API service issues.)")

# %% [markdown]
# ![Attributions Bar Chart](figures/attribution_bar_chart.png)

# %% [markdown]
# The top attribution windows for the first sample show a cluster of high scores around 475–505, which may indicate a biologically relevant motif or domain in the protein sequence. To determine if these regions correspond to known domains, we could cross-reference the high-attribution positions with databases of protein motifs (e.g., Pfam) or perform a BLAST search to find similar sequences with annotated functions. There are other high-attribution windows at the beginning of the sequence (0–30) in other samples, which could suggest the presence of signal peptides or other N-terminal features. Species-specific patterns might emerge if certain motifs are conserved in specific taxa, so it would be interesting to analyze the taxonomic distribution of the samples and see if high-attribution regions correlate with particular evolutionary lineages.

# %% [markdown]
# ---
# 
# ### 2.1.2 Attention Weights Visualization

# %%
# ── Extract attention weights for the same 5-10 sequences ───────────────────
esm2_model.eval()

attention_maps = []  # list of (n_heads, seq_len, seq_len) arrays
attention_attr_pairs = []

# Ensure the fine-tuned PEFT wrapper still advertises attention outputs.
for cfg_owner in (esm2_model.backbone, getattr(esm2_model.backbone, "base_model", None)):
    cfg = getattr(cfg_owner, "config", None)
    if cfg is None:
        continue
    cfg.output_attentions = True
    if hasattr(cfg, "attn_implementation"):
        cfg.attn_implementation = "eager"

with torch.no_grad():
    for i in range(len(high_conf_idx)):
        input_ids_i = selected_val_ids[i:i+1]
        att_mask_i  = selected_val_mask[i:i+1]

        outputs = esm2_model.backbone(
            input_ids=input_ids_i,
            attention_mask=att_mask_i,
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )
        attentions = getattr(outputs, "attentions", None)
        if attentions is None or len(attentions) == 0:
            print(f"[WARN] No attention tensors returned for sample {i}; skipping.")
            continue
        last_layer_attn = attentions[-1]  # (1, n_heads, L, L)
        attn_np = last_layer_attn.squeeze(0).cpu().numpy()
        attention_maps.append(attn_np)  # (n_heads, L, L)
        attention_attr_pairs.append((all_attributions[i], attn_np))

print(f"Attention map shapes: {[a.shape for a in attention_maps]}")

# %% [markdown]
# Attention map shapes: [(20, 512, 512), (20, 512, 512), (20, 512, 512), (20, 512, 512), (20, 512, 512), (20, 512, 512), (20, 512, 512), (20, 512, 512), (20, 512, 512), (20, 512, 512)]

# %%
# ── Average attention: use the head with highest variance ───────────────────
def select_top_heads(attn_matrix, n_top=4):
    """Return indices of the n_top attention heads with highest variance."""
    variances = attn_matrix.reshape(attn_matrix.shape[0], -1).var(axis=1)
    return np.argsort(variances)[::-1][:n_top]

if len(attention_maps) == 0:
    print("[WARN] No attention maps available; skipping attention heatmap plot.")
else:
    fig, axes = plt.subplots(1, min(5, len(attention_maps)), figsize=(18, 4))
    if len(attention_maps) == 1:
        axes = [axes]

    for ax, attn in zip(axes, attention_maps[:5]):
        top_heads  = select_top_heads(attn)
        avg_attn   = attn[top_heads].mean(axis=0)  # (L, L)
        seq_len    = min(avg_attn.shape[0], 64)  # show at most 64 positions
        sns.heatmap(
            avg_attn[:seq_len, :seq_len],
            ax=ax, cmap="viridis",
            cbar=False, xticklabels=False, yticklabels=False,
        )
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")

    plt.suptitle("Attention Weights — Last Layer (avg top-4 heads)", fontsize=13)
    plt.tight_layout()
    fig_path = FIG_DIR / "attention_weights.png"
    fig.savefig(str(fig_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved figure: {fig_path}")

# %% [markdown]
# ![Attention Weights](figures/attention_weights.png)

# %%
# ── Compare attention rollout vs IG attributions ────────────────────────────
if len(attention_maps) == 0:
    print("[WARN] Skipping IG-vs-attention comparison because no attention maps were returned.")
else:
    n_compare = min(5, len(attention_attr_pairs))
    fig, axes = plt.subplots(2, n_compare, figsize=(16, 5))
    if n_compare == 1:
        axes = np.array(axes).reshape(2, 1)

    for col, (attrs, attn) in enumerate(attention_attr_pairs[:n_compare]):
        # IG attribution (row 0)
        axes[0, col].imshow(
            attrs[np.newaxis, :64], aspect="auto",
            cmap="RdBu_r",
            vmin=-np.abs(attrs[:64]).max(), vmax=np.abs(attrs[:64]).max(),
        )
        axes[0, col].set_title(f"Sample {col+1}")
        axes[0, col].set_yticks([])
        if col == 0:
            axes[0, col].set_ylabel("IG")

        # Mean attention over keys (row 1)
        top_heads = select_top_heads(attn)
        avg_attn  = attn[top_heads].mean(axis=0).mean(axis=1)  # mean over keys -> (L,)
        axes[1, col].imshow(
            avg_attn[np.newaxis, :64], aspect="auto", cmap="YlOrRd"
        )
        axes[1, col].set_yticks([])
        axes[1, col].set_xlabel("Token Position")
        if col == 0:
            axes[1, col].set_ylabel("Attention")

    plt.suptitle("Comparison: IG Attributions vs Attention Weights", fontsize=13)
    plt.tight_layout()
    fig_path = FIG_DIR / "ig_attributions_vs_attention_weights.png"
    fig.savefig(str(fig_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved figure: {fig_path}")

# %% [markdown]
# ![IG Attributions vs Attention Weights](figures/ig_attributions_vs_attention_weights.png)

# %% [markdown]
# In the attention heatmaps, we observe that certain heads exhibit more focused attention patterns, attending strongly to specific token positions, while others show more diffuse attention across the sequence. When comparing the IG attributions with the attention weights, we see that in some samples, high IG attribution regions correspond to areas where attention is also concentrated, suggesting that the model is indeed focusing on biologically relevant motifs. However, there are also cases where IG highlights certain positions that do not align with strong attention weights, which could indicate that the model is using a combination of local and global information to make predictions. This discrepancy might arise because attention weights capture pairwise interactions between tokens, while IG captures the overall contribution of each token to the final prediction. Therefore, both methods provide complementary insights into the model's decision-making process.

# %% [markdown]
# ---
# 
# ### 2.1.3 Sparse Autoencoder (SAE) Feature Extraction

# %%
# ── Sparse Autoencoder definition ───────────────────────────────────────────

class SparseAutoencoder(nn.Module):
    """
    Wide sparse autoencoder:
      encoder : input_dim → dict_dim  (with ReLU activation)
      decoder : dict_dim  → input_dim (linear reconstruction)

    Loss = MSE(reconstruction, input) + l1_coeff * mean(|features|)
    """

    def __init__(self, input_dim: int, dict_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, dict_dim)
        self.decoder = nn.Linear(dict_dim, input_dim, bias=False)

    def forward(self, x):
        features = torch.relu(self.encoder(x))   # sparse activations
        recon    = self.decoder(features)
        return recon, features


def sae_loss(x, recon, features, l1_coeff: float = 1e-3):
    mse_loss = nn.functional.mse_loss(recon, x)
    l1_loss  = features.abs().mean()
    return mse_loss + l1_coeff * l1_loss, mse_loss.item(), l1_loss.item()

# %%
# ── Extract layer-6 activations from the ESM2 LoRA model ────────────────────
# We collect activations for a broader sample: 10-20 sequences (mix TF / non-TF)

SAE_SAMPLE_SIZE = 20

# Pick balanced sample
pos_idx = np.where(y_val == 1)[0][:SAE_SAMPLE_SIZE // 2]
neg_idx = np.where(y_val == 0)[0][:SAE_SAMPLE_SIZE // 2]
sae_sample_idx = np.concatenate([pos_idx, neg_idx])
sae_labels     = y_val[sae_sample_idx]

sae_input_ids  = val_enc_esm["input_ids"][sae_sample_idx].to(device)
sae_attn_mask  = val_enc_esm["attention_mask"][sae_sample_idx].to(device)

activation_cache = {}

def hook_layer6(module, input, output):
    # ESM2 layer output is a tuple; first element is the hidden state
    hidden = output[0] if isinstance(output, tuple) else output
    # Mean-pool over sequence length → (batch, hidden_size)
    activation_cache["layer6"] = hidden.mean(dim=1).detach().cpu()

# Register hook on layer 6 of ESM2
hook_handle = esm2_model.backbone.base_model.model.encoder.layer[6].register_forward_hook(
    hook_layer6
)

esm2_model.eval()
with torch.no_grad():
    _ = esm2_model(sae_input_ids, sae_attn_mask)

hook_handle.remove()

activations = activation_cache["layer6"]  # (SAE_SAMPLE_SIZE, hidden_size)
print(f"Activations shape: {activations.shape}")

# %% [markdown]
# Activations shape: torch.Size([20, 480])

# %%
# ── Train SAE ────────────────────────────────────────────────────────────────
INPUT_DIM = activations.shape[1]
DICT_DIM  = INPUT_DIM * 4   # wide dictionary (4× expansion)
SAE_EPOCHS = 200
L1_COEFF   = 1e-3

sae = SparseAutoencoder(INPUT_DIM, DICT_DIM).to(device)
sae_opt = torch.optim.Adam(sae.parameters(), lr=1e-3)

acts_tensor = activations.to(device)

sae_losses = []
for epoch in range(SAE_EPOCHS):
    sae.train()
    sae_opt.zero_grad()
    recon, features = sae(acts_tensor)
    loss, mse, l1  = sae_loss(acts_tensor, recon, features, l1_coeff=L1_COEFF)
    loss.backward()
    sae_opt.step()
    sae_losses.append(loss.item())
    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1}/{SAE_EPOCHS} | Loss: {loss.item():.6f} "
              f"(MSE: {mse:.6f}, L1: {l1:.6f})")

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(sae_losses)
ax.set_xlabel("Epoch")
ax.set_ylabel("Total Loss")
ax.set_title("SAE Training Loss")
fig.tight_layout()
fig_path = FIG_DIR / "sae_training_loss.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"[INFO] Saved figure: {fig_path}")

# %% [markdown]
# ![SAE Training Loss](figures/sae_training_loss.png)

# %%
# ── Analyze learned features ─────────────────────────────────────────────────
sae.eval()
with torch.no_grad():
    _, features_all = sae(acts_tensor)

features_np = features_all.cpu().numpy()  # (n_samples, dict_dim)
print(f"Feature matrix shape: {features_np.shape}")
print(f"Sparsity (fraction zero): {(features_np == 0).mean():.3f}")

# %% [markdown]
# Feature matrix shape: (20, 1920)
# 
# Sparsity (fraction zero): 0.720

# %%
# ── Which features activate most strongly for TFs vs non-TFs? ───────────────
tf_idx  = np.where(sae_labels == 1)[0]
non_tf_idx = np.where(sae_labels == 0)[0]

tf_mean     = features_np[tf_idx].mean(axis=0)
non_tf_mean = features_np[non_tf_idx].mean(axis=0)

diff = tf_mean - non_tf_mean  # positive → more active for TFs

top10_tf     = np.argsort(diff)[::-1][:10]
top10_non_tf = np.argsort(diff)[:10]

print("Top 10 features more active for TFs:", top10_tf)
print("Top 10 features more active for non-TFs:", top10_non_tf)

# %% [markdown]
# Top 10 features more active for TFs: [1525 1664 1704  297  170   55 1815 1370 1480 1711]
# 
# Top 10 features more active for non-TFs: [  38 1371  139 1304 1529  424 1826   93  608 1248]

# %%
# ── Heatmap: feature activations for TF vs non-TF ───────────────────────────
TOP_FEATURES = np.argsort(np.abs(diff))[::-1][:50]  # top 50 discriminative features

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(
    features_np[tf_idx][:, TOP_FEATURES],
    ax=axes[0], cmap="YlOrRd",
    xticklabels=False, yticklabels=[f"TF_{i}" for i in range(len(tf_idx))],
)
axes[0].set_title("TF samples — Top 50 SAE Features")
axes[0].set_xlabel("Feature Index")

sns.heatmap(
    features_np[non_tf_idx][:, TOP_FEATURES],
    ax=axes[1], cmap="YlOrRd",
    xticklabels=False, yticklabels=[f"nonTF_{i}" for i in range(len(non_tf_idx))],
)
axes[1].set_title("Non-TF samples — Top 50 SAE Features")
axes[1].set_xlabel("Feature Index")

plt.suptitle("SAE Feature Activations: TF vs Non-TF", fontsize=13)
plt.tight_layout()
fig_path = FIG_DIR / "sae_feature_activations.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[INFO] Saved figure: {fig_path}")

# %% [markdown]
# ![SAE Feature Activations](figures/sae_feature_activations.png)
# In the above heatmaps, we can see that certain SAE features are consistently more active in non-TF samples compared to TF samples, and vice versa. This suggests that the SAE has learned to capture some underlying patterns that differentiate TFs from non-TFs. To determine if these features correspond to known protein domains, we could analyze the sequences that strongly activate each feature and perform motif enrichment analysis or compare them against databases of protein domains (e.g., Pfam). Additionally, we could investigate if the top SAE features align with the high-attribution regions identified by IG and attention analyses, which would provide further evidence that these features are biologically meaningful and relevant to the model's predictions.

# %%
# ── Compare SAE features to IG attributions and Attention results ────────────
# For each of the SAE samples that also appear in our IG sample set,
# compare the most active SAE feature against the IG top window.

print("=== Comparing SAE Features to IG Attributions & Attention Results ===\n")

print("1. Qualitative SAE vs IG Comparison:")
print("-" * 60)

# Find overlap between SAE sample indices and IG sample indices
overlap_samples = min(5, len(sae_labels), len(all_attributions))

for i in range(overlap_samples):
    top_sae_feat = np.argsort(features_np[i])[::-1][:5]
    label_str = "TF" if sae_labels[i] == 1 else "non-TF"
    
    print(f"\nSample {i} ({label_str}):")
    print(f"  Top 5 SAE features: {top_sae_feat}")
    print(f"  Top SAE activation: {features_np[i, top_sae_feat[0]]:.4f}")
    
    # Get IG attribution info for this sample
    if i < len(all_attributions):
        attrs = all_attributions[i]
        top_attr_windows = top_k_windows(attrs, k=3)
        print(f"  IG top attribution window positions: {[f'{w[0]}-{w[1]}' for w in top_attr_windows]}")
        print(f"  IG top attribution score: {top_attr_windows[0][2]:.6f}")
        
        # Get attention info
        if i < len(attention_maps):
            top_heads = select_top_heads(attention_maps[i])
            avg_attn = attention_maps[i][top_heads].mean(axis=0).mean(axis=1)[:64]
            top_attn_pos = np.argsort(avg_attn)[::-1][:3]
            print(f"  Attention peak positions: {list(top_attn_pos)}")
            print(f"  Attention peak value: {avg_attn[top_attn_pos[0]]:.6f}")

print("\n" + "="*60)
print("2. Quantitative Analysis: SAE-IG Alignment")
print("="*60)

# For samples where all three methods are available, check if they highlight similar regions
alignment_scores = []

for i in range(overlap_samples):
    if i < len(all_attributions) and i < len(attention_maps):
        # Check if top SAE features align with top IG attribution regions
        attrs = all_attributions[i]
        top_attr_windows = top_k_windows(attrs, k=1)
        ig_window_start = top_attr_windows[0][0]
        ig_window_end = top_attr_windows[0][1]
        
        # Also check attention
        top_heads = select_top_heads(attention_maps[i])
        avg_attn = attention_maps[i][top_heads].mean(axis=0).mean(axis=1)[:64]
        attn_peak_pos = np.argmax(avg_attn)
        
        is_tf = "TF" if sae_labels[i] == 1 else "non-TF"
        
        alignment_scores.append({
            "sample": i,
            "label": is_tf,
            "top_sae_feature": np.argsort(features_np[i])[::-1][0],
            "top_sae_activation": features_np[i, np.argsort(features_np[i])[::-1][0]],
            "ig_window_start": ig_window_start,
            "ig_window_end": ig_window_end,
            "ig_max_attribution": top_attr_windows[0][2],
            "attention_peak_pos": attn_peak_pos,
            "attention_peak_val": avg_attn[attn_peak_pos],
        })

if alignment_scores:
    alignment_df = pd.DataFrame(alignment_scores)
    print("\nSummary: Position of peaks in each method")
    print(alignment_df.to_string(index=False))

print("\n" + "="*60)
print("3. Interpretation: Do the methods agree?")
print("="*60)
print("""
✓ HIGH AGREEMENT (IG, Attention, SAE highlight same region):
  → Strong evidence that this is a real biological pattern
  → Model consistently identifies this region as important
  → Feature likely captures meaningful TF structure

△ PARTIAL AGREEMENT (2 out of 3 methods agree):
  → May reflect complementary information sources
  → IG captures gradient-based importance (what changes output most)
  → Attention captures learned focus patterns
  → SAE captures learned feature decomposition
  → Disagreement is informative - different perspectives

✗ LOW/NO AGREEMENT:
  → Methods see different patterns
  → Could indicate:
    - Abstract features (SAE) vs. direct importance (IG)
    - Attention may attend to padding tokens
    - SAE may capture nuanced patterns IG misses
  → Suggests need for manual inspection
""")

# %%
# ── Map SAE features to InterPro domains ──────────────────────────────────────
# For each important SAE feature, query InterPro for sequences that activate it

# Define important features if not already defined
if 'important_features' not in locals():
    important_features = top10_tf[:5]  # focus on top 5 TF features

print("\n" + "="*60)
print("SAE Feature to Known Domain Mapping (via InterPro)")
print("="*60 + "\n")

feature_domain_mapping = []

for feat_idx in important_features[:3]:  # analyze top 3 features (limit API calls)
    print(f"\nFeature {feat_idx} - Searching for domains in activating sequences...")

    activation_scores = features_np[:, feat_idx]
    top_activators = np.argsort(activation_scores)[::-1][:2]  # top 2 samples

    for sample_idx in top_activators:
        orig_idx = val_idx[sample_idx]
        protein_seq = df.loc[orig_idx, "protein_seq"]
        gene_symbol = df.loc[orig_idx, "symbol"]
        label_str = "TF" if sae_labels[sample_idx] == 1 else "non-TF"

        # Query full sequence against InterPro (or a window if sequence is very long)
        query_seq = protein_seq[:500] if len(protein_seq) > 500 else protein_seq

        print(f"  → {gene_symbol} ({label_str}, len={len(protein_seq)})")

        # Use the corrected InterProScan wrapper and pass the required email argument
        interpro_json = query_interpro_domains(query_seq, email="abcd" + str(sample_idx) + "@abc.com")

        if "error" in interpro_json:
            print(f"    InterPro query failed: {interpro_json['error']}")
            continue

        parsed_matches = parse_interpro_matches(interpro_json)
        if parsed_matches:
            print(f"    Found {len(parsed_matches)} match(es):")
            for match in parsed_matches[:3]:
                print(
                    f"      • {match['name']} ({match['accession']}), "
                    f"type={match['type']}, region={match['start']}-{match['end']}"
                )
                feature_domain_mapping.append({
                    "feature_idx": feat_idx,
                    "gene": gene_symbol,
                    "is_tf": sae_labels[sample_idx],
                    "domain": match["name"],
                    "accession": match["accession"],
                    "type": match["type"],
                    "domain_start": match["start"],
                    "domain_end": match["end"],
                })
        else:
            print("    No known domains found")

if feature_domain_mapping:
    print("\n" + "="*60)
    print("Feature-Domain Summary Table:")
    print("="*60)
    feature_domain_df = pd.DataFrame(feature_domain_mapping)
    print(feature_domain_df.to_string(index=False))
else:
    print("\nNo InterPro domain mappings found.")
    print("This may be expected if features capture abstract patterns not matching annotated domains.")

# %%
# ── Interpret SAE features by analyzing which sequences activate them ────────────
# For each important feature, find which training sequences activate it most strongly,
# then look for common patterns (e.g., enriched amino acids, structural properties)

print("=== SAE Feature Interpretation ===\n")

# Analyze top TF-activating features
important_features = top10_tf[:5]  # focus on top 5 TF features

for feat_idx in important_features:
    print(f"\nFeature {feat_idx} (more active in TFs):")
    print(f"  Avg activation (TF): {tf_mean[feat_idx]:.4f}")
    print(f"  Avg activation (non-TF): {non_tf_mean[feat_idx]:.4f}")
    print(f"  Difference: {diff[feat_idx]:.4f}")
    
    # Find samples that activate this feature most strongly
    activation_scores = features_np[:, feat_idx]
    top_activators = np.argsort(activation_scores)[::-1][:3]  # top 3 samples
    
    print(f"  Top activating samples:")
    for rank, sample_idx in enumerate(top_activators):
        label_str = "TF" if sae_labels[sample_idx] == 1 else "non-TF"
        activation_val = activation_scores[sample_idx]
        
        # Get the original protein sequence for this sample
        orig_idx = val_idx[sample_idx]
        protein_seq = df.loc[orig_idx, "protein_seq"]
        gene_symbol = df.loc[orig_idx, "symbol"]
        
        print(f"    {rank+1}. Sample {sample_idx} ({gene_symbol}, {label_str}): "
              f"activation={activation_val:.4f}, seq_len={len(protein_seq)}")
        
        # Analyze amino acid composition in top 200 aa of sequence
        window = protein_seq[:200] if len(protein_seq) > 200 else protein_seq
        aa_counts = {}
        for aa in set(window):
            aa_counts[aa] = window.count(aa) / len(window)
        
        # Find enriched amino acids (compared to average)
        avg_freq = 1.0 / 20  # rough average for 20 amino acids
        enriched = {aa: freq for aa, freq in sorted(aa_counts.items(), 
                                                     key=lambda x: x[1], reverse=True)[:3]}
        print(f"       Enriched AAs: {enriched}")

print("\n" + "="*60)
print("Feature Interpretation Strategy:")
print("="*60)
print("""
1. **Amino Acid Enrichment**: Look for enriched amino acids (C, H, P, E, K)
   - C (Cysteine) → Zinc finger motifs
   - H (Histidine) → Zinc finger motifs
   - K (Lysine), E (Glutamic acid) → Basic regions (DNA binding)
   - P (Proline) → Turns in helical structures

2. **Query InterPro for sequences that activate each feature**:
   - Extract top-activating sequences
   - Search for known domains (done in section 2.1.1b)

3. **Compare to IG/Attention results**:
   - Do high-IG regions correspond to high-SAE feature activation?
   - If yes → suggests feature captures real biological patterns
""")


# %% [markdown]
# ---
# 
# ### 2.1.3b SAE Feature Interpretation: Understanding What Features Represent
# 
# **Challenge**: SAE features are just numbers (indices 1525, 1664, etc.). How do we know what they represent biologically?
# 
# **Solution**: Analyze which sequences **activate** each important feature, identify patterns, and query biological databases.
# 

# %% [markdown]
# SAE vs IG comparison (qualitative):
# 
#   Sample 0 (TF) — top SAE features: [ 308 1450  569 1557 1272]
# 
#   Sample 1 (TF) — top SAE features: [1525  308 1450 1261 1557]
# 
#   Sample 2 (TF) — top SAE features: [ 477 1782 1250 1681 1040]
# 
#   Sample 3 (TF) — top SAE features: [1450 1159  308 1664 1557]
#   
#   Sample 4 (TF) — top SAE features: [1837 1782  477  139  299]

# %% [markdown]
# The SAE features that activate most strongly for TFs may correspond to specific sequence patterns or structural motifs that are characteristic of transcription factors, such as DNA-binding domains (e.g., helix-turn-helix, zinc fingers). To determine if these features align with known protein domains, we could analyze the sequences that strongly activate each feature and perform motif enrichment analysis against databases like Pfam. When comparing the SAE features to the IG and attention attribution results, we might find that certain SAE features correspond to the high-attribution regions identified by IG and attention analyses, suggesting that these features capture biologically relevant information that the model uses for its predictions. However, it's also possible that some SAE features capture more abstract patterns that do not directly align with specific motifs but still contribute to the model's decision-making process.

# %% [markdown]
# # References:
# - https://github.com/sivasatvik/nyu-ai-in-genomics-a4/tasks/6bd42c14-47cd-419f-ad76-0a346e2ddec6 - created the PR with the relevant changes: https://github.com/sivasatvik/nyu-ai-in-genomics-a4/pull/1
# - Various auto completes in Visual Code IDE
# 


