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

# %%
# Load split indices from Assignment 3
with open(DATA_DIR / "tf_split_indices.json", "r") as f:
    split_indices = json.load(f)

print("Available splits:", list(split_indices.keys()))

# %%
# Load precomputed embeddings
dna_embeddings = np.load(DATA_DIR / "dna_embeddings.npy")
protein_embeddings = np.load(DATA_DIR / "protein_embeddings.npy")

print(f"DNA embeddings shape:     {dna_embeddings.shape}")
print(f"Protein embeddings shape: {protein_embeddings.shape}")

# %%
# Extract Split 1 indices (human train → mouse/fly zero-shot transfer)

train_idx = split_indices["split1_train"]
val_idx   = split_indices["split1_val"]
test_idx  = split_indices["split1_test"]

print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

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
# ---
# 
# ## 1.2 LoRA Adapter Fine-Tuning (NT, Evo-2, ESM2)

# %%
# Download model snapshot from Hugging Face Hub
from huggingface_hub import snapshot_download
local_dir = "./models/nucleotide-transformer-500m-human-ref"
if not Path(local_dir).exists():
    snapshot_download(repo_id="InstaDeepAI/nucleotide-transformer-500m-human-ref", local_dir=local_dir)
local_dir = "./models/esm2_t33_650M_UR50D"
if not Path(local_dir).exists():
    snapshot_download(repo_id="facebook/esm2_t33_650M_UR50D", local_dir=local_dir)

# %%
from dataclasses import dataclass
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = Path("data")

# Models to run
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
# ### 1.2.2 Evo-2 LoRA (CDS; environment-dependent - optional)

# %%
if RUN_EVO2_LORA:
    # NOTE: Evo2 requires a special installation and may not be available in all environments.
    # Uncomment and adapt the code below if Evo2 is available in your environment.
    print("=== Evo-2 LoRA Fine-Tuning ===")

    # from evo2 import Evo2  # hypothetical import; adjust to your installation
    # evo2_model = Evo2(EVO2_MODEL_NAME)

    # Hook into the specified layer to extract sequence-level embeddings
    # then apply LoRA and a classifier head similarly to NT above.

    # --- Your Evo-2 LoRA implementation here ---
    pass
else:
    print("Evo-2 LoRA skipped (RUN_EVO2_LORA=False)")

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
    esm2_base = AutoModel.from_pretrained(ESM2_MODEL_ID, enable_attentions=True)
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
# ---
# 
# ## 1.3 Frozen vs LoRA Comparison

# %%
# Paste or re-load your frozen-backbone metrics from Assignment 3 here
# Example structure (replace values with your actual results):
frozen_metrics = {
    "NT_frozen":   {"accuracy": 0.0, "precision": 0.0, "recall": 0.0,
                    "f1": 0.0, "roc_auc": 0.0, "prc_auc": 0.0},
    "ESM2_frozen": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0,
                    "f1": 0.0, "roc_auc": 0.0, "prc_auc": 0.0},
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
# **Discussion:** Summarize where LoRA helps most and whether it changes the precision–recall tradeoff.
# 
# > *Your answer here.*

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
        base_model = AutoModel.from_pretrained(ESM2_MODEL_ID, enable_attentions=False)
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
# **Discussion:** Is there a sweet spot between the number of parameters, runtime, and performance?
# 
# > *Your answer here.*

# %% [markdown]
# ---
# 
# # Interpretability & Applications
# 
# ## 2.1 Attribution Analysis: Understanding Model Decisions
# 
# **Learning Objective:** What drives your models' predictions? Do they learn biologically "meaningful" patterns? Here you will use a few different methodologies to try and identify what the model is learning.
# 
# **Why Attribution Matters:**
# - Opening the black box — What do foundation models learn?
# - If they actually learn useful biology, we should be able to trace residue level (nucleotide or amino acid) to results
# - Comparing multiple methods (gradients, attention, SHAP) validates findings
# - Biological validation: do high-attribution regions match known TF motifs (e.g. HLH, Zinc finger, etc.)?
# 
# ---
# 
# ### 2.1.1 Integrated Gradients for Attribution
# 
# **What is Integrated Gradients?**
# 
# Once again we use our favourite library, captum, to help us with this task. We integrate gradients along a path from a random sequence baseline to an input sequence to identify which positions in the sequence are most important for the model's prediction.

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
# **Discussion:** Do high-attribution regions correspond to any known domains? Are there species-specific attribution patterns?
# 
# > *Your answer here.*

# %% [markdown]
# ---
# 
# ### 2.1.2 Attention Weights Visualization
# 
# **What are Attention Weights?**
# 
# Transformer models (NT, Evo2, ESM2) use multi-head self-attention to weigh the importance of different positions. Attention weights show which tokens the model "attends to" when making predictions.

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
# **Discussion:** Are attention patterns sparse (focused on specific motifs) or diffuse? How does that compare with IG? Where do they agree or disagree?
# 
# > *Your answer here.*

# %% [markdown]
# ---
# 
# ### 2.1.3 Sparse Autoencoder (SAE) Feature Extraction
# 
# **What are Sparse Autoencoders?**
# 
# The recent breakthrough in mechanistic interpretability are Sparse Autoencoders (SAEs). They aim to decompose a model's latent dimension into interpretable sparse features. Each SAE feature ideally corresponds to a meaningful biological concept (e.g., a specific motif, structural domain, or sequence pattern).
# 
# SAEs project dense model activations into a **much higher** dimension "dictionary" using a linear encoder and a ReLU, paired with an L1 penalty to encourage monosemanticity.

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

# %%
# ── Analyze learned features ─────────────────────────────────────────────────
sae.eval()
with torch.no_grad():
    _, features_all = sae(acts_tensor)

features_np = features_all.cpu().numpy()  # (n_samples, dict_dim)
print(f"Feature matrix shape: {features_np.shape}")
print(f"Sparsity (fraction zero): {(features_np == 0).mean():.3f}")

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

# %%
# ── Compare SAE features to IG attributions ──────────────────────────────────
# For each of the SAE samples that also appear in our IG sample set,
# compare the most active SAE feature against the IG top window.

# (Conceptual comparison — adapt to your actual data structure)
print("SAE vs IG comparison (qualitative):")
for i in range(min(5, len(sae_labels))):
    top_sae_feat = np.argsort(features_np[i])[::-1][:5]
    label_str = "TF" if sae_labels[i] == 1 else "non-TF"
    print(f"  Sample {i} ({label_str}) — top SAE features: {top_sae_feat}")

# %% [markdown]
# **Discussion:** Which SAE features activate most strongly for TFs vs non-TFs? Do features correspond to known protein domains (e.g., DNA-binding domains, zinc fingers)? How do SAE features compare to the IG and attention attribution results from 2.1.1 and 2.1.2?
# 
# > *Your answer here.*


