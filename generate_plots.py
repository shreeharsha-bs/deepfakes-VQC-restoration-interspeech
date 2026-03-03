#!/usr/bin/env python3
"""
Generate all SVG plots for the GitHub Pages site.
Data is hardcoded from experiment results documented in EXPERIMENTS.md and parsed from log files.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path
import os

# ─── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

OUTDIR = Path(__file__).parent / "plots"
OUTDIR.mkdir(exist_ok=True)

# Colour palette (muted, accessible)
C = {
    "real": "#2E86AB",
    "real_conv": "#A23B72",
    "fake": "#F18F01",
    "fake_conv": "#C73E1D",
    "train": "#2E86AB",
    "val": "#C73E1D",
    "accent": "#3B1F2B",
    "muted": "#888888",
}
CLASS_COLORS = [C["real"], C["real_conv"], C["fake"], C["fake_conv"]]
CLASS_LABELS_4WAY = ["Real", "Real→Conv", "Fake", "Fake→Conv"]
CLASS_LABELS_BINARY = ["Bonafide", "Spoof"]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

# --- Training curves (from logs) ---
TRAINING_CURVES = {
    "DF-Arena Binary\n(MLAAD)": {
        "train_loss": [0.1566, 0.0819, 0.0701, 0.0609, 0.0571, 0.0523, 0.0511, 0.0369, 0.0358, 0.0350],
        "val_loss":   [0.0493, 0.0355, 0.0326, 0.0450, 0.0346, 0.0770, 0.0497, 0.0377, 0.0430, 0.0293],
        "train_acc":  [0.9399, 0.9703, 0.9748, 0.9780, 0.9789, 0.9807, 0.9817, 0.9875, 0.9874, 0.9870],
        "val_acc":    [0.9843, 0.9882, 0.9902, 0.9856, 0.9908, 0.9649, 0.9793, 0.9898, 0.9836, 0.9915],
    },
    "DF-Arena Binary\n(MLAAD+ASV)": {
        "train_loss": [0.3397, 0.2449, 0.2242, 0.2030, 0.1961, 0.1837, 0.1716, 0.1659, 0.1537, 0.1332, 0.1211, 0.1146, 0.1071],
        "val_loss":   [0.4074, 0.2895, 0.2798, 0.2918, 0.1850, 0.2514, 0.3977, 0.2782, 0.1911, 0.1971, 0.2188, 0.2309, 0.2562],
        "train_acc":  [0.8423, 0.8874, 0.8967, 0.9072, 0.9108, 0.9166, 0.9221, 0.9259, 0.9331, 0.9434, 0.9475, 0.9517, 0.9529],
        "val_acc":    [0.8075, 0.8620, 0.8644, 0.8654, 0.9147, 0.8887, 0.8533, 0.8914, 0.9202, 0.9213, 0.9091, 0.9070, 0.9104],
    },
    "DF-Arena Binary\n(+Sidon)": {
        "train_loss": [0.1714, 0.1580, 0.1507, 0.1435, 0.1395, 0.1342, 0.1250, 0.1197, 0.1131, 0.1107, 0.0965, 0.0968, 0.0929, 0.0898],
        "val_loss":   [0.2618, 0.2492, 0.2740, 0.3304, 0.2394, 0.2155, 0.2284, 0.2500, 0.4440, 0.3216, 0.2164, 0.3674, 0.3813, 0.4763],
        "train_acc":  [0.9237, 0.9304, 0.9321, 0.9380, 0.9399, 0.9413, 0.9455, 0.9472, 0.9505, 0.9520, 0.9595, 0.9577, 0.9598, 0.9620],
        "val_acc":    [0.8788, 0.9029, 0.8891, 0.8738, 0.9054, 0.9115, 0.9029, 0.9062, 0.8716, 0.9087, 0.9229, 0.9029, 0.8880, 0.8777],
    },
    "DF-Arena 4-Way\n(+Sidon)": {
        "train_loss": [0.1730, 0.1349, 0.1246, 0.1148, 0.1106, 0.0952, 0.0942, 0.0937, 0.0912, 0.0846, 0.0832, 0.0828, 0.0843, 0.0752, 0.0739, 0.0722, 0.0748, 0.0735, 0.0667, 0.0692],
        "val_loss":   [0.2685, 0.3077, 0.3237, 0.3134, 0.5800, 0.2860, 0.3040, 0.3073, 0.2372, 0.2513, 0.3306, 0.2557, 0.3514, 0.1974, 0.2742, 0.2999, 0.3095, 0.3500, 0.2905, 0.3089],
        "train_acc":  [0.9308, 0.9455, 0.9490, 0.9525, 0.9549, 0.9612, 0.9612, 0.9620, 0.9614, 0.9648, 0.9650, 0.9660, 0.9649, 0.9685, 0.9691, 0.9701, 0.9686, 0.9691, 0.9725, 0.9718],
        "val_acc":    [0.8921, 0.8813, 0.8779, 0.8854, 0.8160, 0.8936, 0.8974, 0.8977, 0.9162, 0.9092, 0.8980, 0.9108, 0.8855, 0.9259, 0.9137, 0.9068, 0.8977, 0.8955, 0.9139, 0.9029],
    },
}

# --- Confusion matrices ---
# (row = true, col = predicted), row-major

CM_DATA = {
    "MLP 4-Way — MLAAD Seen": {
        "matrix": np.array([
            [1365, 0, 0, 0],
            [0, 1365, 0, 0],
            [0, 0, 1365, 0],
            [0, 55, 0, 1310]
        ]),
        "labels": CLASS_LABELS_4WAY,
        "acc": 99.0,
    },
    "MLP 4-Way — MLAAD Unseen (OuteTTS)": {
        "matrix": np.array([
            [257, 0, 0, 0],
            [0, 255, 0, 2],
            [0, 0, 257, 0],
            [0, 105, 0, 152]
        ]),
        "labels": CLASS_LABELS_4WAY,
        "acc": 89.6,
    },
    "MLP 4-Way — ASVspoof5": {
        "matrix": np.array([
            [480, 20, 760, 740],
            [20, 20, 360, 1600],
            [200, 0, 1080, 720],
            [20, 0, 80, 1900]
        ]),
        "labels": CLASS_LABELS_4WAY,
        "acc": 43.5,
    },
    "DF-Arena 4-Way — MLAAD Seen": {
        "matrix": np.array([
            [1365, 0, 0, 0],
            [0, 1338, 0, 27],
            [0, 0, 1365, 0],
            [0, 31, 0, 1334]
        ]),
        "labels": CLASS_LABELS_4WAY,
        "acc": 98.9,
    },
    "DF-Arena 4-Way — MLAAD Unseen": {
        "matrix": np.array([
            [257, 0, 0, 0],
            [0, 251, 0, 6],
            [0, 0, 257, 0],
            [0, 14, 0, 243]
        ]),
        "labels": CLASS_LABELS_4WAY,
        "acc": 98.1,
    },
    "DF-Arena 4-Way — ASVspoof5": {
        "matrix": np.array([
            [1, 0, 1993, 6],
            [0, 0, 312, 1688],
            [0, 0, 1960, 40],
            [3, 0, 0, 1997]
        ]),
        "labels": CLASS_LABELS_4WAY,
        "acc": 49.4,
    },
    "DF-Arena 4-Way Mixed — ASVspoof5": {
        "matrix": np.array([
            [1894, 0, 106, 0],
            [0, 1316, 0, 684],
            [34, 0, 1952, 14],
            [0, 224, 0, 1776]
        ]),
        "labels": CLASS_LABELS_4WAY,
        "acc": 86.8,
    },
    "DF-Arena 4-Way+Sidon — ASVspoof5": {
        "matrix": np.array([
            [1900, 10, 82, 8],
            [0, 1316, 6, 678],
            [22, 0, 1976, 2],
            [0, 204, 0, 1796]
        ]),
        "labels": CLASS_LABELS_4WAY,
        "acc": 87.2,
    },
    "DF-Arena Binary — MLAAD Seen": {
        "matrix": np.array([
            [2703, 27],
            [60, 2670]
        ]),
        "labels": CLASS_LABELS_BINARY,
        "acc": 98.4,
    },
    "DF-Arena Binary Mixed — ASVspoof5": {
        "matrix": np.array([
            [2935, 1065],
            [239, 3761]
        ]),
        "labels": CLASS_LABELS_BINARY,
        "acc": 83.7,
    },
    "DF-Arena Binary+Sidon — Sidon Test": {
        "matrix": np.array([
            [364, 93],
            [0, 462]
        ]),
        "labels": CLASS_LABELS_BINARY,
        "acc": 89.9,
    },
}



# --- EER comparison (key finding) ---
EER_SOURCE_VS_MEDIATION = {
    "DF-Arena 4-Way\n(MLAAD only)": {"source": 46.3, "mediation": 0.15},
    "DF-Arena 4-Way\n(Mixed)":      {"source": 11.8, "mediation": 0.03},
    "DF-Arena 4-Way\n(+Sidon)":     {"source": 11.2, "mediation": 0.05},
    "DF-Arena Binary\n(Pretrained)": {"source": 44.2, "mediation": None},
    "DF-Arena Binary\n(MLAAD)":      {"source": 1.43, "mediation": None},
    "DF-Arena Binary\n(Mixed)":      {"source": 13.3, "mediation": None},
    "DF-Arena Binary\n(+Sidon)":     {"source": 14.3, "mediation": None},
    "MLP Binary":                    {"source": 54.7, "mediation": None},
}

# --- Benign transform (Sidon) flip rates ---
SIDON_FLIPS = {
    "DF-Arena\nBinary": {"real_flip": 79.5, "fake_flip": 3.2},
    "MLP\n4-Way\n(source)": {"real_flip": 24.0, "fake_flip": 1.7},
    "MLP\nBinary": {"real_flip": 10.0, "fake_flip": 1.9},
    "MLP\nUnconverted": {"real_flip": 37.0, "fake_flip": 0.6},
}

# --- Subset classifiers ---
SUBSET_ACC = {
    "Unconverted": {"seen": 100.0, "unseen": 99.6},
    "Modal Conv.": {"seen": 97.9, "unseen": 76.4},
    "Breathy":     {"seen": 97.1, "unseen": 81.9},
    "Creaky":      {"seen": 98.5, "unseen": 84.9},
}

# --- Cross-domain progression (ASVspoof5) ---
PROGRESSION = {
    "metric": ["Overall Acc.", "Real→Correct", "Source EER", "Mediation Acc.", "Mediation EER"],
    "Binary\n(pretrained)": [94.9, 69.9, 44.1, None, None],
    "4-Way\n(MLAAD)":       [49.4, 0.1, 46.3, 99.5, 0.15],
    "4-Way\n(Mixed)":       [86.8, 94.7, 11.8, 100.0, 0.03],
    "4-Way\n(+Sidon)":      [87.2, 95.0, 11.2, 100.0, 0.05],
}


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(ax, cm, labels, title, acc=None, cmap="Blues", normalize=True):
    """Plot a single confusion matrix."""
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm.astype(float) / row_sums
    else:
        cm_norm = cm.astype(float)
    
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    n = len(labels)
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            count = cm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.0%}\n({count})",
                    ha="center", va="center", fontsize=8, color=color)
    
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ttl = title
    if acc is not None:
        ttl += f"  (Acc: {acc:.1f}%)"
    ax.set_title(ttl, fontsize=11, fontweight="bold")
    return im


def save(fig, name):
    path = OUTDIR / f"{name}.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ─── 1. Training curves ────────────────────────────────────────────────────────
def gen_training_curves():
    print("Generating training curves…")
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    
    for idx, (name, data) in enumerate(TRAINING_CURVES.items()):
        epochs = range(1, len(data["train_loss"]) + 1)
        
        # Loss
        ax = axes[0, idx]
        ax.plot(epochs, data["train_loss"], "-o", color=C["train"], label="Train", markersize=3, linewidth=1.5)
        ax.plot(epochs, data["val_loss"], "-s", color=C["val"], label="Val", markersize=3, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.legend(frameon=False, fontsize=8)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        
        # Accuracy
        ax = axes[1, idx]
        ax.plot(epochs, data["train_acc"], "-o", color=C["train"], label="Train", markersize=3, linewidth=1.5)
        ax.plot(epochs, data["val_acc"], "-s", color=C["val"], label="Val", markersize=3, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.78, 1.01)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(frameon=False, fontsize=8)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    axes[0, 0].set_ylabel("Cross-Entropy Loss")
    axes[1, 0].set_ylabel("Accuracy")
    fig.suptitle("Training & Validation Curves — DF-Arena Fine-Tuning Experiments", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "training_curves")


# ─── 2. Confusion matrices ─────────────────────────────────────────────────────
def gen_confusion_matrices():
    print("Generating confusion matrices…")
    
    # Group 1: MLP 4-Way (Seen / Unseen / ASVspoof5) 
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    keys = [
        "MLP 4-Way — MLAAD Seen",
        "MLP 4-Way — MLAAD Unseen (OuteTTS)",
        "MLP 4-Way — ASVspoof5",
    ]
    for ax, key in zip(axes, keys):
        d = CM_DATA[key]
        plot_confusion_matrix(ax, d["matrix"], d["labels"], key, d["acc"])
    fig.suptitle("MLP 4-Way Intervention Depth Classifier — Confusion Matrices", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "cm_mlp_4way")
    
    # Group 2: DF-Arena 4-Way (MLAAD Seen / Unseen / ASVspoof5)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    keys = [
        "DF-Arena 4-Way — MLAAD Seen",
        "DF-Arena 4-Way — MLAAD Unseen",
        "DF-Arena 4-Way — ASVspoof5",
    ]
    for ax, key in zip(axes, keys):
        d = CM_DATA[key]
        plot_confusion_matrix(ax, d["matrix"], d["labels"], key, d["acc"])
    fig.suptitle("DF-Arena 4-Way (MLAAD-only) — Confusion Matrices", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "cm_dfarena_4way")
    
    # Group 3: DF-Arena 4-Way progression on ASVspoof5
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    keys = [
        "DF-Arena 4-Way — ASVspoof5",
        "DF-Arena 4-Way Mixed — ASVspoof5",
        "DF-Arena 4-Way+Sidon — ASVspoof5",
    ]
    for ax, key in zip(axes, keys):
        d = CM_DATA[key]
        plot_confusion_matrix(ax, d["matrix"], d["labels"], key, d["acc"])
    fig.suptitle("DF-Arena 4-Way on ASVspoof5 — Training Data Progression", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "cm_dfarena_4way_progression")
    
    # Group 4: DF-Arena Binary variants
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    keys = [
        "DF-Arena Binary — MLAAD Seen",
        "DF-Arena Binary Mixed — ASVspoof5",
        "DF-Arena Binary+Sidon — Sidon Test",
    ]
    for ax, key in zip(axes, keys):
        d = CM_DATA[key]
        plot_confusion_matrix(ax, d["matrix"], d["labels"], key, d["acc"])
    fig.suptitle("DF-Arena Binary — Confusion Matrices", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "cm_dfarena_binary")





# ─── 4. Source vs Mediation EER ─────────────────────────────────────────────────
def gen_eer_comparison():
    print("Generating EER comparison (source vs mediation)…")
    
    # 4-way models with both source and mediation EER
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Panel A: Source vs Mediation EER for 4-way models on ASVspoof5
    models_4way = ["DF-Arena 4-Way\n(MLAAD only)", "DF-Arena 4-Way\n(Mixed)", "DF-Arena 4-Way\n(+Sidon)"]
    source_eers = [EER_SOURCE_VS_MEDIATION[m]["source"] for m in models_4way]
    mediation_eers = [EER_SOURCE_VS_MEDIATION[m]["mediation"] for m in models_4way]
    
    x = np.arange(len(models_4way))
    w = 0.35
    bars1 = axes[0].bar(x - w/2, source_eers, w, label="Source EER", color=C["fake_conv"], alpha=0.85)
    bars2 = axes[0].bar(x + w/2, mediation_eers, w, label="Mediation EER", color=C["real"], alpha=0.85)
    
    for bar, val in zip(bars1, source_eers):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, mediation_eers):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models_4way, fontsize=9)
    axes[0].set_ylabel("EER (%)")
    axes[0].set_title("Source vs. Mediation EER on ASVspoof5\n(same model, same predictions)", fontsize=11, fontweight="bold")
    axes[0].legend(frameon=False, fontsize=10)
    axes[0].set_ylim(0, 55)
    axes[0].axhline(50, color=C["muted"], linestyle="--", linewidth=0.8, label="Chance")
    
    # Panel B: All experiments source EER comparison
    all_exps = list(EER_SOURCE_VS_MEDIATION.keys())
    all_source = [EER_SOURCE_VS_MEDIATION[e]["source"] for e in all_exps]
    colors = [C["fake_conv"] if EER_SOURCE_VS_MEDIATION[e]["mediation"] is not None else C["fake"] for e in all_exps]
    
    x2 = np.arange(len(all_exps))
    bars = axes[1].barh(x2, all_source, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, all_source):
        axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", ha="left", va="center", fontsize=9)
    
    axes[1].set_yticks(x2)
    axes[1].set_yticklabels(all_exps, fontsize=8)
    axes[1].set_xlabel("Source EER (%) on ASVspoof5")
    axes[1].set_title("Cross-Domain Source Detection EER\n(lower = better real/fake separation)", fontsize=11, fontweight="bold")
    axes[1].axvline(50, color=C["muted"], linestyle="--", linewidth=0.8)
    axes[1].set_xlim(0, 60)
    axes[1].invert_yaxis()
    
    fig.tight_layout()
    save(fig, "eer_comparison")


# ─── 5. Benign transform comparison ────────────────────────────────────────────
def gen_sidon_flips():
    print("Generating Sidon benign transform comparison…")
    fig, ax = plt.subplots(figsize=(9, 5))
    
    detectors = list(SIDON_FLIPS.keys())
    real_flips = [SIDON_FLIPS[d]["real_flip"] for d in detectors]
    fake_flips = [SIDON_FLIPS[d]["fake_flip"] for d in detectors]
    
    x = np.arange(len(detectors))
    w = 0.35
    bars1 = ax.bar(x - w/2, real_flips, w, label="Enhanced Real → Fake", color=C["real"], alpha=0.85)
    bars2 = ax.bar(x + w/2, fake_flips, w, label="Enhanced Fake flip", color=C["fake"], alpha=0.85)
    
    for bar, val in zip(bars1, real_flips):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, val in zip(bars2, fake_flips):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    ax.set_xticks(x)
    ax.set_xticklabels(detectors, fontsize=9)
    ax.set_ylabel("Prediction Flip Rate (%)")
    ax.set_title("Sidon Speech Enhancement — Prediction Flips by Detector\n(Non-adversarial transform causes 79.5% of real speech to be labeled fake by DF-Arena)",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    ax.set_ylim(0, 95)
    fig.tight_layout()
    save(fig, "sidon_flips")


# ─── 6. Subset classifier comparison ───────────────────────────────────────────
def gen_subset_comparison():
    print("Generating subset classifier comparison…")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    subsets = list(SUBSET_ACC.keys())
    seen = [SUBSET_ACC[s]["seen"] for s in subsets]
    unseen = [SUBSET_ACC[s]["unseen"] for s in subsets]
    
    x = np.arange(len(subsets))
    w = 0.35
    bars1 = ax.bar(x - w/2, seen, w, label="Seen architectures", color=C["real"], alpha=0.85)
    bars2 = ax.bar(x + w/2, unseen, w, label="Unseen (OuteTTS)", color=C["fake_conv"], alpha=0.85)
    
    for bar, val in zip(bars1, seen):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, unseen):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    ax.set_xticks(x)
    ax.set_xticklabels(subsets, fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Subset Binary Classifiers — Seen vs. Unseen Accuracy\n(Voice-converted subsets generalize poorly to unseen TTS)",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    ax.set_ylim(70, 105)
    fig.tight_layout()
    save(fig, "subset_classifiers")


# ─── 7. Cross-domain progression ───────────────────────────────────────────────
def gen_progression():
    print("Generating cross-domain progression…")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    metrics = PROGRESSION["metric"]
    model_keys = [k for k in PROGRESSION.keys() if k != "metric"]
    colors_prog = [C["muted"], C["fake_conv"], C["real"], C["real_conv"]]
    
    # Panel A: Accuracy-like metrics (higher = better)
    acc_metrics = [0, 1, 3]  # Overall Acc, Real→Correct, Mediation Acc
    x = np.arange(len(acc_metrics))
    w = 0.18
    
    for i, mk in enumerate(model_keys):
        vals = [PROGRESSION[mk][j] if PROGRESSION[mk][j] is not None else 0 for j in acc_metrics]
        offset = (i - (len(model_keys) - 1) / 2) * w
        bars = axes[0].bar(x + offset, vals, w, label=mk, color=colors_prog[i], alpha=0.85)
        for bar, val, raw in zip(bars, vals, [PROGRESSION[mk][j] for j in acc_metrics]):
            if raw is not None:
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f"{val:.1f}", ha="center", va="bottom", fontsize=7.5)
    
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([metrics[j] for j in acc_metrics], fontsize=9)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Cross-Domain Progression — Accuracy Metrics\n(ASVspoof5, all models)", fontsize=11, fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[0].set_ylim(0, 110)
    
    # Panel B: EER metrics (lower = better)
    eer_metrics = [2, 4]  # Source EER, Mediation EER
    x2 = np.arange(len(eer_metrics))
    
    for i, mk in enumerate(model_keys):
        vals = [PROGRESSION[mk][j] if PROGRESSION[mk][j] is not None else 0 for j in eer_metrics]
        offset = (i - (len(model_keys) - 1) / 2) * w
        bars = axes[1].bar(x2 + offset, vals, w, label=mk, color=colors_prog[i], alpha=0.85)
        for bar, val, raw in zip(bars, vals, [PROGRESSION[mk][j] for j in eer_metrics]):
            if raw is not None and val > 0.5:
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                            f"{val:.1f}" if val > 1 else f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)
    
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([metrics[j] for j in eer_metrics], fontsize=9)
    axes[1].set_ylabel("EER (%)")
    axes[1].set_title("Cross-Domain Progression — EER Metrics\n(lower is better)", fontsize=11, fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].set_ylim(0, 55)
    axes[1].axhline(50, color=C["muted"], linestyle="--", linewidth=0.8, alpha=0.5)
    
    fig.tight_layout()
    save(fig, "progression")


# ─── 8. Mediation vs Source decomposition summary ──────────────────────────────
def gen_decomposition_summary():
    """The hero figure: showing the 300x EER gap."""
    print("Generating decomposition summary…")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    categories = ["Source Axis\n(Real vs. Fake)", "Mediation Axis\n(Processed vs. Unprocessed)"]
    eers = [46.3, 0.15]
    accs = [49.9, 99.5]
    
    x = np.arange(2)
    w = 0.35
    
    bars1 = ax.bar(x - w/2, eers, w, label="EER (%)", color=[C["fake_conv"], C["real"]], alpha=0.85)
    bars2 = ax.bar(x + w/2, accs, w, label="Accuracy (%)", color=[C["fake_conv"], C["real"]], alpha=0.5,
                   hatch="//", edgecolor="white")
    
    ax.text(bars1[0].get_x() + bars1[0].get_width()/2, bars1[0].get_height() + 1,
            "46.3%", ha="center", va="bottom", fontsize=14, fontweight="bold", color=C["fake_conv"])
    ax.text(bars1[1].get_x() + bars1[1].get_width()/2, bars1[1].get_height() + 1,
            "0.15%", ha="center", va="bottom", fontsize=14, fontweight="bold", color=C["real"])
    ax.text(bars2[0].get_x() + bars2[0].get_width()/2, bars2[0].get_height() + 1,
            "49.9%", ha="center", va="bottom", fontsize=11, color=C["fake_conv"])
    ax.text(bars2[1].get_x() + bars2[1].get_width()/2, bars2[1].get_height() + 1,
            "99.5%", ha="center", va="bottom", fontsize=11, color=C["real"])
    
    # Add the 300x annotation
    ax.annotate("", xy=(0.82, 46.3), xytext=(0.82, 0.15),
               arrowprops=dict(arrowstyle="<->", color=C["accent"], lw=2))
    ax.text(1.1, 23, "300×\nEER gap", fontsize=13, fontweight="bold",
            color=C["accent"], ha="center", va="center")
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("The Core Finding: Same Model, Same Data, Different Collapse\n"
                 "DF-Arena 4-Way on ASVspoof5 — Mediation generalizes, source does not",
                 fontsize=12, fontweight="bold")
    ax.legend(["EER", "Accuracy"], frameon=False, fontsize=10, loc="upper left")
    ax.set_ylim(0, 110)
    ax.axhline(50, color=C["muted"], linestyle="--", linewidth=0.8, alpha=0.5)
    
    fig.tight_layout()
    save(fig, "decomposition_hero")


# ─── 9. Processing spectrum ────────────────────────────────────────────────────
def gen_processing_spectrum():
    """Detector sensitivity across the spectrum of generative processing."""
    print("Generating processing spectrum…")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    
    transforms = ["None\n(Original)", "Speech Enh.\n(Sidon)", "Voice Conv.\n(Modal)", "TTS\n(Synthesis)"]
    dfarena = [0, 79.5, 100, 99]
    mlp_binary = [0, 10.0, 0, 0]  # binary treats VC real as real
    
    x = np.arange(len(transforms))
    w = 0.35
    
    ax.bar(x - w/2, dfarena, w, label="DF-Arena Binary (→ Fake %)", color=C["fake_conv"], alpha=0.85)
    ax.bar(x + w/2, mlp_binary, w, label="MLP Binary (→ Fake %)", color=C["real"], alpha=0.85)
    
    for i, (d, m) in enumerate(zip(dfarena, mlp_binary)):
        ax.text(x[i] - w/2, d + 1.5, f"{d}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(x[i] + w/2, m + 1.5, f"{m}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    # Gradient arrow
    ax.annotate("", xy=(3.5, -8), xytext=(-0.5, -8),
               arrowprops=dict(arrowstyle="->", color=C["accent"], lw=2),
               annotation_clip=False)
    ax.text(1.5, -12, "← Less processing                More processing →",
            ha="center", fontsize=9, color=C["accent"],
            transform=ax.transData, clip_on=False)
    
    ax.set_xticks(x)
    ax.set_xticklabels(transforms, fontsize=10)
    ax.set_ylabel("% of Real Speech Labeled as Fake")
    ax.set_title("The Processing Spectrum: Detector Sensitivity to Generative Processing\n"
                 "(Applied to genuine human speech — all labels should be \"Real\")",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    ax.set_ylim(-15, 115)
    
    fig.tight_layout()
    save(fig, "processing_spectrum")


# ─── 10. Dataset overview ──────────────────────────────────────────────────────
def gen_dataset_overview():
    """Visual overview of dataset composition."""
    print("Generating dataset overview…")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # MLAAD
    mlaad_labels = ["Real\n(69,855)", "Real→Conv\n(~18,544)", "Fake\n(~9,009)", "Fake→Conv\n(~18,544)"]
    mlaad_sizes = [69855, 18544, 9009, 18544]
    axes[0].pie(mlaad_sizes, labels=mlaad_labels, colors=CLASS_COLORS,
                autopct="%1.0f%%", startangle=140, textprops={"fontsize": 9})
    axes[0].set_title("MLAAD Dataset Composition", fontsize=11, fontweight="bold")
    
    # ASVspoof5
    asv_labels = ["Real\n(14,171)", "Real→Conv\n(56,684)", "Fake\n(54,017)", "Fake→Conv\n(78,601)"]
    asv_sizes = [14171, 56684, 54017, 78601]
    axes[1].pie(asv_sizes, labels=asv_labels, colors=CLASS_COLORS,
                autopct="%1.0f%%", startangle=140, textprops={"fontsize": 9})
    axes[1].set_title("ASVspoof5 Dataset Composition", fontsize=11, fontweight="bold")
    
    fig.suptitle("Training & Evaluation Datasets", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "dataset_overview")


# ─── 11. Comprehensive EER heatmap ─────────────────────────────────────────────
def gen_eer_heatmap():
    """Heatmap of EER across all experiments and datasets."""
    print("Generating EER heatmap…")
    
    experiments = [
        "MLP Binary", "MLP 4-Way\n(source)",
        "DF-Arena\nBinary (pre)", "DF-Arena\nBinary (ft)",
        "DF-Arena\nBinary (mix)", "DF-Arena\nBinary (+Sid)",
        "DF-Arena\n4-Way (med)", "DF-Arena\n4-Way mix (med)", "DF-Arena\n4-Way +Sid (med)",
    ]
    
    datasets = ["MLAAD\nSeen", "MLAAD\nUnseen", "ASVspoof5"]
    
    # EER values (NaN = not available)
    data = np.array([
        [np.nan, np.nan, 54.7],      # MLP Binary
        [np.nan, np.nan, 42.3],      # MLP 4-Way source
        [0.05, np.nan, 44.2],        # DF-Arena Binary pretrained
        [1.43, np.nan, np.nan],      # DF-Arena Binary ft
        [1.36, 2.33, 13.3],         # DF-Arena Binary mix
        [1.43, 2.14, 14.3],         # DF-Arena Binary +Sidon
        [np.nan, np.nan, 0.15],      # DF-Arena 4-Way mediation
        [0.00, 0.00, 0.03],         # DF-Arena 4-Way mix mediation
        [0.00, 0.00, 0.05],         # DF-Arena 4-Way +Sid mediation
    ])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap="RdYlGn_r", vmin=0, vmax=55, aspect="auto")
    
    for i in range(len(experiments)):
        for j in range(len(datasets)):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if val > 30 else "black"
                ax.text(j, i, f"{val:.1f}%" if val >= 1 else f"{val:.2f}%",
                        ha="center", va="center", fontsize=9, color=color, fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color=C["muted"])
    
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_yticks(range(len(experiments)))
    ax.set_yticklabels(experiments, fontsize=9)
    ax.set_title("EER (%) Across All Experiments and Datasets\n(Green = good, Red = poor)", fontsize=12, fontweight="bold")
    
    plt.colorbar(im, ax=ax, label="EER (%)", shrink=0.8)
    
    # Separator line between source and mediation
    ax.axhline(5.5, color="black", linewidth=2)
    ax.text(-0.7, 4, "Source\nDetection", fontsize=8, va="center", ha="right", fontweight="bold", color=C["fake_conv"])
    ax.text(-0.7, 7, "Mediation\nDetection", fontsize=8, va="center", ha="right", fontweight="bold", color=C["real"])
    
    fig.tight_layout()
    save(fig, "eer_heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Output directory: {OUTDIR}")
    gen_training_curves()
    gen_confusion_matrices()
    gen_eer_comparison()
    gen_sidon_flips()
    gen_subset_comparison()
    gen_progression()
    gen_decomposition_summary()
    gen_processing_spectrum()
    gen_dataset_overview()
    gen_eer_heatmap()
    print(f"\nDone! Generated {len(list(OUTDIR.glob('*.svg')))} SVGs in {OUTDIR}")
