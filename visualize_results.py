"""
Chicken Disease Classification - Model Evaluation & Visualization
Generates: Confusion Matrix, Classification Report, ROC Curve, 
Accuracy/Precision/Recall/F1 bar chart, and per-class metrics.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
from pathlib import Path
import json

# ── Configuration ──────────────────────────────────────────────
MODEL_PATH = Path("artifacts/training/model.h5")
DATA_DIR = Path("artifacts/data_ingestion/Chicken-fecal-images")
OUTPUT_DIR = Path("artifacts/evaluation_results")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
CLASS_NAMES = ["Coccidiosis", "Healthy"]

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Color Palette ──────────────────────────────────────────────
COLORS = {
    'bg_dark': '#0f0f1a',
    'bg_card': '#1a1a2e',
    'accent1': '#e94560',
    'accent2': '#0f3460',
    'accent3': '#16213e',
    'gold': '#f0c040',
    'teal': '#00d2ff',
    'green': '#00e676',
    'purple': '#7c4dff',
    'orange': '#ff6e40',
    'text': '#e0e0e0',
    'text_dim': '#888899',
}

# ── Setup matplotlib style ────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': COLORS['bg_dark'],
    'axes.facecolor': COLORS['bg_card'],
    'axes.edgecolor': COLORS['text_dim'],
    'axes.labelcolor': COLORS['text'],
    'text.color': COLORS['text'],
    'xtick.color': COLORS['text_dim'],
    'ytick.color': COLORS['text_dim'],
    'font.family': 'sans-serif',
    'font.size': 11,
})


def load_and_predict():
    """Load model and generate predictions on test set."""
    print("Loading model...")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    
    # Use the pre-split test directory
    test_dir = DATA_DIR / "test"
    print(f"Preparing test data from: {test_dir}")
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )
    
    val_generator = datagen.flow_from_directory(
        directory=str(test_dir),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        interpolation="bilinear",
        shuffle=False,
        class_mode='categorical'
    )
    
    print(f"Validation samples: {val_generator.samples}")
    print(f"Class indices: {val_generator.class_indices}")
    
    # Get predictions
    y_pred_probs = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_generator.classes
    
    return y_true, y_pred, y_pred_probs, val_generator


def compute_metrics(y_true, y_pred):
    """Compute all classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'precision_per_class': precision_score(y_true, y_pred, average=None).tolist(),
        'recall_per_class': recall_score(y_true, y_pred, average=None).tolist(),
        'f1_per_class': f1_score(y_true, y_pred, average=None).tolist(),
    }
    
    print("\n" + "="*50)
    print("  CLASSIFICATION METRICS")
    print("="*50)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print("="*50)
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    return metrics


def plot_confusion_matrix(y_true, y_pred):
    """Plot a stunning confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(COLORS['bg_dark'])
    
    # Custom colormap
    cmap = sns.color_palette([COLORS['bg_card'], COLORS['accent2'], COLORS['teal']], as_cmap=True)
    
    sns.heatmap(
        cm, annot=False, cmap=cmap, ax=ax,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=2, linecolor=COLORS['bg_dark'],
        cbar_kws={'label': 'Count', 'shrink': 0.8}
    )
    
    # Add custom annotations with count + percentage
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            count = cm[i, j]
            pct = cm_normalized[i, j] * 100
            color = COLORS['text'] if cm_normalized[i, j] < 0.7 else COLORS['bg_dark']
            ax.text(j + 0.5, i + 0.35, f"{count}",
                    ha='center', va='center', fontsize=28, fontweight='bold', color=color)
            ax.text(j + 0.5, i + 0.65, f"({pct:.1f}%)",
                    ha='center', va='center', fontsize=13, color=color, alpha=0.85)
    
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_title('Confusion Matrix', fontsize=18, fontweight='bold', pad=20, color=COLORS['teal'])
    ax.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(str(path), dpi=200, bbox_inches='tight', facecolor=COLORS['bg_dark'])
    plt.close(fig)
    print(f"Saved: {path}")
    return cm


def plot_metrics_bar(metrics):
    """Plot overall metrics as a stylish bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS['bg_dark'])
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
    colors = [COLORS['teal'], COLORS['green'], COLORS['purple'], COLORS['gold']]
    
    bars = ax.bar(metric_names, metric_values, color=colors, width=0.55,
                  edgecolor=[c + '88' for c in colors], linewidth=1.5, zorder=3)
    
    # Add value labels on bars
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f'{val:.2%}', ha='center', va='bottom', fontsize=15,
                fontweight='bold', color=COLORS['text'])
    
    # Add glow effect
    for bar, color in zip(bars, colors):
        ax.bar(bar.get_x() + bar.get_width()/2, bar.get_height(),
               width=bar.get_width() * 1.15, alpha=0.08, color=color, zorder=2)
    
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Overall Classification Metrics', fontsize=18, fontweight='bold',
                 pad=20, color=COLORS['teal'])
    ax.tick_params(axis='x', labelsize=13)
    ax.grid(axis='y', alpha=0.15, color=COLORS['text_dim'])
    ax.axhline(y=1.0, color=COLORS['accent1'], linestyle='--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    path = OUTPUT_DIR / "metrics_overview.png"
    fig.savefig(str(path), dpi=200, bbox_inches='tight', facecolor=COLORS['bg_dark'])
    plt.close(fig)
    print(f"Saved: {path}")


def plot_per_class_metrics(metrics):
    """Plot per-class precision, recall, f1 as grouped bars."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS['bg_dark'])
    
    x = np.arange(len(CLASS_NAMES))
    width = 0.22
    
    prec = metrics['precision_per_class']
    rec = metrics['recall_per_class']
    f1 = metrics['f1_per_class']
    
    bars1 = ax.bar(x - width, prec, width, label='Precision', color=COLORS['teal'],
                   edgecolor=COLORS['teal'] + '88', linewidth=1.2, zorder=3)
    bars2 = ax.bar(x, rec, width, label='Recall', color=COLORS['purple'],
                   edgecolor=COLORS['purple'] + '88', linewidth=1.2, zorder=3)
    bars3 = ax.bar(x + width, f1, width, label='F1 Score', color=COLORS['gold'],
                   edgecolor=COLORS['gold'] + '88', linewidth=1.2, zorder=3)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                    f'{bar.get_height():.2f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold', color=COLORS['text'])
    
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=13)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Per-Class Metrics', fontsize=18, fontweight='bold',
                 pad=20, color=COLORS['teal'])
    ax.legend(loc='upper right', fontsize=11, facecolor=COLORS['bg_card'],
              edgecolor=COLORS['text_dim'], labelcolor=COLORS['text'])
    ax.grid(axis='y', alpha=0.15, color=COLORS['text_dim'])
    
    plt.tight_layout()
    path = OUTPUT_DIR / "per_class_metrics.png"
    fig.savefig(str(path), dpi=200, bbox_inches='tight', facecolor=COLORS['bg_dark'])
    plt.close(fig)
    print(f"Saved: {path}")


def plot_roc_curve(y_true, y_pred_probs):
    """Plot ROC curve for each class."""
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(COLORS['bg_dark'])
    
    colors_roc = [COLORS['teal'], COLORS['accent1']]
    
    for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors_roc)):
        y_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f'{cls_name} (AUC = {roc_auc:.3f})')
        ax.fill_between(fpr, tpr, alpha=0.08, color=color)
    
    ax.plot([0, 1], [0, 1], 'w--', alpha=0.25, linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title('ROC Curve', fontsize=18, fontweight='bold', pad=20, color=COLORS['teal'])
    ax.legend(loc='lower right', fontsize=12, facecolor=COLORS['bg_card'],
              edgecolor=COLORS['text_dim'], labelcolor=COLORS['text'])
    ax.grid(alpha=0.15, color=COLORS['text_dim'])
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    
    plt.tight_layout()
    path = OUTPUT_DIR / "roc_curve.png"
    fig.savefig(str(path), dpi=200, bbox_inches='tight', facecolor=COLORS['bg_dark'])
    plt.close(fig)
    print(f"Saved: {path}")


def plot_prediction_distribution(y_pred_probs):
    """Plot prediction confidence distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(COLORS['bg_dark'])
    
    colors_cls = [COLORS['accent1'], COLORS['teal']]
    
    for i, (ax, cls_name, color) in enumerate(zip(axes, CLASS_NAMES, colors_cls)):
        ax.set_facecolor(COLORS['bg_card'])
        ax.hist(y_pred_probs[:, i], bins=25, color=color, alpha=0.75,
                edgecolor=color, linewidth=0.8)
        ax.set_xlabel('Prediction Probability', fontsize=11, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title(f'{cls_name}', fontsize=14, fontweight='bold', color=color, pad=10)
        ax.grid(alpha=0.15, color=COLORS['text_dim'])
    
    fig.suptitle('Prediction Confidence Distribution', fontsize=16, fontweight='bold',
                 color=COLORS['teal'], y=1.02)
    
    plt.tight_layout()
    path = OUTPUT_DIR / "prediction_distribution.png"
    fig.savefig(str(path), dpi=200, bbox_inches='tight', facecolor=COLORS['bg_dark'])
    plt.close(fig)
    print(f"Saved: {path}")


def create_dashboard(metrics, cm):
    """Create a single comprehensive dashboard image."""
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(COLORS['bg_dark'])
    
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # ── Title ──
    fig.suptitle('🐔  Chicken Disease Classification — Evaluation Dashboard',
                 fontsize=22, fontweight='bold', color=COLORS['teal'], y=0.98)
    
    # ── 1. Metrics Cards (top-left) ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(COLORS['bg_dark'])
    ax1.axis('off')
    
    card_data = [
        ('Accuracy', metrics['accuracy'], COLORS['teal']),
        ('Precision', metrics['precision'], COLORS['green']),
        ('Recall', metrics['recall'], COLORS['purple']),
        ('F1 Score', metrics['f1'], COLORS['gold']),
    ]
    
    for idx, (name, val, color) in enumerate(card_data):
        y_pos = 0.85 - idx * 0.23
        ax1.text(0.05, y_pos, name, fontsize=13, fontweight='bold',
                 color=COLORS['text_dim'], transform=ax1.transAxes)
        ax1.text(0.05, y_pos - 0.1, f'{val:.2%}', fontsize=22, fontweight='bold',
                 color=color, transform=ax1.transAxes)
        # Progress bar background
        bar_bg = FancyBboxPatch((0.55, y_pos - 0.08), 0.42, 0.06,
                                boxstyle="round,pad=0.01", facecolor=COLORS['bg_card'],
                                transform=ax1.transAxes, clip_on=False)
        ax1.add_patch(bar_bg)
        # Progress bar fill
        bar_fill = FancyBboxPatch((0.55, y_pos - 0.08), 0.42 * val, 0.06,
                                  boxstyle="round,pad=0.01", facecolor=color, alpha=0.8,
                                  transform=ax1.transAxes, clip_on=False)
        ax1.add_patch(bar_fill)
    
    ax1.set_title('Overall Metrics', fontsize=15, fontweight='bold',
                  color=COLORS['text'], pad=15, loc='left')
    
    # ── 2. Confusion Matrix (top-center) ──
    ax2 = fig.add_subplot(gs[0, 1])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cmap = sns.color_palette([COLORS['bg_card'], COLORS['accent2'], COLORS['teal']], as_cmap=True)
    
    sns.heatmap(cm, annot=False, cmap=cmap, ax=ax2,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=2, linecolor=COLORS['bg_dark'], cbar=False)
    
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            color = COLORS['text'] if cm_normalized[i, j] < 0.7 else COLORS['bg_dark']
            ax2.text(j + 0.5, i + 0.38, f"{cm[i, j]}",
                     ha='center', va='center', fontsize=22, fontweight='bold', color=color)
            ax2.text(j + 0.5, i + 0.65, f"({cm_normalized[i, j]*100:.1f}%)",
                     ha='center', va='center', fontsize=10, color=color, alpha=0.85)
    
    ax2.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax2.set_ylabel('True', fontsize=11, fontweight='bold')
    ax2.set_title('Confusion Matrix', fontsize=15, fontweight='bold', pad=15, color=COLORS['text'])
    
    # ── 3. Per-Class Metrics (top-right) ──
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(CLASS_NAMES))
    width = 0.22
    
    ax3.bar(x - width, metrics['precision_per_class'], width, label='Precision',
            color=COLORS['teal'], zorder=3)
    ax3.bar(x, metrics['recall_per_class'], width, label='Recall',
            color=COLORS['purple'], zorder=3)
    ax3.bar(x + width, metrics['f1_per_class'], width, label='F1',
            color=COLORS['gold'], zorder=3)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax3.set_ylim(0, 1.15)
    ax3.set_title('Per-Class Metrics', fontsize=15, fontweight='bold', pad=15, color=COLORS['text'])
    ax3.legend(fontsize=9, facecolor=COLORS['bg_card'], edgecolor=COLORS['text_dim'],
               labelcolor=COLORS['text'])
    ax3.grid(axis='y', alpha=0.12, color=COLORS['text_dim'])
    
    # ── 4. Overall Bar (bottom-left) ──
    ax4 = fig.add_subplot(gs[1, 0])
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
    bar_colors = [COLORS['teal'], COLORS['green'], COLORS['purple'], COLORS['gold']]
    
    bars = ax4.barh(metric_names, metric_values, color=bar_colors, height=0.5, zorder=3)
    for bar, val in zip(bars, metric_values):
        ax4.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{val:.2%}', ha='left', va='center', fontsize=12, fontweight='bold')
    
    ax4.set_xlim(0, 1.2)
    ax4.set_title('Metrics Summary', fontsize=15, fontweight='bold', pad=15, color=COLORS['text'])
    ax4.grid(axis='x', alpha=0.12, color=COLORS['text_dim'])
    
    # ── 5. Placeholder for ROC info (bottom-center & right) ──
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.set_facecolor(COLORS['bg_card'])
    ax5.axis('off')
    
    # Build a nice text summary
    summary_lines = [
        f"Model: VGG16 (Transfer Learning)",
        f"Classes: {', '.join(CLASS_NAMES)}",
        f"Image Size: 224 × 224 × 3",
        f"Batch Size: {BATCH_SIZE}",
        f"",
        f"━━━ Weighted Averages ━━━",
        f"  Accuracy  :  {metrics['accuracy']:.4f}",
        f"  Precision :  {metrics['precision']:.4f}",
        f"  Recall    :  {metrics['recall']:.4f}",
        f"  F1 Score  :  {metrics['f1']:.4f}",
        f"",
        f"━━━ Per Class ━━━",
    ]
    for i, cls in enumerate(CLASS_NAMES):
        summary_lines.append(
            f"  {cls}: P={metrics['precision_per_class'][i]:.3f}  "
            f"R={metrics['recall_per_class'][i]:.3f}  "
            f"F1={metrics['f1_per_class'][i]:.3f}"
        )
    
    ax5.text(0.05, 0.95, '\n'.join(summary_lines), fontsize=12,
             fontfamily='monospace', color=COLORS['text'],
             verticalalignment='top', transform=ax5.transAxes,
             bbox=dict(boxstyle='round,pad=0.8', facecolor=COLORS['bg_card'],
                       edgecolor=COLORS['teal'], alpha=0.9))
    ax5.set_title('Classification Report', fontsize=15, fontweight='bold',
                  pad=15, color=COLORS['text'])
    
    path = OUTPUT_DIR / "evaluation_dashboard.png"
    fig.savefig(str(path), dpi=180, bbox_inches='tight', facecolor=COLORS['bg_dark'])
    plt.close(fig)
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  Chicken Disease Classification - Evaluation")
    print("=" * 55)
    
    # Step 1: Load model and predict
    y_true, y_pred, y_pred_probs, val_gen = load_and_predict()
    
    # Step 2: Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    
    # Step 3: Generate individual plots
    print("\nGenerating visualizations...")
    cm = plot_confusion_matrix(y_true, y_pred)
    plot_metrics_bar(metrics)
    plot_per_class_metrics(metrics)
    plot_roc_curve(y_true, y_pred_probs)
    plot_prediction_distribution(y_pred_probs)
    
    # Step 4: Generate comprehensive dashboard
    create_dashboard(metrics, cm)
    
    # Step 5: Save metrics to JSON
    save_metrics = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1_score': float(metrics['f1']),
        'per_class': {
            cls: {
                'precision': float(metrics['precision_per_class'][i]),
                'recall': float(metrics['recall_per_class'][i]),
                'f1': float(metrics['f1_per_class'][i]),
            }
            for i, cls in enumerate(CLASS_NAMES)
        }
    }
    
    json_path = OUTPUT_DIR / "evaluation_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(save_metrics, f, indent=4)
    print(f"Saved: {json_path}")
    
    print("\n" + "=" * 55)
    print(f"  All results saved to: {OUTPUT_DIR}")
    print("=" * 55)
