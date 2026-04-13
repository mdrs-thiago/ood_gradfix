import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
import os

def plot_score_distributions(in_scores, out_scores, method_name, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.histplot(in_scores, color="blue", label="In-Distribution", kde=True, stat="density", alpha=0.5, bins=50)
    sns.histplot(out_scores, color="red", label="Out-of-Distribution", kde=True, stat="density", alpha=0.5, bins=50)
    plt.title(f"Score Distribution - {method_name}")
    plt.xlabel("OOD Score")
    plt.ylabel("Density")
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_auroc_curve(in_scores, out_scores, method_name, save_path=None):
    y_true = np.concatenate([np.zeros(len(in_scores)), np.ones(len(out_scores))])
    y_scores = np.concatenate([in_scores, out_scores])
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {method_name}')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
