import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def compute_auroc(in_scores, out_scores):
    y_true = np.concatenate([np.zeros(len(in_scores)), np.ones(len(out_scores))])
    y_scores = np.concatenate([in_scores, out_scores])
    auroc = roc_auc_score(y_true, y_scores)
    return auroc

def compute_aupr(in_scores, out_scores):
    y_true = np.concatenate([np.zeros(len(in_scores)), np.ones(len(out_scores))])
    y_scores = np.concatenate([in_scores, out_scores])
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    aupr = auc(recall, precision)
    return aupr

def compute_fpr_at_tpr95(in_scores, out_scores):
    y_true = np.concatenate([np.zeros(len(in_scores)), np.ones(len(out_scores))])
    y_scores = np.concatenate([in_scores, out_scores])
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    tpr95_index = np.where(recall >= 0.95)[0][0]
    fpr_at_tpr95 = 1 - precision[tpr95_index]
    return fpr_at_tpr95
