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
    # Standard OOD convention: TPR=95% on In-Distribution (ID) data.
    # Since higher scores -> more OOD, we set a threshold such that 95% of ID scores are below it.
    threshold = np.percentile(in_scores, 95)
    
    # FPR is the proportion of OOD data incorrectly classified as ID (scores below threshold)
    fpr_at_tpr95 = np.mean(out_scores < threshold)
    return float(fpr_at_tpr95)

def gather_metrics(metrics_list):
    """
    Given a list of metric dictionaries, compute best, mean, and std for each metric.
    """
    aggregated = {}
    if not metrics_list:
        return aggregated
    
    keys = metrics_list[0].keys()
    for k in keys:
        vals = [m[k] for m in metrics_list]
        aggregated[f"{k}_mean"] = np.mean(vals)
        aggregated[f"{k}_std"] = np.std(vals)
        
        if k in ['auroc', 'aupr']:
            aggregated[f"{k}_best"] = np.max(vals)
        else:
            aggregated[f"{k}_best"] = np.min(vals)
            
    return aggregated
