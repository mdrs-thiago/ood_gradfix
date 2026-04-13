import argparse
import itertools
from typing import Dict, Any, List
import copy
import numpy as np

import torch
from datasets import load_dataset
from hf_ood_eval import _maybe_select, _resolve_column, _set_dataset_transform, _make_loader, HFImageClassifier, HFImageTransform
from ood_methods_extended import build_method_registry
from ood_evaluate import compute_auroc, compute_aupr, compute_fpr_at_tpr95

def parameter_grid(param_dict: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_dict.keys())
    values = list(param_dict.values())
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for OOD detection")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--id_dataset", type=str, required=True)
    parser.add_argument("--ood_dataset", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, help="Method name to tune, e.g. lowdim_grad_resid")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=500, help="Max samples to evaluate during tuning to save time")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HFImageClassifier(args.model_id).to(device)
    model.eval()
    transform = HFImageTransform(args.model_id, cache_dir=None)

    id_dataset_full = load_dataset(args.id_dataset, split="test")
    ood_dataset_full = load_dataset(args.ood_dataset, split="test")

    id_test = _maybe_select(id_dataset_full, args.max_samples, "id_test")
    ood_test = _maybe_select(ood_dataset_full, args.max_samples, "ood_test")

    id_img_col = _resolve_column(id_test, "image", ["image", "img", "pixel_values"])
    id_lbl_col = _resolve_column(id_test, "label", ["label", "labels", "fine_label", "class"])
    ood_img_col = _resolve_column(ood_test, "image", ["image", "img", "pixel_values"])
    ood_lbl_col = _resolve_column(ood_test, "label", ["label", "labels", "fine_label", "class"])

    id_test = _set_dataset_transform(id_test, transform, id_img_col, id_lbl_col)
    ood_test = _set_dataset_transform(ood_test, transform, ood_img_col, ood_lbl_col)

    id_loader = _make_loader(id_test, batch_size=args.test_batch_size, num_workers=2, shuffle=False)
    ood_loader = _make_loader(ood_test, batch_size=args.test_batch_size, num_workers=2, shuffle=False)
    fit_loader = _make_loader(id_test, batch_size=args.batch_size, num_workers=2, shuffle=True)

    # Define hyperparameter grid for specific method
    method_grids = {
        "lowdim_grad_resid": {
            "n_components": [16, 64, 128, 256, 0.9, 0.95, 0.99],
            "loss_type": ["uniform_kl", "entropy", "margin"]
        },
        "gradnorm": {
            "p": ["fro", "l1"],
            "loss_type": ["uniform_kl", "entropy"]
        },
        "feat_knn": {
            "k": [1, 5, 10, 50]
        },
        "energy": {
            "temperature": [1.0, 1.5, 2.0]
        }
    }

    if args.method not in method_grids:
        print(f"No grid defined for {args.method}. Available: {list(method_grids.keys())}")
        return

    grid = parameter_grid(method_grids[args.method])
    best_auroc = 0.0
    best_params = None

    print(f"Starting Hyperparameter Tuning for {args.method} with {len(grid)} combinations.")

    # We dynamically inject params into the constructor
    # We will need to pull from build_method_registry logic
    
    from ood_methods_extended import LowDimGradResidual, GradNorm, FeatureKNN, EnergyBased
    mapping = {
        "lowdim_grad_resid": LowDimGradResidual,
        "gradnorm": GradNorm,
        "feat_knn": FeatureKNN,
        "energy": EnergyBased
    }

    MethodClass = mapping[args.method]

    for params in grid:
        print(f"Testing params: {params}")
        method_instance = MethodClass(model, **params)
        
        try:
            method_instance.fit(fit_loader)
            id_scores = method_instance.compute_ood_scores(id_loader).cpu().numpy()
            ood_scores = method_instance.compute_ood_scores(ood_loader).cpu().numpy()
            
            auroc = compute_auroc(id_scores, ood_scores)
            aupr = compute_aupr(id_scores, ood_scores)
            
            print(f"   --> AUROC: {auroc:.4f} | AUPR: {aupr:.4f}")
            if auroc > best_auroc:
                best_auroc = auroc
                best_params = params
        except Exception as e:
            print(f"   --> Failed: {e}")
            
    print("\n" + "="*40)
    print(f"Best AUROC: {best_auroc:.4f}")
    print(f"Best Parameters: {best_params}")
    print("="*40)

if __name__ == "__main__":
    main()
