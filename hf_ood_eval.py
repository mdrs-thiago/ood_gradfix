import argparse
import gc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

from ood_methods_extended import build_method_registry
from ood_evaluate import compute_auroc, compute_aupr, compute_fpr_at_tpr95


def _resolve_column(dataset, preferred: str, candidates: List[str]) -> str:
    if preferred in dataset.column_names:
        return preferred
    for name in candidates:
        if name in dataset.column_names:
            return name
    raise KeyError(
        f"Column '{preferred}' not found. Available columns: {dataset.column_names}"
    )


def _parse_methods(methods_csv: str, registry: Dict[str, Any]) -> List[str]:
    requested = [m.strip().lower() for m in methods_csv.split(",") if m.strip()]
    unknown = [m for m in requested if m not in registry]
    if unknown:
        raise ValueError(
            f"Unknown method(s): {unknown}. Available: {sorted(registry.keys())}"
        )
    return requested


def _maybe_select(dataset, max_samples: Optional[int], label: str):
    if max_samples is None or max_samples <= 0:
        return dataset
    try:
        max_n = min(max_samples, len(dataset))
        return dataset.select(range(max_n))
    except Exception as exc:
        print(f"[warn] could not subset {label} to {max_samples} samples ({exc})")
        return dataset


class HFImageClassifier(nn.Module):
    def __init__(self, model_id: str, cache_dir: Optional[str] = None):
        super().__init__()
        self.model = AutoModelForImageClassification.from_pretrained(
            model_id, cache_dir=cache_dir
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits


class HFImageTransform:
    def __init__(self, model_id: str, cache_dir: Optional[str]):
        self.processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir)

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        images = batch["image"]
        processed = self.processor(images=images, return_tensors="pt")
        pixel_values = processed["pixel_values"]
        if isinstance(images, list):
            batch["pixel_values"] = pixel_values
            batch["labels"] = batch["label"]
        else:
            if pixel_values.dim() == 4 and pixel_values.size(0) == 1:
                pixel_values = pixel_values[0]
            batch["pixel_values"] = pixel_values
            batch["labels"] = batch["label"]
        return batch


class ColumnAdapter:
    def __init__(self, transform, image_column: str, label_column: str):
        self.transform = transform
        self.image_column = image_column
        self.label_column = label_column

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch["image"] = batch[self.image_column]
        batch["label"] = batch[self.label_column]
        return self.transform(batch)


def _set_dataset_transform(dataset, transform, image_column: str, label_column: str):
    dataset.set_transform(ColumnAdapter(transform, image_column, label_column))
    return dataset


def _collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
    else:
        pixel_values = [item["pixel_values"] for item in batch]
        labels = [item["labels"] for item in batch]

    if isinstance(pixel_values, list):
        pixel_values = torch.stack(pixel_values)
    elif pixel_values.dim() == 3:
        pixel_values = pixel_values.unsqueeze(0)

    if isinstance(labels, list):
        labels = torch.tensor([int(x) for x in labels])
    elif torch.is_tensor(labels) and labels.dim() == 0:
        labels = labels.unsqueeze(0)
    return pixel_values, labels


def _make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )


def main():
    parser = argparse.ArgumentParser(description="HF OOD evaluation with post-hoc methods")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--id_dataset", type=str, required=True)
    parser.add_argument("--ood_dataset", type=str, required=True)
    parser.add_argument("--id_config", type=str, default=None)
    parser.add_argument("--ood_config", type=str, default=None)
    parser.add_argument("--id_train_split", type=str, default="train")
    parser.add_argument("--id_test_split", type=str, default="test")
    parser.add_argument("--ood_test_split", type=str, default="test")
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--methods", type=str, default="msp,energy")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_id_train", type=int, default=10000)
    parser.add_argument("--max_id_test", type=int, default=10000)
    parser.add_argument("--max_ood_test", type=int, default=10000)
    parser.add_argument("--auto_fit_cap", type=int, default=10000)
    args = parser.parse_args()

    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = HFImageClassifier(args.model_id, cache_dir=args.cache_dir).to(device)
    model.eval()

    transform = HFImageTransform(args.model_id, cache_dir=args.cache_dir)

    id_train = load_dataset(
        args.id_dataset,
        args.id_config,
        split=args.id_train_split,
        cache_dir=args.cache_dir,
    )
    id_test = load_dataset(
        args.id_dataset,
        args.id_config,
        split=args.id_test_split,
        cache_dir=args.cache_dir,
    )

    ood_test = load_dataset(
        args.ood_dataset,
        args.ood_config,
        split=args.ood_test_split,
        cache_dir=args.cache_dir,
    )

    id_train = _maybe_select(id_train, args.max_id_train, "id_train")
    id_test = _maybe_select(id_test, args.max_id_test, "id_test")
    ood_test = _maybe_select(ood_test, args.max_ood_test, "ood_test")

    img_candidates = ["image", "img", "pixel_values", "pixels", "array", "filepath", "path"]
    label_candidates = ["label", "labels", "fine_label", "coarse_label", "class"]

    id_image_column = _resolve_column(id_train, args.image_column, img_candidates)
    id_label_column = _resolve_column(id_train, args.label_column, label_candidates)
    ood_image_column = _resolve_column(ood_test, args.image_column, img_candidates)
    ood_label_column = _resolve_column(ood_test, args.label_column, label_candidates)

    id_train = _set_dataset_transform(
        id_train, transform, id_image_column, id_label_column
    )
    id_test = _set_dataset_transform(
        id_test, transform, id_image_column, id_label_column
    )
    ood_test = _set_dataset_transform(
        ood_test, transform, ood_image_column, ood_label_column
    )

    id_train_loader = _make_loader(
        id_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    id_test_loader = _make_loader(
        id_test, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False
    )
    ood_test_loader = _make_loader(
        ood_test, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False
    )

    registry = build_method_registry(model)
    selected = _parse_methods(args.methods, registry)

    print("Selected methods:", ", ".join(selected))

    def extract_all_features(model, loader, device):
        from ood_methods_extended import HeadExtractor, _to_device, _iter_inputs
        extractor = HeadExtractor(model)
        feats, logits_list, labels = [], [], []
        with torch.no_grad():
            for x, y in _iter_inputs(loader):
                x = _to_device(x, device)
                head_io = extractor.forward(x)
                feats.append(head_io.h.cpu())
                logits_list.append(head_io.logits.cpu())
                if y is not None:
                    labels.append(y.cpu())
        H = torch.cat(feats, dim=0)
        L = torch.cat(logits_list, dim=0)
        Y = torch.cat(labels, dim=0) if labels else None
        return H, L, Y
        
    print("Pre-computing features for ID train...")
    id_train_h, id_train_logits, id_train_y = extract_all_features(model, id_train_loader, device)
    print("Pre-computing features for ID test...")
    id_test_h, id_test_logits, _ = extract_all_features(model, id_test_loader, device)
    print("Pre-computing features for OOD test...")
    ood_test_h, ood_test_logits, _ = extract_all_features(model, ood_test_loader, device)

    print("\nResults:")
    heavy_fit_methods = {
        "feat_knn",
        "feat_maha",
        "gradnorm",
        "gradorth",
        "lowdim_grad_resid",
        "gradvec_maha",
        "twosided_resid",
        "twosided_code_maha",
        "feat_gmm",
        "feat_pca",
        "react",
        "vim",
    }

    for name in selected:
        method = registry[name]
        try:
            fit_dataset = id_train
            if args.max_id_train is None and name in heavy_fit_methods:
                if args.auto_fit_cap is not None and len(fit_dataset) > args.auto_fit_cap:
                    print(
                        f"[warn] {name}: auto-capping fit set to {args.auto_fit_cap} samples. "
                        "Override with --max_id_train or --auto_fit_cap."
                    )
                    fit_dataset = _maybe_select(fit_dataset, args.auto_fit_cap, "id_train")
            fit_loader = _make_loader(
                fit_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
            )
            if hasattr(method, "fit_features"):
                if fit_dataset is id_train:
                    method.fit_features(id_train_h, id_train_logits, id_train_y)
                else:
                    fit_loader_ext = _make_loader(fit_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
                    f_h, f_l, f_y = extract_all_features(model, fit_loader_ext, device)
                    method.fit_features(f_h, f_l, f_y)
            else:
                method.fit(fit_loader)
        except Exception as exc:  # fit may be optional
            print(f"[warn] {name}: fit failed ({exc}) - skipping scoring")
            del registry[name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if hasattr(method, "compute_ood_scores_features"):
            id_scores = method.compute_ood_scores_features(id_test_h, id_test_logits).cpu().numpy()
            ood_scores = method.compute_ood_scores_features(ood_test_h, ood_test_logits).cpu().numpy()
        else:
            id_scores = method.compute_ood_scores(id_test_loader).cpu().numpy()
            ood_scores = method.compute_ood_scores(ood_test_loader).cpu().numpy()

        auroc = compute_auroc(id_scores, ood_scores)
        aupr = compute_aupr(id_scores, ood_scores)
        fpr95 = compute_fpr_at_tpr95(id_scores, ood_scores)

        print(
            f"{name:20s} | AUROC={auroc:.4f} | AUPR={aupr:.4f} | "
            f"FPR@95={fpr95:.4f} | id_mean={np.mean(id_scores):.4f} | "
            f"ood_mean={np.mean(ood_scores):.4f}"
        )
        del registry[name]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
