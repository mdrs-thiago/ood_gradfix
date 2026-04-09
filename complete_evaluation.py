import argparse 
from preprocessing import get_traditional_dataloader, get_custom_dataloader
from train_model import train_model, train_evaluating_ood
import torch
from models import build_torchvision_model
from loss_functions import get_traditional_criterion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OOD Detection Evaluation")
    parser.add_argument('--model', type=str, required=True, help='Name of the pretrained model')
    parser.add_argument('--in_dataset', type=str, required=True, help='In-distribution dataset name')
    parser.add_argument('--out_dataset', type=str, required=True, help='Out-of-distribution dataset name')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for dataloaders')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for dataloaders')
    parser.add_argument('--ood_method', type=str, default='MSP', help='OOD detection method to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    _, weights = build_torchvision_model('resnet18', 1, pretrained=True)
    transforms = weights.transforms()

    train_loader, val_loader, test_loader, n_classes = get_traditional_dataloader(args.in_dataset, transform_data=transforms)
    train_loader_1b, _, test_loader_1b, _ = get_traditional_dataloader(args.in_dataset, transform_data=transforms, batch_size=1)
    _, _, ood_loader, _ = get_traditional_dataloader(args.out_dataset, transform_data=transforms, batch_size=1, only_test=True)

    model, _ = build_torchvision_model(args.model, n_classes, pretrained=True)
    
    criterion = get_traditional_criterion('cross_entropy')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008)
    # model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
    model = train_evaluating_ood(model, train_loader, val_loader, train_loader_1b, test_loader_1b, ood_loader, criterion, optimizer, num_epochs=10, ood_method_name=args.ood_method)