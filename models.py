import torch
import torch.nn as nn

from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    swin_t, Swin_T_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    vit_b_16, ViT_B_16_Weights,
)

def compute_class_weights_from_splits(train_split, device):
    counts = torch.tensor([len(paths) for paths in train_split], dtype=torch.float)
    total = counts.sum()
    weights = total / (counts.clamp(min=1.0))  # inverse frequency
    weights = weights / weights.mean()         # normalize
    return weights.to(device)

def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
    

def build_torchvision_model(name, num_classes, pretrained=True):

    name = name.lower()

    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        for param in model.fc.parameters():
            param.requires_grad = True
        return model, weights

    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet50(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        for param in model.fc.parameters():
            param.requires_grad = True
        return model, weights

    if name == "densenet121":
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet121(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model, weights

    if name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        for param in model.classifier[-1].parameters():
            param.requires_grad = True
        return model, weights

    if name == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        for param in model.classifier[-1].parameters():
            param.requires_grad = True
        return model, weights

    if name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_tiny(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        for param in model.classifier[-1].parameters():
            param.requires_grad = True
        return model, weights

    if name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        for param in model.classifier[-1].parameters():
            param.requires_grad = True
        return model, weights

    if name == "swin_t":
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = swin_t(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        model.head = nn.Linear(model.head.in_features, num_classes)
        for param in model.head.parameters():
            param.requires_grad = True
        return model, weights

    if name == "vit_b_16":
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = vit_b_16(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        model.heads[-1] = nn.Linear(model.heads[-1].in_features, num_classes)
        for param in model.heads[-1].parameters():
            param.requires_grad = True
        return model, weights

    raise ValueError(f"Unknown model name: {name}")

def set_classifier_head(model, model_name, num_classes, dropout_p=0.2):
    """
    Replaces the final classifier with Dropout + Linear, and returns the *head module*
    whose parameters should be trained.
    """
    name = model_name.lower()

    if name in ("resnet18", "resnet50"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
        return model.fc

    if name == "densenet121":
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
        return model.classifier

    if name in ("efficientnet_b0", "convnext_tiny", "mobilenet_v3_small", "mobilenet_v3_large"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
        return model.classifier[-1]

    if name == "swin_t":
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
        return model.head

    if name in ("vit_b_16",):
        # torchvision ViT: model.heads is a Sequential; last is Linear
        in_features = model.heads[-1].in_features
        model.heads[-1] = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
        return model.heads[-1]

    raise ValueError(f"Unsupported model_name for head replacement: {model_name}")


def freeze_backbone_train_head_only(model, head_module):
    """
    Freezes all parameters, then unfreezes only those inside head_module.
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in head_module.parameters():
        p.requires_grad = True