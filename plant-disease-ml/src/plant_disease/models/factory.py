from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as tvm

def _replace_head(model: nn.Module, in_features: int, num_classes: int) -> nn.Module:
    head = nn.Sequential(
        nn.Linear(in_features, num_classes)
    )
    return head

def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, num_classes)
        return m
    elif name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feats = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feats, num_classes)
        return m
    elif name == "vit_b_16":
        m = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feats = m.heads.head.in_features
        m.heads.head = nn.Linear(in_feats, num_classes)
        return m
    else:
        raise ValueError(f"Unknown model name: {name}")
