"""DenseNet-121 baseline: frozen ImageNet backbone and one binary logit."""
from pathlib import Path
import torch
from torch import nn
from torchvision.models import densenet121, DenseNet121_Weights


def build_model(pretrained=True, freeze_backbone=True):
    # Keep downloaded weights inside the ignored project model directory.
    cache=Path(__file__).resolve().parents[1]/'outputs/models/torch_cache'
    torch.hub.set_dir(str(cache))
    weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model=densenet121(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad=False
    model.classifier=nn.Linear(model.classifier.in_features,1)
    return model


def set_classifier_training(model):
    """Train the classifier while keeping backbone BatchNorm statistics fixed."""
    model.train()
    model.features.eval()
