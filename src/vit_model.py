"""Hugging Face ViT for one-logit pneumonia label classification."""
from torch import nn
from transformers import ViTForImageClassification, ViTImageProcessor

MODEL_ID = "google/vit-base-patch16-224"
MODEL_REVISION = "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3"


class ViTBinaryClassifier(nn.Module):
    def __init__(self, network, unfreeze_last_block):
        super().__init__()
        self.network = network
        self.unfreeze_last_block = unfreeze_last_block
        network.vit.requires_grad_(False)
        if unfreeze_last_block:
            network.vit.encoder.layer[-1].requires_grad_(True)
            network.vit.layernorm.requires_grad_(True)

    @property
    def classifier(self):
        return self.network.classifier

    def forward(self, images):
        return self.network(pixel_values=images).logits

    def train_selected_layers(self):
        self.eval()
        self.classifier.train()
        if self.unfreeze_last_block:
            self.network.vit.encoder.layer[-1].train()
            self.network.vit.layernorm.train()


def load_vit(cache_dir, unfreeze_last_block):
    processor = ViTImageProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir, revision=MODEL_REVISION)
    network = ViTForImageClassification.from_pretrained(
        MODEL_ID, cache_dir=cache_dir, revision=MODEL_REVISION,
        num_labels=1, ignore_mismatched_sizes=True, use_safetensors=True,
    )
    return ViTBinaryClassifier(network, unfreeze_last_block), processor
