"""Hugging Face ViT for one-logit pneumonia label classification."""
from torch import nn
from transformers import ViTForImageClassification, ViTImageProcessor

MODEL_ID = "google/vit-base-patch16-224"
MODEL_REVISION = "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3"


class ViTBinaryClassifier(nn.Module):
    def __init__(self, network, unfreeze_last_block=False, training_scope=None):
        super().__init__()
        self.network = network
        self.training_scope = training_scope or ('last_block' if unfreeze_last_block else 'frozen')
        if self.training_scope not in ('frozen', 'last_block', 'llrd'):
            raise ValueError('Unknown ViT training scope')
        self.unfreeze_last_block = self.training_scope == 'last_block'
        network.vit.requires_grad_(False)
        if self.training_scope == 'llrd':
            network.vit.requires_grad_(True)
        elif self.unfreeze_last_block:
            network.vit.encoder.layer[-1].requires_grad_(True)
            network.vit.layernorm.requires_grad_(True)

    @property
    def classifier(self):
        return self.network.classifier

    def forward(self, images):
        return self.network(pixel_values=images).logits

    def train_selected_layers(self):
        if self.training_scope == 'llrd':
            self.train()
            return
        self.eval()
        self.classifier.train()
        if self.unfreeze_last_block:
            self.network.vit.encoder.layer[-1].train()
            self.network.vit.layernorm.train()


def load_vit(cache_dir, unfreeze_last_block=False, training_scope=None):
    processor = ViTImageProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir, revision=MODEL_REVISION)
    network = ViTForImageClassification.from_pretrained(
        MODEL_ID, cache_dir=cache_dir, revision=MODEL_REVISION,
        num_labels=1, ignore_mismatched_sizes=True, use_safetensors=True,
    )
    return ViTBinaryClassifier(network, unfreeze_last_block, training_scope), processor


class ViTInputTransform:
    """Shared by training, preview and inference; does not consume random state."""
    def __init__(self, processor, use_clahe=False):
        from src.preprocessing import ApplyCLAHE
        self.processor = processor
        self.clahe = ApplyCLAHE() if use_clahe else None

    def __call__(self, image):
        if self.clahe is not None:
            image = self.clahe(image)
        return self.processor(image.convert('RGB'), return_tensors='pt')['pixel_values'][0]
