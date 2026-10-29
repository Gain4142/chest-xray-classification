"""Deterministic preprocessing; CLAHE is an optional comparison condition."""
import math
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def to_rgb(image):
    """Normalize Pillow image mode without modifying the source file."""
    return image.convert('RGB')


class ApplyCLAHE:
    """Apply OpenCV CLAHE to 8-bit grayscale before replicating RGB channels."""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        if not math.isfinite(clip_limit) or clip_limit <= 0:
            raise ValueError('clip_limit must be finite and positive.')
        if len(tile_grid_size) != 2 or any(not isinstance(x, int) or x < 1 for x in tile_grid_size):
            raise ValueError('tile_grid_size must contain two positive integers.')
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tuple(tile_grid_size)

    def __call__(self, image):
        import cv2
        grayscale = np.asarray(image.convert('L'), dtype=np.uint8)
        equalizer = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        adjusted = equalizer.apply(grayscale)
        return Image.fromarray(adjusted).convert('RGB')


def build_transform(image_size=224, *, use_clahe=False, clip_limit=2.0, tile_grid_size=(8, 8)):
    """No cropping or augmentation. Default retains the no-CLAHE baseline.

    Order with CLAHE: source grayscale -> CLAHE -> RGB -> resize -> tensor -> normalize.
    The same CLAHE configuration must be used for training and inference if selected.
    """
    mode_transform = ApplyCLAHE(clip_limit, tile_grid_size) if use_clahe else to_rgb
    return transforms.Compose([
        transforms.Lambda(mode_transform),
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def build_eval_transform(image_size=224, *, use_clahe=False, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Deterministic preprocessing for validation/test; no random augmentation."""
    return build_transform(image_size, use_clahe=use_clahe, clip_limit=clip_limit, tile_grid_size=tile_grid_size)


def build_train_transform(image_size=224, *, augment=True, use_clahe=False, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Optional weak augmentation after resizing, before tensor normalization.

    Rotation: probability 0.5, angle uniform in [-5, 5] degrees, black fill.
    Brightness/contrast: probability 0.5, each factor in [0.9, 1.1].
    No flips, random crops, noise, erasing or color hue/saturation changes.
    Rotation keeps the output canvas size; small corner regions may be lost.
    """
    base = build_eval_transform(image_size, use_clahe=use_clahe, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    if not augment:
        return base
    augmentation = [
        transforms.RandomApply([
            transforms.RandomRotation(degrees=5, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0),
        ], p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ], p=0.5),
    ]
    # The deterministic pipeline has mode conversion + resize before tensor/normalize.
    return transforms.Compose(base.transforms[:2] + augmentation + base.transforms[2:])
