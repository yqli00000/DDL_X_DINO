from __future__ import annotations

import random
from typing import Dict

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        return False


def build_image_transform(image_size: int) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def build_mask_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def _sample_multiple_in_range(min_size: int, max_size: int, patch_size: int) -> int:
    low = max(1, int(min_size) // int(patch_size))
    high = max(low, int(max_size) // int(patch_size))
    return random.randint(low, high) * int(patch_size)


def _apply_gps_style_aug(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    aug_type = random.choice(["none", "hflip", "rot90", "rot180", "rot270"])
    if aug_type == "hflip":
        return TF.hflip(image), TF.hflip(mask)
    if aug_type == "rot90":
        return TF.rotate(image, 90), TF.rotate(mask, 90)
    if aug_type == "rot180":
        return TF.rotate(image, 180), TF.rotate(mask, 180)
    if aug_type == "rot270":
        return TF.rotate(image, 270), TF.rotate(mask, 270)
    return image, mask


def batch_to_labels(batch: Dict[str, object], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "label": batch["label"].to(device),
        "mask": batch["mask"].to(device),
    }
