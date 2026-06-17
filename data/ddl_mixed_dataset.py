from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from .ddl_canvas_augment import DDLCanvasAugment
from .ddl_dataset_utils import (
    _nullcontext,
    _apply_gps_style_aug,
    _sample_multiple_in_range,
    build_image_transform,
    build_mask_transform,
)


VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class DDLMixedDataset(Dataset):
    """
    Unified dataset for mixed DDL directory layouts.

    Supported phase1 layout:

      root/
        fake/
          xxx.png
        real/
          yyy.png
        masks/
          xxx.png

    Notes:
    - fake samples try to read masks from ``root/masks/<filename>``.
    - real samples do not have GT masks, so an all-zero mask is generated.

    Supported phase2 layout:

      root/
        imgs/
          xxx.png
        masks/
          xxx.png
        labels.txt

    labels.txt format:
      <filename>,<label>

    Phase2 mask policy:
    - fake(label=1): try to read ``masks/<filename>``
    - real(label=0): generate an all-zero mask

    Dataset composition:
    - train mode:
      phase1_train + phase1_valid + phase2_train_split
    - val mode:
      phase2_val_split only
    """

    def __init__(
        self,
        *,
        phase1_train_root: str | Path | None = None,
        phase1_valid_root: str | Path | None = None,
        phase2_test_root: str | Path | None = None,
        image_size: int = 512,
        train: bool = False,
        gps_style_augment: bool = False,
        canvas_augment: bool = False,
        canvas_augment_eval: bool = False,
        canvas_augment_eval_seed: int = 2026,
        canvas_augment_prob: float = 0.5,
        canvas_augment_modes: tuple[str, ...] = ("shrink", "repeat", "mosaic"),
        canvas_augment_scale_range: tuple[float, float] = (0.45, 0.9),
        canvas_augment_repeat_count_range: tuple[int, int] = (2, 4),
        canvas_augment_mosaic_count_range: tuple[int, int] | None = None,
        canvas_augment_mosaic_grid_choices: tuple[tuple[int, int], ...] = ((2, 2), (2, 2), (2, 2), (3, 3)),
        canvas_augment_mosaic_output_sizes: tuple[tuple[int, int], ...] | None = None,
        min_size: int = 384,
        max_size: int = 768,
        patch_size: int = 16,
        phase2_val_ratio: float = 0.1,
        phase2_split_seed: int = 42,
        labels_filename: str = "labels.txt",
        phase2_images_dirname: str = "imgs",
        phase2_masks_dirname: str = "masks",
        require_phase1_fake_masks: bool = False,
        require_phase2_fake_masks: bool = True,
    ) -> None:
        self.phase1_train_root = None if phase1_train_root is None else Path(phase1_train_root)
        self.phase1_valid_root = None if phase1_valid_root is None else Path(phase1_valid_root)
        self.phase2_test_root = None if phase2_test_root is None else Path(phase2_test_root)

        self.train = train
        self.gps_style_augment = bool(gps_style_augment and train)
        self.canvas_augment_eval_seed = int(canvas_augment_eval_seed)
        canvas_augment_enabled = bool(canvas_augment and (train or canvas_augment_eval))
        self.canvas_augment = (
            DDLCanvasAugment(
                p=canvas_augment_prob,
                modes=canvas_augment_modes,
                scale_range=canvas_augment_scale_range,
                repeat_count_range=canvas_augment_repeat_count_range,
                mosaic_count_range=canvas_augment_mosaic_count_range,
                mosaic_grid_choices=canvas_augment_mosaic_grid_choices,
                mosaic_output_sizes=canvas_augment_mosaic_output_sizes,
            )
            if canvas_augment_enabled
            else None
        )
        self.min_size = int(min_size)
        self.max_size = int(max_size)
        self.patch_size = int(patch_size)
        self.phase2_val_ratio = float(phase2_val_ratio)
        self.phase2_split_seed = int(phase2_split_seed)
        self.labels_filename = str(labels_filename)
        self.phase2_images_dirname = str(phase2_images_dirname)
        self.phase2_masks_dirname = str(phase2_masks_dirname)
        self.require_phase1_fake_masks = bool(require_phase1_fake_masks)
        self.require_phase2_fake_masks = bool(require_phase2_fake_masks)

        self.image_transform = build_image_transform(image_size)
        self.mask_transform = build_mask_transform(image_size)
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        self.samples = self._build_samples()
        if not self.samples:
            raise ValueError("No valid samples were found for DDLMixedDataset.")

    def _ensure_dir(self, root: Path | None, name: str) -> Path | None:
        if root is None:
            return None
        if not root.exists():
            raise FileNotFoundError(f"{name} not found: {root}")
        return root

    def _iter_image_files(self, root: Path) -> List[Path]:
        files = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES]
        return sorted(files)

    def _phase1_samples_from_root(self, root: Path, source_name: str) -> List[Dict[str, object]]:
        fake_dir = root / "fake"
        real_dir = root / "real"
        masks_dir = root / "masks"

        if not fake_dir.exists():
            raise FileNotFoundError(f"Missing fake directory under {root}")
        if not real_dir.exists():
            raise FileNotFoundError(f"Missing real directory under {root}")

        samples: List[Dict[str, object]] = []

        for image_path in self._iter_image_files(fake_dir):
            mask_path = masks_dir / image_path.name
            if not mask_path.exists():
                if self.require_phase1_fake_masks:
                    raise FileNotFoundError(f"Missing fake mask: {mask_path}")
                mask_path = None
            samples.append(
                {
                    "uid": image_path.stem,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "label": 1,
                    "source": source_name,
                    "layout": "phase1",
                }
            )

        for image_path in self._iter_image_files(real_dir):
            samples.append(
                {
                    "uid": image_path.stem,
                    "image_path": image_path,
                    "mask_path": None,
                    "label": 0,
                    "source": source_name,
                    "layout": "phase1",
                }
            )

        return samples

    def _resolve_phase2_images_dir(self, root: Path) -> Path:
        direct_imgs = root / self.phase2_images_dirname
        if direct_imgs.exists():
            return direct_imgs

        # Some extractions may flatten images directly under root.
        image_files = self._iter_image_files(root)
        if image_files:
            return root

        raise FileNotFoundError(
            f"Could not find phase2 images under {direct_imgs} or directly under {root}"
        )

    def _phase2_samples(self) -> List[Dict[str, object]]:
        root = self._ensure_dir(self.phase2_test_root, "phase2_test_root")
        if root is None:
            return []

        labels_path = root / self.labels_filename
        masks_dir = root / self.phase2_masks_dirname
        images_dir = self._resolve_phase2_images_dir(root)

        if not labels_path.exists():
            raise FileNotFoundError(f"Missing phase2 labels file: {labels_path}")
        if not masks_dir.exists() and self.require_phase2_fake_masks:
            raise FileNotFoundError(f"Missing phase2 masks directory: {masks_dir}")

        samples: List[Dict[str, object]] = []
        for line_number, raw_line in enumerate(labels_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                image_name, label_text = [part.strip() for part in line.split(",", 1)]
            except ValueError as exc:
                raise ValueError(f"Invalid phase2 labels line {line_number}: {raw_line}") from exc

            image_path = images_dir / image_name
            if not image_path.exists():
                raise FileNotFoundError(f"Missing phase2 image for line {line_number}: {image_path}")

            label = int(label_text)
            if label == 0:
                mask_path = None
            else:
                mask_path = masks_dir / image_name
                if not mask_path.exists():
                    if self.require_phase2_fake_masks:
                        raise FileNotFoundError(f"Missing phase2 fake mask for line {line_number}: {mask_path}")
                    mask_path = None

            samples.append(
                {
                    "uid": Path(image_name).stem,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "label": label,
                    "source": "phase2_test",
                    "layout": "phase2",
                }
            )
        return samples

    def _split_phase2_samples(self, samples: Sequence[Dict[str, object]]) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        if not samples:
            return [], []
        if not 0.0 < self.phase2_val_ratio < 1.0:
            raise ValueError(f"phase2_val_ratio must be between 0 and 1, got {self.phase2_val_ratio}")

        indices = list(range(len(samples)))
        rng = random.Random(self.phase2_split_seed)
        rng.shuffle(indices)

        val_count = max(1, int(round(len(samples) * self.phase2_val_ratio)))
        val_index_set = set(indices[:val_count])

        train_split = [sample for idx, sample in enumerate(samples) if idx not in val_index_set]
        val_split = [sample for idx, sample in enumerate(samples) if idx in val_index_set]
        return train_split, val_split

    def _build_samples(self) -> List[Dict[str, object]]:
        samples: List[Dict[str, object]] = []

        phase2_all = self._phase2_samples()
        phase2_train, phase2_val = self._split_phase2_samples(phase2_all)

        if self.train:
            train_root = self._ensure_dir(self.phase1_train_root, "phase1_train_root")
            valid_root = self._ensure_dir(self.phase1_valid_root, "phase1_valid_root")
            if train_root is not None:
                samples.extend(self._phase1_samples_from_root(train_root, "phase1_train"))
            if valid_root is not None:
                samples.extend(self._phase1_samples_from_root(valid_root, "phase1_valid"))
            samples.extend(phase2_train)
        else:
            samples.extend(phase2_val)

        return samples

    @staticmethod
    def _build_fallback_mask(image_size: tuple[int, int], label: int) -> Image.Image:
        fill = 0 if int(label) == 0 else 255
        return Image.new("L", image_size, color=fill)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_mask_label(self, index: int) -> tuple[Image.Image, Image.Image, int, str]:
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")

        mask_path = sample["mask_path"]
        if mask_path is None:
            mask = self._build_fallback_mask(image.size, int(sample["label"]))
        else:
            mask = Image.open(mask_path).convert("L")

        label = int(sample["label"])
        return image, mask, label, str(sample["uid"])

    def _sample_mosaic_indices(self, index: int, count: int, rng: random.Random | None = None) -> List[int]:
        count = max(1, int(count))
        if len(self.samples) <= 1:
            return [index]
        pool = [idx for idx in range(len(self.samples)) if idx != index]
        extra_count = min(count - 1, len(pool))
        sampler = rng if rng is not None else random
        return [index] + sampler.sample(pool, extra_count)

    def _apply_canvas_augment(
        self,
        index: int,
        image: Image.Image,
        mask: Image.Image,
        label: int,
    ) -> tuple[Image.Image, Image.Image, int]:
        if self.canvas_augment is None:
            return image, mask, label

        rng = None if self.train else random.Random(self.canvas_augment_eval_seed + int(index))
        with self.canvas_augment.use_rng(rng) if rng is not None else _nullcontext():
            mode = self.canvas_augment.sample_mode()
            if mode is None:
                return image, mask, label
            if mode in {"shrink", "repeat"}:
                image, mask = self.canvas_augment.apply_single(image, mask, mode)
                return image, mask, label

            grid = self.canvas_augment.sample_mosaic_grid()
            count = max(grid[0] * grid[1], self.canvas_augment.sample_mosaic_count())
            indices = self._sample_mosaic_indices(index, count, rng=rng)
            pairs = []
            labels = []
            for mosaic_index in indices:
                mosaic_image, mosaic_mask, mosaic_label, _ = self._load_image_mask_label(mosaic_index)
                pairs.append((mosaic_image, mosaic_mask))
                labels.append(mosaic_label)
            output_size = self.canvas_augment.sample_mosaic_output_size(image.size)
            image, mask = self.canvas_augment.compose_mosaic_pairs(pairs, output_size=output_size, grid=grid)
            return image, mask, int(any(item != 0 for item in labels))

    def __getitem__(self, index: int) -> Dict[str, object]:
        image, mask, label, uid = self._load_image_mask_label(index)
        image, mask, label = self._apply_canvas_augment(index, image, mask, label)

        item: Dict[str, object] = {
            "uid": uid,
            "label": torch.tensor(label, dtype=torch.long),
        }
        if self.gps_style_augment:
            item["image"] = image
            item["mask_image"] = mask
        else:
            item["pixel_values"] = self.image_transform(image)
            item["mask"] = (self.mask_transform(mask) > 0.5).float()
        return item

    def collate_fn(self, batch: List[Dict[str, object]]) -> Dict[str, object]:
        if not self.gps_style_augment:
            return {
                "uid": [item["uid"] for item in batch],
                "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
                "label": torch.stack([item["label"] for item in batch]),
                "mask": torch.stack([item["mask"] for item in batch]),
            }

        height = _sample_multiple_in_range(self.min_size, self.max_size, self.patch_size)
        width = _sample_multiple_in_range(self.min_size, self.max_size, self.patch_size)
        image_interp = transforms.InterpolationMode.BILINEAR

        images = []
        masks = []
        for item in batch:
            image, mask = _apply_gps_style_aug(item["image"], item["mask_image"])
            image = TF.resize(image, [height, width], interpolation=image_interp)
            mask = TF.resize(mask, [height, width], interpolation=transforms.InterpolationMode.NEAREST)
            images.append(self.normalize(TF.to_tensor(image)))
            masks.append((TF.to_tensor(mask) > 0.5).float())

        return {
            "uid": [item["uid"] for item in batch],
            "pixel_values": torch.stack(images),
            "label": torch.stack([item["label"] for item in batch]),
            "mask": torch.stack(masks),
        }
