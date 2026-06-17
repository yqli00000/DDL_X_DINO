from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator
from typing import Literal, Sequence

from PIL import Image


CanvasAugmentMode = Literal["shrink", "repeat", "mosaic"]
BackgroundMode = Literal["corner", "random", "mean"]


@dataclass(frozen=True)
class PasteBox:
    """A sampled paste operation shared by the image and its mask."""

    src_box: tuple[int, int, int, int] | None
    dst_xy: tuple[int, int]
    dst_size: tuple[int, int]


class DDLCanvasAugment:
    """
    PIL image/mask pair augmentation that mimics common DDL test-time layouts.

    Supported layouts:
    - shrink: shrink one image and paste it onto a solid-color canvas.
    - repeat: paste several resized copies of one image onto a solid-color canvas.
    - mosaic: compose several different image/mask pairs into one tiled image.

    The same sampled geometry is always applied to ``image`` and ``mask``. Image
    canvases use a solid RGB background; mask canvases use zero background.
    """

    def __init__(
        self,
        p: float = 0.5,
        modes: Sequence[CanvasAugmentMode] = ("shrink", "repeat", "mosaic"),
        scale_range: tuple[float, float] = (0.35, 0.85),
        repeat_count_range: tuple[int, int] = (2, 4),
        mosaic_count_range: tuple[int, int] | None = None,
        mosaic_grid_choices: Sequence[tuple[int, int]] = ((2, 2), (2, 2), (2, 2), (3, 3)),
        mosaic_output_sizes: Sequence[tuple[int, int]] | None = None,
        background: BackgroundMode | tuple[int, int, int] = "corner",
        jitter_background: int = 12,
        allow_overlap: bool = True,
        rng: random.Random | None = None,
    ) -> None:
        if not 0.0 <= float(p) <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if not modes:
            raise ValueError("modes must contain at least one augmentation mode.")

        self.p = float(p)
        self.modes = tuple(modes)
        self.scale_range = self._validate_float_range(scale_range, "scale_range", lower=0.0)
        self.repeat_count_range = self._validate_int_range(repeat_count_range, "repeat_count_range", lower=1)
        self.mosaic_grid_choices = tuple(mosaic_grid_choices)
        max_mosaic_cells = max(int(rows) * int(cols) for rows, cols in self.mosaic_grid_choices)
        if mosaic_count_range is None:
            self.mosaic_count_range = (2, max_mosaic_cells)
        else:
            self.mosaic_count_range = self._validate_int_range(mosaic_count_range, "mosaic_count_range", lower=2)
        self.mosaic_output_sizes = None if mosaic_output_sizes is None else tuple(mosaic_output_sizes)
        self.background = background
        self.jitter_background = max(0, int(jitter_background))
        self.allow_overlap = bool(allow_overlap)
        self.rng = rng if rng is not None else random

        for mode in self.modes:
            if mode not in {"shrink", "repeat", "mosaic"}:
                raise ValueError(f"Unsupported canvas augmentation mode: {mode}")
        for rows, cols in self.mosaic_grid_choices:
            if int(rows) <= 0 or int(cols) <= 0:
                raise ValueError(f"Invalid mosaic grid: {(rows, cols)}")
        if self.mosaic_output_sizes is not None:
            for width, height in self.mosaic_output_sizes:
                if int(width) <= 0 or int(height) <= 0:
                    raise ValueError(f"Invalid mosaic output size: {(width, height)}")

    @contextmanager
    def use_rng(self, rng: random.Random) -> Iterator[None]:
        previous_rng = self.rng
        self.rng = rng
        try:
            yield
        finally:
            self.rng = previous_rng

    @staticmethod
    def _validate_float_range(
        value: tuple[float, float],
        name: str,
        lower: float,
    ) -> tuple[float, float]:
        lo, hi = float(value[0]), float(value[1])
        if lo < lower or hi < lo:
            raise ValueError(f"{name} must satisfy {lower} <= min <= max, got {value}")
        return lo, hi

    @staticmethod
    def _validate_int_range(value: tuple[int, int], name: str, lower: int) -> tuple[int, int]:
        lo, hi = int(value[0]), int(value[1])
        if lo < lower or hi < lo:
            raise ValueError(f"{name} must satisfy {lower} <= min <= max, got {value}")
        return lo, hi

    @staticmethod
    def _resize(image: Image.Image, size: tuple[int, int], interpolation: int) -> Image.Image:
        return image.resize(size, interpolation)

    @staticmethod
    def _crop_or_identity(image: Image.Image, box: tuple[int, int, int, int] | None) -> Image.Image:
        if box is None:
            return image
        return image.crop(box)

    def _sample_background(self, image: Image.Image) -> tuple[int, int, int]:
        if isinstance(self.background, tuple):
            return tuple(max(0, min(255, int(v))) for v in self.background)

        rgb = image.convert("RGB")
        width, height = rgb.size
        if self.background == "random":
            base = tuple(self.rng.randint(0, 255) for _ in range(3))
        elif self.background == "mean":
            thumb = rgb.resize((1, 1), Image.Resampling.BILINEAR)
            base = thumb.getpixel((0, 0))
        elif self.background == "corner":
            corners = [
                rgb.getpixel((0, 0)),
                rgb.getpixel((max(0, width - 1), 0)),
                rgb.getpixel((0, max(0, height - 1))),
                rgb.getpixel((max(0, width - 1), max(0, height - 1))),
            ]
            base = self.rng.choice(corners)
        else:
            raise ValueError(f"Unsupported background mode: {self.background}")

        if self.jitter_background <= 0:
            return tuple(int(v) for v in base)

        return tuple(
            max(0, min(255, int(v) + self.rng.randint(-self.jitter_background, self.jitter_background)))
            for v in base
        )

    def _new_canvases(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        return (
            Image.new("RGB", image.size, color=self._sample_background(image)),
            Image.new("L", mask.size, color=0),
        )

    def _paste_pair(
        self,
        image_canvas: Image.Image,
        mask_canvas: Image.Image,
        image: Image.Image,
        mask: Image.Image,
        paste_box: PasteBox,
    ) -> None:
        image_patch = self._crop_or_identity(image, paste_box.src_box)
        mask_patch = self._crop_or_identity(mask, paste_box.src_box)
        image_patch = self._resize(image_patch, paste_box.dst_size, Image.Resampling.BILINEAR)
        mask_patch = self._resize(mask_patch, paste_box.dst_size, Image.Resampling.NEAREST)
        image_canvas.paste(image_patch, paste_box.dst_xy)
        mask_canvas.paste(mask_patch, paste_box.dst_xy)

    def _sample_whole_image_box(self, canvas_size: tuple[int, int]) -> PasteBox:
        width, height = canvas_size
        scale = self.rng.uniform(*self.scale_range)
        dst_width = max(1, min(width, round(width * scale)))
        dst_height = max(1, min(height, round(height * scale)))
        x = self.rng.randint(0, max(0, width - dst_width))
        y = self.rng.randint(0, max(0, height - dst_height))
        return PasteBox(src_box=None, dst_xy=(x, y), dst_size=(dst_width, dst_height))

    def _shrink(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        image_canvas, mask_canvas = self._new_canvases(image, mask)
        self._paste_pair(image_canvas, mask_canvas, image, mask, self._sample_whole_image_box(image.size))
        return image_canvas, mask_canvas

    def _repeat(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        image_canvas, mask_canvas = self._new_canvases(image, mask)
        count = self.rng.randint(*self.repeat_count_range)
        boxes: list[PasteBox] = []
        for _ in range(count):
            box = self._sample_whole_image_box(image.size)
            if self.allow_overlap or not self._overlaps_any(box, boxes):
                boxes.append(box)
        if not boxes:
            boxes.append(self._sample_whole_image_box(image.size))

        for box in boxes:
            self._paste_pair(image_canvas, mask_canvas, image, mask, box)
        return image_canvas, mask_canvas

    def sample_mode(self) -> CanvasAugmentMode | None:
        """Return one augmentation mode, or ``None`` when augmentation is skipped."""

        if self.rng.random() > self.p:
            return None
        return self.rng.choice(self.modes)

    def sample_mosaic_count(self) -> int:
        return self.rng.randint(*self.mosaic_count_range)

    def sample_mosaic_grid(self) -> tuple[int, int]:
        rows, cols = self.rng.choice(self.mosaic_grid_choices)
        return int(rows), int(cols)

    def sample_mosaic_output_size(self, fallback_size: tuple[int, int]) -> tuple[int, int]:
        if self.mosaic_output_sizes is None:
            return fallback_size
        width, height = self.rng.choice(self.mosaic_output_sizes)
        return int(width), int(height)

    def apply_single(
        self,
        image: Image.Image,
        mask: Image.Image,
        mode: CanvasAugmentMode,
    ) -> tuple[Image.Image, Image.Image]:
        if image.size != mask.size:
            raise ValueError(f"image and mask must have the same size, got {image.size} and {mask.size}")

        image = image.convert("RGB")
        mask = mask.convert("L")
        if mode == "shrink":
            return self._shrink(image, mask)
        if mode == "repeat":
            return self._repeat(image, mask)
        raise ValueError(f"Mode '{mode}' requires multiple samples; use compose_mosaic_pairs instead.")

    @staticmethod
    def _rect(box: PasteBox) -> tuple[int, int, int, int]:
        x, y = box.dst_xy
        width, height = box.dst_size
        return x, y, x + width, y + height

    def _overlaps_any(self, box: PasteBox, boxes: Sequence[PasteBox]) -> bool:
        left, top, right, bottom = self._rect(box)
        for other in boxes:
            other_left, other_top, other_right, other_bottom = self._rect(other)
            if left < other_right and right > other_left and top < other_bottom and bottom > other_top:
                return True
        return False

    def compose_mosaic_pairs(
        self,
        pairs: Sequence[tuple[Image.Image, Image.Image]],
        output_size: tuple[int, int] | None = None,
        grid: tuple[int, int] | None = None,
    ) -> tuple[Image.Image, Image.Image]:
        """
        Compose several image/mask pairs into one tiled image.

        This is useful for a dataset-level mosaic where the caller can provide
        multiple samples. The caller is responsible for assigning the final label.
        """

        if not pairs:
            raise ValueError("pairs must contain at least one image/mask pair.")
        for pair_image, pair_mask in pairs:
            if pair_image.size != pair_mask.size:
                raise ValueError(
                    f"Every image/mask pair must have the same size, got {pair_image.size} and {pair_mask.size}"
                )

        first_image, first_mask = pairs[0]
        canvas_size = output_size if output_size is not None else first_image.size
        rows, cols = grid if grid is not None else self.rng.choice(self.mosaic_grid_choices)
        if rows <= 0 or cols <= 0:
            raise ValueError(f"grid must contain positive row/col counts, got {grid}")

        image_canvas = Image.new("RGB", canvas_size, color=self._sample_background(first_image))
        mask_canvas = Image.new("L", canvas_size, color=0)
        canvas_width, canvas_height = canvas_size
        pair_list = list(pairs)
        cell_count = rows * cols

        for cell_index in range(cell_count):
            row = cell_index // cols
            col = cell_index % cols
            if cell_index < len(pair_list):
                pair_image, pair_mask = pair_list[cell_index]
            else:
                pair_image, pair_mask = self.rng.choice(pair_list)
            pair_image = pair_image.convert("RGB")
            pair_mask = pair_mask.convert("L")
            left = round(col * canvas_width / cols)
            right = round((col + 1) * canvas_width / cols)
            top = round(row * canvas_height / rows)
            bottom = round((row + 1) * canvas_height / rows)
            paste_box = PasteBox(
                src_box=None,
                dst_xy=(left, top),
                dst_size=(max(1, right - left), max(1, bottom - top)),
            )
            self._paste_pair(image_canvas, mask_canvas, pair_image, pair_mask, paste_box)
        return image_canvas, mask_canvas

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if image.size != mask.size:
            raise ValueError(f"image and mask must have the same size, got {image.size} and {mask.size}")
        if self.rng.random() > self.p:
            return image, mask

        image = image.convert("RGB")
        mask = mask.convert("L")
        mode = self.rng.choice(self.modes)
        if mode == "shrink":
            return self._shrink(image, mask)
        if mode == "repeat":
            return self._repeat(image, mask)
        if mode == "mosaic":
            return self.compose_mosaic_pairs([(image, mask)])
        raise ValueError(f"Unsupported canvas augmentation mode: {mode}")


def apply_ddl_canvas_augment(
    image: Image.Image,
    mask: Image.Image,
    p: float = 1.0,
    modes: Sequence[CanvasAugmentMode] = ("shrink", "repeat", "mosaic"),
) -> tuple[Image.Image, Image.Image]:
    """Convenience wrapper for one-off use in existing dataset code."""

    return DDLCanvasAugment(p=p, modes=modes)(image, mask)
