from __future__ import annotations

import argparse
import base64
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import io
import json
import logging
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, MutableMapping

import numpy as np
from openai import OpenAI
from PIL import Image
from tqdm import tqdm


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Second-stage script: read existing JSON and mask files. "
            "For real samples, copy the original JSON to a new JSON folder. "
            "For fake samples, call the API to refine 'Visible forgery traces' and save the modified JSON to the new folder. "
            "Original JSON files will not be overwritten."
        )
    )

    parser.add_argument("--image-path", default=None, help="Path to a single original image.")
    parser.add_argument("--image-dir", default=None, help="Directory containing original images.")
    parser.add_argument(
        "--image-list",
        default=None,
        help=(
            "Text file containing images to process, one per line. "
            "Each line can be an absolute/relative image path or an image filename under --image-dir."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Directory containing original json/ and mask/ subdirectories.")
    parser.add_argument(
        "--new-json-dir",
        default=None,
        help=(
            "Directory to save the new JSON files. "
            "Default: output-dir/json_api_refined"
        ),
    )

    parser.add_argument("--limit-images", type=int, default=None, help="Only process the first N sorted images.")
    parser.add_argument("--start-index", type=int, default=None, help="Start offset in the sorted image list, inclusive.")
    parser.add_argument("--end-index", type=int, default=None, help="End offset in the sorted image list, exclusive.")

    parser.add_argument("--explain-api-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--explain-api-key", default=None)
    parser.add_argument("--explain-model", default="qwen3.6-plus")
    parser.add_argument("--explain-timeout", type=int, default=60)
    parser.add_argument("--explain-max-tokens", type=int, default=700)
    parser.add_argument("--explain-workers", type=int, default=1, help="Number of concurrent explain API calls.")
    parser.add_argument("--max-api-calls", type=int, default=None, help="Stop calling the explain API after this many calls.")

    parser.add_argument(
        "--skip-empty-old-traces",
        action="store_true",
        help="For fake samples, skip API calls when the existing Visible forgery traces field is empty and copy the original JSON instead.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without writing new JSON files.",
    )

    parser.add_argument("--log-file", default=None)
    parser.add_argument("--summary-json", default=None)

    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_logger(output_dir: str | Path, log_file: str | None) -> logging.Logger:
    logger = logging.getLogger("update_visible_forgery_traces_new_json")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    log_path = Path(log_file) if log_file else Path(output_dir) / "update_visible_forgery_traces_new_json.log"
    ensure_dir(log_path.parent)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

    logger.info("log_file=%s", log_path)
    return logger


def collect_image_paths(
    image_path: str | None,
    image_dir: str | None,
    image_list: str | None = None,
) -> List[Path]:
    if image_path:
        return [Path(image_path)]

    if image_list:
        list_path = Path(image_list)
        if not list_path.exists():
            raise FileNotFoundError(f"Image list not found: {list_path}")

        root = Path(image_dir) if image_dir else None
        name_to_path: Dict[str, Path] = {}
        if root is not None:
            if not root.exists():
                raise FileNotFoundError(f"Image directory not found: {root}")
            name_to_path = {
                path.name: path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            }

        image_paths: List[Path] = []
        for line_number, raw_line in enumerate(list_path.read_text(encoding="utf-8").splitlines(), start=1):
            item = raw_line.strip()
            if not item or item.startswith("#"):
                continue

            candidate = Path(item)
            if candidate.exists():
                image_paths.append(candidate)
                continue

            if root is not None:
                resolved = name_to_path.get(candidate.name)
                if resolved is not None:
                    image_paths.append(resolved)
                    continue

            raise FileNotFoundError(
                f"Image listed at {list_path}:{line_number} was not found: {item}. "
                "Use an existing path or provide --image-dir for filename lookup."
            )

        if not image_paths:
            raise ValueError(f"No images found in list: {list_path}")

        return image_paths

    if not image_dir:
        raise ValueError("One of --image-path, --image-list, or --image-dir must be provided.")

    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")

    image_paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    if not image_paths:
        raise ValueError(f"No supported images found in {root}")

    return image_paths


def build_output_stem(name: str) -> str:
    safe = Path(name).stem.strip()
    return safe if safe else "sample"


def guess_mime_type(filename: str, default: str = "image/png") -> str:
    suffix = Path(filename).suffix.lower()
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }
    return mapping.get(suffix, default)


def encode_data_url(data: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('utf-8')}"


def increment_stat(
    api_stats: MutableMapping[str, int] | None,
    key: str,
    amount: int = 1,
    lock: threading.Lock | None = None,
) -> None:
    if api_stats is None:
        return

    if lock is None:
        api_stats[key] = api_stats.get(key, 0) + amount
        return

    with lock:
        api_stats[key] = api_stats.get(key, 0) + amount


def get_stat(
    api_stats: MutableMapping[str, int] | None,
    key: str,
    lock: threading.Lock | None = None,
) -> int:
    if api_stats is None:
        return 0

    if lock is None:
        return api_stats.get(key, 0)

    with lock:
        return api_stats.get(key, 0)


def is_fake_prediction(prediction: Any) -> bool:
    """
    Robustly judge whether the JSON classification result means fake.

    Expected original value:
        "fake" or "real"

    Also supports:
        true/false, 1/0, "1"/"0", "true"/"false"
    """
    if isinstance(prediction, bool):
        return prediction

    if isinstance(prediction, (int, float)):
        return int(prediction) == 1

    value = str(prediction).strip().lower()
    return value in {"fake", "1", "true", "yes", "forged", "tampered", "manipulated"}


def build_mask_overlay_bytes(
    image_path: Path,
    mask_path: Path,
    alpha: float = 0.45,
) -> bytes:
    """
    Create an overlay image:
        original image + red transparent suspicious region.

    This helps the vision-language API understand the mask position.
    """
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image).astype(np.float32)

    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(image.size, Image.Resampling.NEAREST)
    mask_np = np.array(mask) > 127

    overlay = image_np.copy()
    red = np.zeros_like(image_np)
    red[..., 0] = 255.0

    overlay[mask_np] = image_np[mask_np] * (1.0 - alpha) + red[mask_np] * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")
    return buffer.getvalue()


def build_mask_crop_bytes(
    image_path: Path,
    mask_path: Path,
    pad_ratio: float = 0.25,
) -> bytes | None:
    """
    Crop the original image around the mask bounding box.
    If the mask is empty, return None.
    """
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(image.size, Image.Resampling.NEAREST)
    mask_np = np.array(mask) > 127

    ys, xs = np.where(mask_np)
    if len(xs) == 0 or len(ys) == 0:
        return None

    h, w = mask_np.shape
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    box_w = max(1, x2 - x1 + 1)
    box_h = max(1, y2 - y1 + 1)

    pad_x = int(box_w * pad_ratio)
    pad_y = int(box_h * pad_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)

    crop = image_np[y1 : y2 + 1, x1 : x2 + 1]

    buffer = io.BytesIO()
    Image.fromarray(crop).save(buffer, format="PNG")
    return buffer.getvalue()


def build_plain_image_bytes(image_path: Path) -> bytes:
    buffer = io.BytesIO()
    Image.open(image_path).convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def build_api_prompt(old_traces: str, prediction: Any, has_crop: bool = True, has_overlay: bool = True) -> str:
    old_traces = old_traces.strip() if isinstance(old_traces, str) else ""
    fake_prediction = is_fake_prediction(prediction)

    crop_text = (
        "The third image is a crop centered on the same highlighted suspicious region. "
        if has_crop and has_overlay
        else ""
    )
    focus_text = (
        "The second image is the same image with a semi-transparent red overlay marking predicted suspicious regions. "
        if has_overlay
        else "The second image is the same image without any highlighted suspicious region. "
    )
    common_header = (
        "You are writing an official image-forensics annotation in the same style as a forensic dataset ground-truth description.\n\n"
        "The first image is the original image. "
        f"{focus_text}"
        f"{crop_text}"
        "Use the provided images only to focus your forensic analysis; do not mention the overlay, mask, crop, or model prediction in the final answer.\n\n"
        f"The classification result is {'fake' if fake_prediction else 'real'}.\n\n"
        "Previous rough description, if useful: "
        f"{old_traces if old_traces else 'None'}.\n"
        "Use the previous rough description only as weak reference. Do not copy duplicated Summary lines, repeated paragraphs, "
        "or any wording that conflicts with the visible image evidence.\n\n"
    )

    if fake_prediction:
        return (
            common_header +
            "Output format:\n"
            "1. First paragraph: describe the visible image content, including subject, apparent age/gender if visible, "
            "clothing, accessories, background, camera setting, expression, and lighting.\n"
            "2. Then write 4 to 6 markdown bullet points analyzing localized visible forensic evidence.\n"
            "3. End without a Summary sentence; it will be appended separately.\n\n"
            "For fake images, focus only on the suspicious visual region indicated by the auxiliary images, but do not mention "
            "mask, overlay, crop, highlighted region, prediction, or bounding box.\n\n"
            "Prefer these forensic headings when applicable, and keep the headings stable rather than inventing new wording:\n"
            "- **Inconsistent skin texture and shading**\n"
            "- **Mismatched lighting direction and intensity**\n"
            "- **Anatomical disproportion and symmetry**\n"
            "- **Edge artifacts and color bleeding**\n"
            "- **Blurred or distorted edges**\n"
            "- **Inconsistent eye reflections**\n"
            "- **Resolution and noise mismatch**\n\n"
            "Cover concrete visible evidence when applicable, using dataset-style terminology such as pores, micro-texture, "
            "vellus hair, skin tone, luminance discontinuity, halo-like boundary, blending, shadowing, cast shadow, "
            "specular highlights, catchlights, symmetry, anatomical proportion, nose bridge, alae, philtrum, vermilion border, "
            "eyelid crease, iris, sclera, hairline, cheek, jawline, clothing edges, object boundaries, resolution, and pixel distribution.\n\n"
            "Keep the tone observational and technical, not conversational. Do not mention confidence scores, model thresholds, "
            "algorithms, bounding box coordinates, or that a mask/overlay/crop was provided."
            "Each bullet must form a complete evidence chain: describe the suspicious local visual cue, compare it with adjacent normal regions, explain the expected natural appearance, and state why the observed cue supports manipulation.\n"
            "Whenever possible, compare the suspicious region with nearby unaltered reference areas, such as surrounding skin, the opposite side of the face, adjacent hair, clothing, object boundaries, or background.\n"
            "For facial manipulations, use anatomically specific descriptions when visible, such as eyelid crease, lid margin, medial and lateral canthi, brow ridge, orbital rim, philtrum, vermilion border, jawline, hairline, and cheek contour.\n"
        )

    return (
        common_header +
        "Output format:\n"
        "1. First paragraph: describe the visible image content, including subject, apparent age/gender if visible, "
        "clothing, accessories, background, camera setting, expression, and lighting.\n"
        "2. Second paragraph: explain authenticity in one continuous paragraph. Begin with either "
        "\"No signs of manipulation are detected.\" or \"No visual evidence of manipulation is present.\"\n"
        "3. Do not use markdown bullet points for real images.\n"
        "4. End without a Summary sentence; it will be appended separately.\n\n"
        "For real images, explain consistency in lighting, shadows, hair or object edges, skin texture, reflections, "
        "resolution and noise, depth of field, perspective, anatomy, and physical plausibility. Mention the absence of "
        "copy-paste artifacts, halos, unnatural smoothing, cloned patterns, and resolution shifts when visually appropriate.\n\n"
        "Keep the tone observational and technical, not conversational. Do not mention confidence scores, model thresholds, "
        "algorithms, bounding box coordinates, or that a mask/overlay/crop was provided."
    )


def strip_summary(text: str) -> str:
    return re.sub(
        r"\s*Summary:\s*This image has(?: not)? been tampered with\.?\s*$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )


def append_summary(text: str, prediction: Any) -> str:
    text = strip_summary(text)
    if is_fake_prediction(prediction):
        summary = "Summary: This image has been tampered with."
    else:
        summary = "Summary: This image has not been tampered with."
    return f"{text.rstrip()}\n\n{summary}"


def generate_visible_forgery_traces_from_old_text(
    image_bytes: bytes,
    image_name: str,
    overlay_bytes: bytes,
    crop_bytes: bytes | None,
    old_traces: str,
    prediction: Any,
    has_overlay: bool,
    *,
    api_url: str | None,
    api_key: str | None,
    api_model: str,
    timeout: int,
    max_tokens: int,
    max_api_calls: int | None = None,
    api_stats: MutableMapping[str, int] | None = None,
    api_stats_lock: threading.Lock | None = None,
    logger: logging.Logger | None = None,
) -> str:
    """
    Call the external API to rewrite the visible forgery trace description.

    Important:
    - Uses original image + red overlay + optional crop.
    - If the API fails, returns old_traces so the JSON content is not destroyed.
    """
    if not api_url:
        if logger:
            logger.warning("skip_api reason=no_api_url image=%s", image_name)
        return append_summary(old_traces, prediction)

    if max_api_calls is not None and get_stat(api_stats, "api_calls", lock=api_stats_lock) >= max_api_calls:
        if logger:
            logger.warning("skip_api reason=max_api_calls image=%s limit=%s", image_name, max_api_calls)
        return append_summary(old_traces, prediction)

    try:
        increment_stat(api_stats, "api_calls", lock=api_stats_lock)

        if logger:
            logger.info("call_api image=%s model=%s", image_name, api_model)

        content: List[Dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": encode_data_url(image_bytes, guess_mime_type(image_name)),
                },
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": encode_data_url(overlay_bytes, "image/png"),
                },
            },
        ]

        if crop_bytes is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encode_data_url(crop_bytes, "image/png"),
                    },
                }
            )

        content.append(
            {
                "type": "text",
                "text": build_api_prompt(
                    old_traces,
                    prediction,
                    has_crop=crop_bytes is not None,
                    has_overlay=has_overlay,
                ),
            }
        )

        client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=api_url,
            timeout=float(timeout),
        )

        completion = client.chat.completions.create(
            model=api_model,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            max_tokens=int(max_tokens),
        )

    except Exception as exc:
        increment_stat(api_stats, "api_failed", lock=api_stats_lock)
        if logger:
            logger.warning("api_failed image=%s error=%s: %s", image_name, type(exc).__name__, exc)
        return append_summary(old_traces, prediction)

    increment_stat(api_stats, "api_succeeded", lock=api_stats_lock)

    usage = getattr(completion, "usage", None)
    if usage is not None:
        for usage_key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(usage, usage_key, None)
            if isinstance(value, int):
                increment_stat(api_stats, usage_key, amount=value, lock=api_stats_lock)

        if logger:
            logger.info(
                "api_usage image=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                image_name,
                getattr(usage, "prompt_tokens", None),
                getattr(usage, "completion_tokens", None),
                getattr(usage, "total_tokens", None),
            )

    content = completion.choices[0].message.content
    if isinstance(content, str) and content.strip():
        return append_summary(content.strip(), prediction)

    increment_stat(api_stats, "api_empty_response", lock=api_stats_lock)
    return append_summary(old_traces, prediction)


def read_json_record(json_path: Path) -> Dict[str, Any]:
    return json.loads(json_path.read_text(encoding="utf-8"))


def write_json_record(json_path: Path, record: Dict[str, Any]) -> None:
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_json(src_json_path: Path, dst_json_path: Path, dry_run: bool = False) -> None:
    ensure_dir(dst_json_path.parent)
    if not dry_run:
        shutil.copy2(src_json_path, dst_json_path)


def update_one_record(
    image_path: str | Path,
    *,
    output_dir: str | Path,
    new_json_dir: str | Path,
    api_url: str | None,
    api_key: str | None,
    api_model: str,
    timeout: int,
    max_tokens: int,
    max_api_calls: int | None,
    skip_empty_old_traces: bool,
    dry_run: bool,
    api_stats: MutableMapping[str, int],
    api_stats_lock: threading.Lock,
    logger: logging.Logger | None = None,
) -> Dict[str, object]:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    new_json_dir = ensure_dir(Path(new_json_dir))

    output_stem = build_output_stem(image_path.name)
    old_json_path = output_dir / "json" / f"{output_stem}.json"
    mask_path = output_dir / "mask" / f"{output_stem}.png"
    new_json_path = new_json_dir / f"{output_stem}.json"

    if not image_path.exists():
        increment_stat(api_stats, "missing_image", lock=api_stats_lock)
        return {
            "image_path": str(image_path),
            "old_json_path": str(old_json_path),
            "new_json_path": str(new_json_path),
            "mask_path": str(mask_path),
            "status": "missing_image",
        }

    if not old_json_path.exists():
        increment_stat(api_stats, "missing_json", lock=api_stats_lock)
        if logger:
            logger.warning("missing_json image=%s json=%s", image_path, old_json_path)
        return {
            "image_path": str(image_path),
            "old_json_path": str(old_json_path),
            "new_json_path": str(new_json_path),
            "mask_path": str(mask_path),
            "status": "missing_json",
        }

    try:
        record = read_json_record(old_json_path)
    except Exception as exc:
        increment_stat(api_stats, "bad_json", lock=api_stats_lock)
        if logger:
            logger.warning("bad_json json=%s error=%s: %s", old_json_path, type(exc).__name__, exc)
        return {
            "image_path": str(image_path),
            "old_json_path": str(old_json_path),
            "new_json_path": str(new_json_path),
            "mask_path": str(mask_path),
            "status": "bad_json",
            "error": f"{type(exc).__name__}: {exc}",
        }

    prediction = record.get("Classification result", "")
    old_traces = record.get("Visible forgery traces", "")
    if not isinstance(old_traces, str):
        old_traces = json.dumps(old_traces, ensure_ascii=False)

    fake_prediction = is_fake_prediction(prediction)

    # Fake samples should have a mask for overlay/crop. Real samples can still be
    # rewritten from the original image without a highlighted region.
    if fake_prediction and not mask_path.exists():
        increment_stat(api_stats, "missing_mask_for_fake", lock=api_stats_lock)
        if logger:
            logger.warning("missing_mask_for_fake image=%s mask=%s", image_path, mask_path)

        # To avoid losing a fake record, copy the original JSON when mask is missing.
        copy_json(old_json_path, new_json_path, dry_run=dry_run)
        increment_stat(api_stats, "copied_fake_due_to_missing_mask", lock=api_stats_lock)
        return {
            "image_path": str(image_path),
            "old_json_path": str(old_json_path),
            "new_json_path": str(new_json_path),
            "mask_path": str(mask_path),
            "status": "copied_fake_due_to_missing_mask" if not dry_run else "dry_run_copy_fake_due_to_missing_mask",
            "classification_result": prediction,
            "old_visible_forgery_traces": old_traces,
        }

    if fake_prediction and skip_empty_old_traces and not old_traces.strip():
        copy_json(old_json_path, new_json_path, dry_run=dry_run)
        increment_stat(api_stats, "copied_fake_due_to_empty_old_traces", lock=api_stats_lock)
        return {
            "image_path": str(image_path),
            "old_json_path": str(old_json_path),
            "new_json_path": str(new_json_path),
            "mask_path": str(mask_path),
            "status": "copied_fake_due_to_empty_old_traces" if not dry_run else "dry_run_copy_fake_due_to_empty_old_traces",
            "classification_result": prediction,
            "old_visible_forgery_traces": old_traces,
        }

    image_bytes = image_path.read_bytes()
    has_overlay = mask_path.exists()
    overlay_bytes = build_mask_overlay_bytes(image_path, mask_path) if has_overlay else build_plain_image_bytes(image_path)
    crop_bytes = build_mask_crop_bytes(image_path, mask_path) if has_overlay else None

    new_traces = generate_visible_forgery_traces_from_old_text(
        image_bytes,
        image_path.name,
        overlay_bytes,
        crop_bytes,
        old_traces,
        prediction,
        has_overlay,
        api_url=api_url,
        api_key=api_key,
        api_model=api_model,
        timeout=timeout,
        max_tokens=max_tokens,
        max_api_calls=max_api_calls,
        api_stats=api_stats,
        api_stats_lock=api_stats_lock,
        logger=logger,
    )

    record["Visible forgery traces"] = new_traces

    ensure_dir(new_json_path.parent)
    if not dry_run:
        write_json_record(new_json_path, record)
        status = "updated_fake" if fake_prediction else "updated_real"
        increment_stat(api_stats, status, lock=api_stats_lock)
    else:
        status = "dry_run_update_fake" if fake_prediction else "dry_run_update_real"
        increment_stat(api_stats, status, lock=api_stats_lock)

    return {
        "image_path": str(image_path),
        "old_json_path": str(old_json_path),
        "new_json_path": str(new_json_path),
        "mask_path": str(mask_path),
        "status": status,
        "classification_result": prediction,
        "old_visible_forgery_traces": old_traces,
        "new_visible_forgery_traces": new_traces,
    }


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    new_json_dir = Path(args.new_json_dir) if args.new_json_dir else output_dir / "json_api_refined"
    ensure_dir(new_json_dir)

    logger = configure_logger(output_dir, args.log_file)
    logger.info("new_json_dir=%s", new_json_dir)

    image_paths = collect_image_paths(args.image_path, args.image_dir, args.image_list)
    total_available_images = len(image_paths)

    if args.start_index is not None or args.end_index is not None:
        start_index = max(0, int(args.start_index or 0))
        end_index = None if args.end_index is None else max(start_index, int(args.end_index))
        image_paths = image_paths[start_index:end_index]
    elif args.limit_images is not None:
        image_paths = image_paths[: max(0, int(args.limit_images))]

    api_stats: Dict[str, int] = {}
    api_stats_lock = threading.Lock()

    explain_workers = max(1, int(args.explain_workers))
    max_pending_explains = max(1, explain_workers * 4)

    results: List[Dict[str, object]] = []

    def collect_finished(
        pending: MutableMapping[Future, Path],
        *,
        block: bool = False,
    ) -> None:
        if not pending:
            return

        timeout = None if block else 0
        done, _ = wait(pending.keys(), timeout=timeout, return_when=FIRST_COMPLETED)

        for future in done:
            image_path = pending.pop(future)
            try:
                results.append(future.result())
            except Exception as exc:
                increment_stat(api_stats, "worker_failed", lock=api_stats_lock)
                if logger:
                    logger.warning("worker_failed image=%s error=%s: %s", image_path, type(exc).__name__, exc)
                results.append(
                    {
                        "image_path": str(image_path),
                        "status": "worker_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    with ThreadPoolExecutor(max_workers=explain_workers) as executor:
        pending: Dict[Future, Path] = {}

        progress = tqdm(total=len(image_paths), desc="Create new JSON", dynamic_ncols=True)

        for image_path in image_paths:
            future = executor.submit(
                update_one_record,
                image_path,
                output_dir=output_dir,
                new_json_dir=new_json_dir,
                api_url=args.explain_api_url,
                api_key=args.explain_api_key,
                api_model=args.explain_model,
                timeout=args.explain_timeout,
                max_tokens=args.explain_max_tokens,
                max_api_calls=args.max_api_calls,
                skip_empty_old_traces=args.skip_empty_old_traces,
                dry_run=args.dry_run,
                api_stats=api_stats,
                api_stats_lock=api_stats_lock,
                logger=logger,
            )
            pending[future] = image_path

            if len(pending) >= max_pending_explains:
                collect_finished(pending, block=True)

            collect_finished(pending, block=False)
            progress.update(1)

        progress.close()

        while pending:
            collect_finished(pending, block=True)

    summary_path = Path(args.summary_json) if args.summary_json else output_dir / "update_visible_forgery_traces_new_json_summary.json"
    ensure_dir(summary_path.parent)

    summary = {
        "total_available_images": total_available_images,
        "num_images": len(image_paths),
        "num_results": len(results),
        "new_json_dir": str(new_json_dir),
        "api_stats": api_stats,
        "dry_run": bool(args.dry_run),
        "summary_json": str(summary_path),
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("summary_json=%s", summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
