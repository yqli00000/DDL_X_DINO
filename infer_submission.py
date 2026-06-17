from __future__ import annotations

import argparse
import base64
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import io
import json
import logging
import math
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Sequence

import cv2
import numpy as np
import torch
from openai import OpenAI
from PIL import Image
from PIL import ImageFile
from torchvision import transforms

from infer import _load_model_from_checkpoint, choose_device
from tqdm import tqdm

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DDL inference and export submission-style JSON files.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit-images", type=int, default=None, help="Only process the first N sorted images.")
    parser.add_argument("--start-index", type=int, default=None, help="Start offset in the sorted image list, inclusive.")
    parser.add_argument("--end-index", type=int, default=None, help="End offset in the sorted image list, exclusive.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fake-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-box-area", type=int, default=16)
    parser.add_argument("--save-mask-png", action="store_true")
    parser.add_argument("--explain-api-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--explain-api-key", default=None)
    parser.add_argument("--explain-model", default="qwen3.6-plus")
    parser.add_argument("--explain-timeout", type=int, default=60)
    parser.add_argument("--explain-max-tokens", type=int, default=80)
    parser.add_argument("--explain-workers", type=int, default=1, help="Number of concurrent explain API calls.")
    parser.add_argument("--max-api-calls", type=int, default=None, help="Stop calling the explain API after this many calls.")
    parser.add_argument("--log-file", default=None, help="Write runtime logs here. Defaults to output-dir/infer_submission.log.")
    parser.add_argument("--summary-json", default=None, help="Write run summary here. Defaults to output-dir/infer_summary.json.")
    parser.add_argument("--score-jsonl", default=None, help="Write per-image logits/probabilities here. Defaults to output-dir/infer_scores.jsonl.")
    parser.add_argument("--reuse-existing-traces", action="store_true")
    parser.add_argument("--backbone-path", default=None, help="Override DINOv3 backbone path saved in checkpoint.")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_logger(output_dir: str | Path, log_file: str | None) -> logging.Logger:
    logger = logging.getLogger("infer_submission")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    log_path = Path(log_file) if log_file else Path(output_dir) / "infer_submission.log"
    ensure_dir(log_path.parent)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.info("log_file=%s", log_path)
    return logger


def collect_image_paths(image_path: str | None, image_dir: str | None) -> List[Path]:
    if image_path:
        return [Path(image_path)]
    if not image_dir:
        raise ValueError("One of --image-path or --image-dir must be provided.")

    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")

    image_paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise ValueError(f"No supported images found in {root}")
    return image_paths


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


def build_inference_transform(image_size: int) -> transforms.Compose:
    """
    Inference uses full-image resize to 512x512 by default.

    Note:
    - This avoids center crop, so no image region is discarded at test time.
    - It is slightly different from the fixed validation preprocessing used in training.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def classify_result(fake_probability: float, fake_threshold: float) -> str:
    """Main function 1: decide whether the image is real or fake."""
    return "fake" if float(fake_probability) >= float(fake_threshold) else "real"


def probability_to_logit(probability: float, eps: float = 1e-6) -> float:
    probability = min(1.0 - eps, max(eps, float(probability)))
    return float(math.log(probability / (1.0 - probability)))


def restore_mask_to_original_size(pred_mask_prob: np.ndarray, original_width: int, original_height: int) -> np.ndarray:
    """Resize the predicted mask back to the original image size."""
    return cv2.resize(
        np.squeeze(pred_mask_prob).astype(np.float32),
        (int(original_width), int(original_height)),
        interpolation=cv2.INTER_LINEAR,
    )


def normalize_box_to_submission(box: Sequence[int], original_width: int, original_height: int) -> List[int]:
    """
    Convert [x1, y1, x2, y2] from original-image coordinates into the submission scale.

    Submission rule:
    x' = round(x / W * 1000)
    y' = round(y / H * 1000)
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    return [
        int(round(x1 / max(1, original_width) * 1000)),
        int(round(y1 / max(1, original_height) * 1000)),
        int(round(x2 / max(1, original_width) * 1000)),
        int(round(y2 / max(1, original_height) * 1000)),
    ]


def compute_bounding_boxes(
    pred_mask_prob: np.ndarray,
    prediction: str,
    mask_threshold: float,
    min_box_area: int,
    original_width: int,
    original_height: int,
) -> List[List[int]]:
    """
    Main function 2:
    1. Restore the predicted mask to the original image size.
    2. Threshold the restored mask.
    3. Extract connected components.
    4. Convert each region into [x1, y1, x2, y2] and normalize to submission coordinates.
    """
    if prediction == "real":
        return []

    restored_mask_prob = restore_mask_to_original_size(pred_mask_prob, original_width, original_height)
    binary_mask = (restored_mask_prob >= float(mask_threshold)).astype(np.uint8)
    if binary_mask.sum() == 0:
        return []

    num_components, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    boxes: List[List[int]] = []
    for component_idx in range(1, num_components):
        x, y, w, h, area = stats[component_idx]
        if int(area) < int(min_box_area):
            continue
        boxes.append(
            normalize_box_to_submission(
                [int(x), int(y), int(x + w - 1), int(y + h - 1)],
                original_width,
                original_height,
            )
        )
    return boxes


def _build_fallback_traces(prediction: str, fake_probability: float, boxes: Sequence[Sequence[int]]) -> str:
    if prediction == "real":
        return append_summary(
            "None. The image does not show obvious visible forgery traces. "
            "The lighting, edges, texture continuity, and overall physical consistency appear coherent.",
            prediction,
        )
    if not boxes:
        return append_summary(
            f"The image is predicted as fake with confidence {fake_probability:.4f}, "
            "but no stable localized visible forgery traces were extracted.",
            prediction,
        )
    return append_summary(
        f"The image is predicted as fake with confidence {fake_probability:.4f}. "
        f"Visible forgery traces are associated with {len(boxes)} localized suspicious region(s).",
        prediction,
    )


def strip_summary(text: str) -> str:
    return re.sub(r"\s*Summary:\s*This image has(?: not)? been tampered with\.?\s*$", "", text.strip(), flags=re.IGNORECASE)


def append_summary(text: str, prediction: str) -> str:
    text = strip_summary(text)
    if prediction == "fake":
        summary = "Summary: This image has been tampered with."
    else:
        summary = "Summary: This image has not been tampered with."
    return f"{text.rstrip()}\n\n{summary}"


def build_overlay_image_bytes(image_bytes: bytes, mask_bytes: bytes, *, alpha: int = 95) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L").resize(image.size, resample=Image.Resampling.NEAREST)
    mask_np = np.array(mask)
    overlay_alpha = Image.fromarray(((mask_np > 0).astype(np.uint8) * int(alpha)).astype(np.uint8), mode="L")
    overlay = Image.new("RGBA", image.size, (235, 60, 45, 0))
    overlay.putalpha(overlay_alpha)
    blended = Image.alpha_composite(image, overlay).convert("RGB")
    buffer = io.BytesIO()
    blended.save(buffer, format="PNG")
    return buffer.getvalue()


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


def get_stat(api_stats: MutableMapping[str, int] | None, key: str, lock: threading.Lock | None = None) -> int:
    if api_stats is None:
        return 0
    if lock is None:
        return api_stats.get(key, 0)
    with lock:
        return api_stats.get(key, 0)


def generate_visible_forgery_traces(
    image_bytes: bytes,
    image_name: str,
    mask_bytes: bytes,
    mask_name: str,
    prediction: str,
    fake_probability: float,
    boxes: Sequence[Sequence[int]],
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
    Main function 3: call an external API to generate the text explanation.

    Note:
    - If `api_url` is not provided, the function falls back to a local template.
    - If the API request fails, inference still continues and uses the fallback text.
    - The request payload is generic JSON and can be adapted to your final API schema.
    """
    if not api_url:
        if logger:
            logger.info("skip_api image=%s reason=no_api_url", image_name)
        return _build_fallback_traces(prediction, fake_probability, boxes)
    if prediction == 'real':
        increment_stat(api_stats, "skipped_real", lock=api_stats_lock)
        return _build_fallback_traces(prediction, fake_probability, boxes)
    if max_api_calls is not None and get_stat(api_stats, "api_calls", lock=api_stats_lock) >= max_api_calls:
        if logger:
            logger.warning("skip_api image=%s reason=max_api_calls limit=%s", image_name, max_api_calls)
        return _build_fallback_traces(prediction, fake_probability, boxes)
    else:
        try:
            increment_stat(api_stats, "api_calls", lock=api_stats_lock)
            if logger:
                logger.info("call_api image=%s model=%s boxes=%d", image_name, api_model, len(boxes))
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
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": encode_data_url(image_bytes, guess_mime_type(image_name)),
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": encode_data_url(
                                        build_overlay_image_bytes(image_bytes, mask_bytes),
                                        "image/png",
                                    ),
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "You are writing an official image-forensics annotation.\n\n"
                                    "The first image is the original image. The second image is the same image with a "
                                    "semi-transparent red overlay marking predicted suspicious regions. Use the overlay "
                                    "only to focus your forensic analysis; do not mention the overlay or mask in the final answer.\n\n"
                                    f"The classification result is {prediction}. "
                                    f"The predicted bounding boxes are {json.dumps([list(box) for box in boxes])} in normalized 0-1000 coordinates.\n\n"
                                    "Write a detailed visible-evidence explanation in the style of a forensic dataset annotation.\n\n"
                                    "Requirements:\n"
                                    "- Begin with one natural paragraph describing the image content: subject, visible attributes, "
                                    "clothing or accessories, background, camera setting, and lighting.\n"
                                    "- Then analyze visible forensic evidence using markdown bullet points.\n"
                                    "- Each bullet point must start with a bold, specific forensic heading written in your own words, "
                                    "tailored to the actual visible evidence rather than copied from a fixed checklist.\n"
                                    "- For fake images, focus on localized visible artifacts such as skin texture discontinuity, "
                                    "lighting or shadow mismatch, abnormal eye reflections, unnatural facial geometry, boundary blending, "
                                    "resolution mismatch, color inconsistency, or texture artifacts.\n"
                                    "- For real images, explain why visible evidence appears consistent, including natural lighting, "
                                    "coherent shadows, organic edges, realistic texture, physical plausibility, depth of field, and absence of copy-paste artifacts.\n"
                                    "- Refer to concrete visible regions when possible, such as eyes, eyelids, nose bridge, lips, hairline, cheek, jawline, clothing edges, background text, or object boundaries.\n"
                                    "- Keep the tone observational and technical, not conversational.\n"
                                    "- Do not mention confidence scores, model thresholds, algorithms, bounding box coordinates, or that a mask/overlay was provided.\n"
                                    "- Do not include a final Summary sentence; it will be appended separately."
                                ),
                            },
                        ],
                    }
                ],
                max_tokens=int(max_tokens),
            )
        except Exception as exc:
            increment_stat(api_stats, "api_failed", lock=api_stats_lock)
            if logger:
                logger.warning("api_failed image=%s error=%s: %s", image_name, type(exc).__name__, exc)
            return _build_fallback_traces(prediction, fake_probability, boxes)

        if api_stats is not None:
            increment_stat(api_stats, "api_succeeded", lock=api_stats_lock)
            usage = getattr(completion, "usage", None)
            for usage_key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(usage, usage_key, None) if usage is not None else None
                if isinstance(value, int):
                    increment_stat(api_stats, usage_key, amount=value, lock=api_stats_lock)
            if logger and usage is not None:
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
    return _build_fallback_traces(prediction, fake_probability, boxes)


def build_json_record(boxes: Sequence[Sequence[int]], traces: str, prediction: str) -> Dict[str, object]:
    return {
        "Bounding boxes": [list(box) for box in boxes],
        "Visible forgery traces": traces,
        "Classification result": prediction,
    }


def write_json_record(json_path: Path, boxes: Sequence[Sequence[int]], traces: str, prediction: str) -> None:
    record = build_json_record(boxes, traces, prediction)
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def load_existing_traces(json_path: Path) -> str | None:
    if not json_path.exists():
        return None
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    traces = payload.get("Visible forgery traces")
    if isinstance(traces, str) and traces.strip():
        return traces.strip()
    return None


def build_output_stem(name: str) -> str:
    """Use the source image filename stem for exported JSON and optional mask files."""
    safe = Path(name).stem.strip()
    return safe if safe else "sample"


def run_single_image_inference(
    model,
    device: torch.device,
    image_path: str | Path,
    *,
    output_dir: str | Path,
    image_size: int = 512,
    fake_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    min_box_area: int = 16,
    explain_api_url: str | None = None,
    explain_api_key: str | None = None,
    explain_model: str = "qwen3.6-plus",
    explain_timeout: int = 60,
    explain_max_tokens: int = 80,
    max_api_calls: int | None = None,
    save_mask_png: bool = True,
    reuse_existing_traces: bool = False,
    defer_traces: bool = False,
    api_stats: MutableMapping[str, int] | None = None,
    api_stats_lock: threading.Lock | None = None,
    logger: logging.Logger | None = None,
) -> Dict[str, object]:
    """
    Read one image, run inference, restore the predicted mask to original size,
    save the mask PNG, and return the submission-style record plus file paths.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    json_dir = ensure_dir(output_dir / "json")
    mask_dir = ensure_dir(output_dir / "mask")

    image_transform = build_inference_transform(image_size)
    image_bytes = image_path.read_bytes()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_width, original_height = image.size

    with torch.no_grad():
        images = image_transform(image).unsqueeze(0).to(device)
        outputs = model(images)

        fake_logit = float(outputs["logits"].detach().cpu().numpy().reshape(-1)[0])
        fake_probability = float(torch.sigmoid(outputs["logits"]).detach().cpu().numpy().reshape(-1)[0])
        pred_mask_prob = outputs["pred_mask"].detach().cpu().numpy()[0]
        prediction = classify_result(fake_probability, fake_threshold)
        boxes = compute_bounding_boxes(
            pred_mask_prob,
            prediction,
            mask_threshold,
            min_box_area,
            original_width,
            original_height,
        )
    output_stem = build_output_stem(image_path.name)
    json_path = json_dir / f"{output_stem}.json"
    restored_mask_prob = restore_mask_to_original_size(pred_mask_prob, original_width, original_height)
    binary_mask = (restored_mask_prob >= mask_threshold).astype(np.uint8) * 255
    if prediction == "real":
        binary_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    mask_path = mask_dir / f"{output_stem}.png"
    mask_image = Image.fromarray(binary_mask)
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    mask_bytes = buffer.getvalue()
    if save_mask_png:
        mask_path.write_bytes(mask_bytes)

    traces = load_existing_traces(json_path) if reuse_existing_traces else None
    if traces is None:
        if defer_traces and prediction == "fake":
            return {
                "image_path": str(image_path),
                "json_path": str(json_path),
                "mask_path": str(mask_path),
                "prediction": prediction,
                "fake_confidence": fake_probability,
                "fake_logit": fake_logit,
                "bounding_boxes": boxes,
                "visible_forgery_traces": None,
                "_trace_payload": {
                    "image_bytes": image_bytes,
                    "image_name": image_path.name,
                    "mask_bytes": mask_bytes,
                    "mask_name": mask_path.name,
                    "prediction": prediction,
                    "fake_probability": fake_probability,
                    "boxes": boxes,
                    "json_path": str(json_path),
                },
            }
        traces = generate_visible_forgery_traces(
            image_bytes,
            image_path.name,
            mask_bytes,
            mask_path.name,
            prediction,
            fake_probability,
            boxes,
            api_url=explain_api_url,
            api_key=explain_api_key,
            api_model=explain_model,
            timeout=explain_timeout,
            max_tokens=explain_max_tokens,
            max_api_calls=max_api_calls,
            api_stats=api_stats,
            api_stats_lock=api_stats_lock,
            logger=logger,
        )
    write_json_record(json_path, boxes, traces, prediction)

    return {
        "image_path": str(image_path),
        "json_path": str(json_path),
        "mask_path": str(mask_path),
        "prediction": prediction,
        "fake_confidence": fake_probability,
        "fake_logit": fake_logit,
        "bounding_boxes": boxes,
        "visible_forgery_traces": traces,
    }


def run_batch_image_inference(
    model,
    device: torch.device,
    image_paths: Sequence[str | Path],
    *,
    output_dir: str | Path,
    image_size: int = 512,
    fake_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    min_box_area: int = 16,
    explain_api_url: str | None = None,
    explain_api_key: str | None = None,
    explain_model: str = "qwen3.6-plus",
    explain_timeout: int = 60,
    explain_max_tokens: int = 80,
    max_api_calls: int | None = None,
    save_mask_png: bool = True,
    reuse_existing_traces: bool = False,
    defer_traces: bool = False,
    api_stats: MutableMapping[str, int] | None = None,
    api_stats_lock: threading.Lock | None = None,
    logger: logging.Logger | None = None,
) -> List[Dict[str, object]]:
    output_dir = Path(output_dir)
    json_dir = ensure_dir(output_dir / "json")
    mask_dir = ensure_dir(output_dir / "mask")
    image_transform = build_inference_transform(image_size)

    samples: List[Dict[str, Any]] = []
    tensors = []
    bad_results: List[Dict[str, object]] = []
    for image_path_like in image_paths:
        image_path = Path(image_path_like)
        try:
            image_bytes = image_path.read_bytes()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            message = f"[bad_image] path={image_path} error={type(exc).__name__}: {exc}"
            tqdm.write(message)
            if logger:
                logger.warning(message)
            output_stem = build_output_stem(image_path.name)
            json_path = json_dir / f"{output_stem}.json"
            mask_path = mask_dir / f"{output_stem}.png"
            traces = "Image could not be decoded; no visible forgery traces are available."
            write_json_record(json_path, [], traces, "real")
            bad_results.append(
                {
                    "image_path": str(image_path),
                    "json_path": str(json_path),
                    "mask_path": str(mask_path),
                    "prediction": "real",
                    "fake_confidence": 0.0,
                    "fake_logit": probability_to_logit(0.0),
                    "bounding_boxes": [],
                    "visible_forgery_traces": traces,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        original_width, original_height = image.size
        samples.append(
            {
                "image_path": image_path,
                "image_bytes": image_bytes,
                "original_width": original_width,
                "original_height": original_height,
            }
        )
        tensors.append(image_transform(image))

    if not tensors:
        return bad_results

    with torch.no_grad():
        images = torch.stack(tensors, dim=0).to(device, non_blocking=True)
        outputs = model(images)
        fake_logits = outputs["logits"].detach().cpu().numpy().reshape(-1)
        fake_probabilities = torch.sigmoid(outputs["logits"]).detach().cpu().numpy().reshape(-1)
        pred_mask_probs = outputs["pred_mask"].detach().cpu().numpy()

    results: List[Dict[str, object]] = bad_results
    for sample_idx, sample in enumerate(samples):
        image_path = sample["image_path"]
        image_bytes = sample["image_bytes"]
        original_width = sample["original_width"]
        original_height = sample["original_height"]
        fake_logit = float(fake_logits[sample_idx])
        fake_probability = float(fake_probabilities[sample_idx])
        pred_mask_prob = pred_mask_probs[sample_idx]
        prediction = classify_result(fake_probability, fake_threshold)
        boxes = compute_bounding_boxes(
            pred_mask_prob,
            prediction,
            mask_threshold,
            min_box_area,
            original_width,
            original_height,
        )

        output_stem = build_output_stem(image_path.name)
        json_path = json_dir / f"{output_stem}.json"
        restored_mask_prob = restore_mask_to_original_size(pred_mask_prob, original_width, original_height)
        binary_mask = (restored_mask_prob >= mask_threshold).astype(np.uint8) * 255
        if prediction == "real":
            binary_mask = np.zeros_like(binary_mask, dtype=np.uint8)

        mask_path = mask_dir / f"{output_stem}.png"
        mask_image = Image.fromarray(binary_mask)
        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        mask_bytes = buffer.getvalue()
        if save_mask_png:
            mask_path.write_bytes(mask_bytes)

        traces = load_existing_traces(json_path) if reuse_existing_traces else None
        if traces is None:
            if defer_traces and prediction == "fake":
                results.append(
                    {
                        "image_path": str(image_path),
                        "json_path": str(json_path),
                        "mask_path": str(mask_path),
                        "prediction": prediction,
                        "fake_confidence": fake_probability,
                        "fake_logit": fake_logit,
                        "bounding_boxes": boxes,
                        "visible_forgery_traces": None,
                        "_trace_payload": {
                            "image_bytes": image_bytes,
                            "image_name": image_path.name,
                            "mask_bytes": mask_bytes,
                            "mask_name": mask_path.name,
                            "prediction": prediction,
                            "fake_probability": fake_probability,
                            "boxes": boxes,
                            "json_path": str(json_path),
                        },
                    }
                )
                continue
            traces = generate_visible_forgery_traces(
                image_bytes,
                image_path.name,
                mask_bytes,
                mask_path.name,
                prediction,
                fake_probability,
                boxes,
                api_url=explain_api_url,
                api_key=explain_api_key,
                api_model=explain_model,
                timeout=explain_timeout,
                max_tokens=explain_max_tokens,
                max_api_calls=max_api_calls,
                api_stats=api_stats,
                api_stats_lock=api_stats_lock,
                logger=logger,
            )
        write_json_record(json_path, boxes, traces, prediction)
        results.append(
            {
                "image_path": str(image_path),
                "json_path": str(json_path),
                "mask_path": str(mask_path),
                "prediction": prediction,
                "fake_confidence": fake_probability,
                "fake_logit": fake_logit,
                "bounding_boxes": boxes,
                "visible_forgery_traces": traces,
            }
        )
    return results


def resolve_deferred_traces(payload: Dict[str, Any], explain_args: Dict[str, Any]) -> Dict[str, object]:
    traces = generate_visible_forgery_traces(
        payload["image_bytes"],
        payload["image_name"],
        payload["mask_bytes"],
        payload["mask_name"],
        payload["prediction"],
        payload["fake_probability"],
        payload["boxes"],
        **explain_args,
    )
    json_path = Path(payload["json_path"])
    write_json_record(json_path, payload["boxes"], traces, payload["prediction"])
    return {
        "json_path": str(json_path),
        "visible_forgery_traces": traces,
    }


def main() -> None:
    args = parse_args()
    logger = configure_logger(args.output_dir, args.log_file)
    device = choose_device(args.device)
    model = _load_model_from_checkpoint(args.checkpoint, device,backbone_path=args.backbone_path,)
    model.eval()

    image_paths = collect_image_paths(args.image_path, args.image_dir)
    total_available_images = len(image_paths)
    if args.start_index is not None or args.end_index is not None:
        start_index = max(0, int(args.start_index or 0))
        end_index = None if args.end_index is None else max(start_index, int(args.end_index))
        image_paths = image_paths[start_index:end_index]
    elif args.limit_images is not None:
        image_paths = image_paths[max(0, int(args.limit_images)):]
    results = []
    api_stats: Dict[str, int] = {}
    api_stats_lock = threading.Lock()
    explain_workers = max(1, int(args.explain_workers))
    max_pending_explains = max(1, explain_workers * 4)
    explain_args = {
        "api_url": args.explain_api_url,
        "api_key": args.explain_api_key,
        "api_model": args.explain_model,
        "timeout": args.explain_timeout,
        "max_tokens": args.explain_max_tokens,
        "max_api_calls": args.max_api_calls,
        "api_stats": api_stats,
        "api_stats_lock": api_stats_lock,
        "logger": logger,
    }

    def collect_finished_explains(
        pending: MutableMapping[Future, Dict[str, object]],
        *,
        block: bool = False,
    ) -> None:
        if not pending:
            return
        timeout = None if block else 0
        done, _ = wait(pending.keys(), timeout=timeout, return_when=FIRST_COMPLETED)
        for future in done:
            result = pending.pop(future)
            trace_result = future.result()
            result["visible_forgery_traces"] = trace_result["visible_forgery_traces"]
            results.append(result)

    batch_size = max(1, int(args.batch_size))

    with ThreadPoolExecutor(max_workers=explain_workers) as explain_executor:
        pending_explains: Dict[Future, Dict[str, object]] = {}
        progress = tqdm(total=len(image_paths), desc="Infer", dynamic_ncols=True)
        for batch_start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[batch_start : batch_start + batch_size]
            batch_results = run_batch_image_inference(
                model,
                device,
                batch_paths,
                output_dir=args.output_dir,
                image_size=args.image_size,
                fake_threshold=args.fake_threshold,
                mask_threshold=args.mask_threshold,
                min_box_area=args.min_box_area,
                explain_api_url=args.explain_api_url,
                explain_api_key=args.explain_api_key,
                explain_model=args.explain_model,
                explain_timeout=args.explain_timeout,
                explain_max_tokens=args.explain_max_tokens,
                max_api_calls=args.max_api_calls,
                save_mask_png=args.save_mask_png,
                reuse_existing_traces=args.reuse_existing_traces,
                defer_traces=explain_workers > 1,
                api_stats=api_stats,
                api_stats_lock=api_stats_lock,
                logger=logger,
            )
            for result in batch_results:
                payload = result.pop("_trace_payload", None)
                if payload is None:
                    results.append(result)
                else:
                    future = explain_executor.submit(resolve_deferred_traces, payload, explain_args)
                    pending_explains[future] = result
                    if len(pending_explains) >= max_pending_explains:
                        collect_finished_explains(pending_explains, block=True)
            collect_finished_explains(pending_explains, block=False)
            progress.update(len(batch_paths))
        progress.close()

        while pending_explains:
            collect_finished_explains(pending_explains, block=True)

    summary_path = Path(args.summary_json) if args.summary_json else Path(args.output_dir) / "infer_summary.json"
    ensure_dir(summary_path.parent)
    num_fake = sum(1 for result in results if result["prediction"] == "fake")
    num_real = sum(1 for result in results if result["prediction"] == "real")
    fake_rate = num_fake / len(results) if results else 0.0

    summary = {
        "total_available_images": total_available_images,
        "limit_images": args.limit_images,
        "num_images": len(image_paths),
        "num_results": len(results),
        "num_fake": num_fake,
        "num_real": num_real,
        "sample_fake_rate": fake_rate,
        "fake_threshold": args.fake_threshold,
        "mask_threshold": args.mask_threshold,
        "min_box_area": args.min_box_area,
        "api_stats": api_stats,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("summary_json=%s", summary_path)

    score_path = Path(args.score_jsonl) if args.score_jsonl else Path(args.output_dir) / "infer_scores.jsonl"
    ensure_dir(score_path.parent)
    with score_path.open("w", encoding="utf-8") as f:
        for result in sorted(results, key=lambda item: str(item.get("image_path", ""))):
            record = {
                "image_path": result.get("image_path"),
                "json_path": result.get("json_path"),
                "prediction": result.get("prediction"),
                "fake_probability": result.get("fake_confidence"),
                "fake_logit": result.get("fake_logit"),
                "num_boxes": len(result.get("bounding_boxes", []) or []),
                "fake_threshold": args.fake_threshold,
                "mask_threshold": args.mask_threshold,
                "min_box_area": args.min_box_area,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("score_jsonl=%s", score_path)


if __name__ == "__main__":
    main()
