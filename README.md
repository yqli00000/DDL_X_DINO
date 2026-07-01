# PATE-Forensics Training and Inference Reproduction

`PATE-Forensics` is the implementation code for training and inference in the [DDL competition](https://ai-safety-workshop-ijcai2026.github.io/Track3.html). It contains source code, the main training configuration, inference scripts, dependency files, and reserved locations for the final model checkpoint, DINOv3 backbone, and datasets.

## Contents

- `train.py`: Lightning training entrypoint.
- `infer.py`: shared checkpoint/model loading utilities used by `infer_submission.py`.
- `infer_submission.py`: required submission inference script.
- `update_json_traces.py`: required second-stage JSON trace refinement script.
- `data/`, `engine/`, `networks/`, `utils/`: source modules required for training and inference.
- `cfgs/train/gps_dino_mask_mixed_phase1_phase2_wo_maskloss.yaml`: main reference training configuration.
- `weights/`: reserved checkpoint location.
- `datasets/`: reserved dataset location.

## Environment

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The project was developed with Python 3.12, PyTorch, TorchVision, Lightning, OmegaConf/Hydra, OpenCV, Pillow, NumPy, scikit-learn, tqdm, rich, and the OpenAI-compatible client library.

## Required External Files

### Weights

For training and inference, the DINOv3 backbone should be placed at:

```text
weights/dinov3-l16
```
The DINOv3 backbone version is `facebook/dinov3-vitl16-pretrain-lvd1689m`.

The DINOv3 backbone weights can be downloaded from [ModelScope](https://www.modelscope.cn/models/facebook/dinov3-vitl16-pretrain-lvd1689m).

For inference, put the checkpoint in the package-local location:

```text
weights/model_best.ckpt
```

The `model_best.ckpt` checkpoint can be downloaded from [Google Drive](https://drive.google.com/file/d/12xMXHFRo6fcOEh0vfw2yhyM5RD53nbYY/view?usp=sharing).

If the DINOv3 path saved in the checkpoint differs on the target machine, use `--backbone-path` during submission inference.


### Dataset

The training config expects the dataset under:

```text
datasets/phase1/track1_inner/track1/train
datasets/phase1/track1_inner/track1/valid
datasets/phase2/track1/test
```

Edit these relative paths in `cfgs/train/gps_dino_mask_mixed_phase1_phase2_wo_maskloss.yaml` if your data layout is different.

## Training

For full training reproduction, run the provided GPU training script from this folder:

```bash
bash run_gps_dino_wandb.sh
```

The script uses `cfgs/train/gps_dino_mask_mixed_phase1_phase2_wo_maskloss.yaml` and launches distributed GPU training through `torch.distributed.run`. Edit `NUM_GPUS`, `CUDA_VISIBLE_DEVICES`, W&B settings, and any resume/pretrained checkpoint arguments in the script to match the target machine.

You can also run the training entrypoint directly:

```bash
python train.py --cfg cfgs/train/gps_dino_mask_mixed_phase1_phase2_wo_maskloss.yaml --logdir gps_dino_mask_mixed_phase1_phase2_reproduce
```

To resume from a checkpoint, example command is:

```bash
python train.py --cfg cfgs/train/gps_dino_mask_mixed_phase1_phase2_wo_maskloss.yaml --resume weights/model_best.ckpt --logdir gps_dino_mask_mixed_phase1_phase2_resume
```

## Submission Inference

Run image-folder inference and export JSON files with the provided script:

```bash
bash infer_submission.sh
```

Outputs are written under:

```text
outputs/test/json/
outputs/test/mask/
outputs/test/infer_summary.json
outputs/test/infer_scores.jsonl
```

`infer_submission.py` can call an OpenAI-compatible vision API to generate detailed visible-trace text. If no valid API key is available, it falls back to local template text. For DashScope-compatible usage, set:

```bash
export DASHSCOPE_API_KEY=your_key
```

`infer_submission.py` supports calling the API directly during inference. To enable API-based trace generation, set `--explain-api-url`, provide `--explain-api-key` or `DASHSCOPE_API_KEY`, and control the number of API requests with parameters such as `--max-api-calls` and `--explain-workers`. This is convenient for small batches, but it can be time-consuming for large submissions because model inference and API calls run in the same workflow.

To reduce total inference time, the recommended workflow is to first run `infer_submission.py` with API calls disabled or limited, which quickly produces prediction JSON files and mask PNG files. Then run `update_json_traces.py` as a second-stage refinement step to update the `Visible forgery traces` field in the generated JSON files without rerunning model inference.

## Second-Stage Trace Refinement

After `infer_submission.py`, optionally refine the `Visible forgery traces` field without overwriting the original JSON folder:

```bash
bash update_submission_new.sh
```

This script reads `outputs/test/json/` and `outputs/test/mask/`, then writes refined JSON files to `json_api_refined/`.
