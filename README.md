# Seg_Attack

Seg_Attack evaluates generator-based adversarial perturbations against object detectors.
It includes two single attacks and the proposed combined pipeline, JanusNET.

JanusNET is the method name used in this repository. It is not a separate third-party architecture.

## Project Structure

| Path | Purpose |
|---|---|
| `train.py` | Train and validate Vanish/Fabricate generators |
| `model.py` | Lightweight U-Net based `MobileUNetGenerator` |
| `evaluate_val_4modes.py` | Image evaluation for `original`, `fabricate`, `vanish`, `janusnet` |
| `evaluate_video_4modes_vhrv.py` | VHRV video evaluation |
| `evaluate_video_4modes_VisDrone.py` | VisDrone video evaluation |
| `evaluate_video_limit_VisDrone.py` | VisDrone video evaluation with false-positive saturation control |
| `evaluate_video_targeted_vanish_VisDrone.py` | Interactive targeted Vanish demo |
| `datasets/` | Local VHRV and VisDrone datasets |
| `weights/` | Detector and FastSAM weights |
| `demo_inputs/` | Local test videos |
| `outputs/image_eval/` | Image evaluation outputs |
| `outputs/video_eval/` | Video evaluation outputs |
| `outputs/training/` | Generator checkpoints and training visualizations |
| `outputs/yolo_runs/` | Ultralytics validation plots |
| `outputs/graphs/` | Poster and JanusNET graphs |
| `outputs/_temporary/` | Smoke-test outputs and temporary files |
| `tools/` | Graph generation and setup utilities |

Large local assets and generated outputs are intentionally excluded from GitHub.

## Local Assets

| Role | Path |
|---|---|
| VHRV dataset | `datasets/vhrv/` |
| VisDrone train dataset | `datasets/VisDrone2019-DET-train/` |
| VisDrone val dataset | `datasets/VisDrone2019-DET-val/` |
| VHRV detector | `weights/detectors/vhrv_yolo11x.pt` |
| VisDrone detector | `weights/detectors/visdrone_best.pt` |
| FastSAM | `weights/segmentation/FastSAM-s.pt` |
| VHRV Vanish generator | `outputs/training/generators/vhrv_vresults/G_v_best.pth` |
| VHRV Fabricate generator | `outputs/training/generators/vhrv_fresults/G_f_best.pth` |
| VisDrone Vanish generator | `outputs/training/generators/visdrone_vresults/G_v_best.pth` |
| VisDrone Fabricate generator | `outputs/training/generators/visdrone_fresults/G_f_best.pth` |

## Attack Modes

| Mode | Description |
|---|---|
| `original` | Clean input baseline |
| `vanish` | Apply `G_v` to object regions to induce missed detections |
| `fabricate` | Apply `G_f` to non-object regions to induce false detections |
| `janusnet` | Combine Vanish and Fabricate using spatially separated masks |

Current JanusNET mask composition:

```python
vanish_mask = FastSAM_mask * BBOX_mask
fabricate_mask = 1 - BBOX_mask
```

## Image Evaluation

VHRV:

```powershell
python .\evaluate_val_4modes.py --data .\datasets\vhrv\vhrv.yaml --weights .\weights\detectors\vhrv_yolo11x.pt --gv .\outputs\training\generators\vhrv_vresults\G_v_best.pth --gf .\outputs\training\generators\vhrv_fresults\G_f_best.pth --fastsam-weights .\weights\segmentation\FastSAM-s.pt --variants original fabricate vanish janusnet --imgsz 640 --batch 1 --outdir .\outputs\image_eval\vhrv
```

VisDrone:

```powershell
python .\evaluate_val_4modes.py --data .\datasets\visdrone.yaml --weights .\weights\detectors\visdrone_best.pt --gv .\outputs\training\generators\visdrone_vresults\G_v_best.pth --gf .\outputs\training\generators\visdrone_fresults\G_f_best.pth --fastsam-weights .\weights\segmentation\FastSAM-s.pt --variants original fabricate vanish janusnet --imgsz 640 --batch 1 --outdir .\outputs\image_eval\visdrone
```

## Video Evaluation

VisDrone four-mode evaluation:

```powershell
python .\evaluate_video_4modes_VisDrone.py --source .\demo_inputs\visdrone\Visdrone_Test_1.mp4 --variants original fabricate vanish janusnet --imgsz 640 --save-overlay
```

Interactive targeted Vanish demo:

```powershell
python .\evaluate_video_targeted_vanish_VisDrone.py --source .\demo_inputs\visdrone\Visdrone_Test_3.mp4 --target car --interactive --save-overlay --vanish-steps 3 --residual-mask bbox --outdir .\outputs\video_eval\visdrone\targeted_vanish
```

`--vanish-steps 1` keeps the original single-pass Vanish behavior.
`--vanish-steps 2` or higher repeats Vanish on remaining target detections in the same frame.
`--residual-mask bbox` uses remaining target bounding boxes for the repeated steps; `fastsam` is tighter but slower.

Targeted demo controls:

| Key | Action |
|---|---|
| `0` - `9` | Select a VisDrone class |
| `A` | Toggle attack ON/OFF |
| `Space` | Pause/resume |
| `Q`, `ESC` | Quit |

## Graph Generation

```powershell
python .\tools\generate_janusnet_graphs.py
python .\tools\generate_poster_graphs.py
```

## Metrics

| Metric | Meaning |
|---|---|
| `mAP50` | Detector performance at IoU 0.50 |
| `Precision` | Correct detections among predictions |
| `Recall` | Detected objects among ground-truth objects |
| `missed` | Original detections removed after attack |
| `new` | New detections created after attack |
| `PSNR` | Pixel-level similarity; higher is less distorted |
| `SSIM` | Structural similarity; higher is less distorted |
| `LPIPS` | Perceptual difference; lower is less distorted |
| `Linf` | Maximum pixel perturbation magnitude |
