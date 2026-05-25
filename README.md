# Seg_Attack

Object detection models can be degraded by applying small generator-based perturbations to image or video inputs. This project evaluates two single attacks, Vanish and Fabricate, and the proposed combined method JanusNET.

JanusNET is the name of the proposed attack pipeline in this repository. It is not a separate external architecture or a downloadable third-party model.

## Current Files

| File | Purpose |
|---|---|
| `train.py` | Train Vanish/Fabricate generators and validate each epoch |
| `evaluate_val_4modes.py` | VHRV validation image evaluation for `original`, `fabricate`, `vanish`, `janusnet` |
| `evaluate_video_4modes_vhrv.py` | VHRV video evaluation |
| `evaluate_video_4modes_VisDrone.py` | VisDrone video evaluation with class-colored overlays |
| `evaluate_video_limit_VisDrone.py` | VisDrone video evaluation with Fabricate saturation control |
| `dataset.py` | Dataset loader with split support |
| `model.py` | Generator architecture |
| `utils.py` | Mask and utility functions |

Large files are intentionally excluded from GitHub: datasets, model weights, generated videos, and evaluation outputs.

## Local Assets

| Role | Path |
|---|---|
| VHRV dataset | `vhrv/` |
| VHRV detector | `vhrv_yolo11x.pt` |
| VHRV Vanish generator | `results/vhrv_vresults/G_v_best.pth` |
| VHRV Fabricate generator | `results/vhrv_fresults/G_f_best.pth` |
| VisDrone detector | `best.pt` |
| VisDrone Vanish generator | `results/visdrone_vresults/G_v_best.pth` |
| VisDrone Fabricate generator | `results/visdrone_fresults/G_f_best.pth` |
| FastSAM | `FastSAM-s.pt` |

## Attack Modes

| Mode | Description |
|---|---|
| `original` | Clean input baseline |
| `vanish` | Applies `G_v` to detected/labelled object regions to induce missed detections |
| `fabricate` | Applies `G_f` to non-object regions to induce false detections |
| `janusnet` | Combines Vanish and Fabricate using mask-based spatial separation |

For video evaluation, JanusNET uses the following mask composition:

```python
vanish_mask = FastSAM_mask * BBOX_mask
fabricate_mask = 1 - BBOX_mask
```

This limits the Vanish component to FastSAM object regions inside detector boxes, while Fabricate is applied to non-object background regions. For VisDrone video evaluation, the Fabricate component is scaled to reduce excessive false positives:

```python
JANUSNET_FABRICATE_ALPHA = 0.7
```

## VHRV Image Evaluation

Command:

```powershell
python evaluate_val_4modes.py `
  --data .\vhrv\vhrv.yaml `
  --weights .\vhrv_yolo11x.pt `
  --gv .\results\vhrv_vresults\G_v_best.pth `
  --gf .\results\vhrv_fresults\G_f_best.pth `
  --fastsam-weights .\FastSAM-s.pt `
  --variants original fabricate vanish janusnet `
  --imgsz 640 `
  --batch 1 `
  --outdir .\eval_outputs
```

Saved VHRV validation result:

| Mode | mAP50 | Precision | Recall | F1 | LPIPS | SSIM | PSNR | Speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 0.9551 | 0.9772 | 0.8998 | 0.9369 | 0.0000 | 1.0000 | inf | 0.0200 |
| Fabricate | 0.7223 | 0.9867 | 0.5613 | 0.7155 | 0.1345 | 0.9954 | 37.38 | 0.0254 |
| Vanish | 0.8463 | 0.9455 | 0.7065 | 0.8087 | 0.0153 | 0.9974 | 41.47 | 0.0246 |
| JanusNET | 0.3797 | 0.8583 | 0.2892 | 0.4326 | 0.2962 | 0.9760 | 30.83 | 0.0457 |

## VisDrone Video Evaluation

Standard command:

```powershell
python evaluate_video_4modes_VisDrone.py `
  --source .\Visdrone_Test_1.mp4 `
  --variants original fabricate vanish janusnet `
  --imgsz 640 `
  --save-overlay `
  --outdir .\video_outputs\VisDrone_Color
```

The overlay output displays the original frame on the left and the attacked frame on the right. Bounding boxes use different colors per class, and the top panel shows class-wise detection counts.

Outputs:

| Output | Description |
|---|---|
| `videos/` | Attack-applied videos without detection overlays |
| `overlays/` | Side-by-side original and attacked detection overlays |
| `video_frame_metrics.csv` | Per-frame metrics |
| `video_summary.csv` | Summary metrics |

VisDrone video results with `alpha_f=0.7`:

| Video | Frames | orig_det | adv_det | missed | new | PSNR | Speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Video 1 | 897 | 41.76 | 68.50 | 22.99 | 49.73 | 37.46 | 0.0420 |
| Video 2 | 1429 | 19.94 | 116.94 | 12.58 | 109.58 | 37.48 | 0.0390 |
| Video 3 | 319 | 26.71 | 160.68 | 22.34 | 156.31 | 36.45 | 0.0406 |

Frame-weighted average over the three videos:

| Mode | orig_det | adv_det | missed | new | PSNR | Speed |
|---|---:|---:|---:|---:|---:|---:|
| Original | 28.16 | 28.16 | 0.00 | 0.00 | inf | 0.0090 |
| Fabricate | 28.16 | 256.81 | 4.36 | 233.01 | 36.70 | 0.0167 |
| Vanish | 28.16 | 14.44 | 15.81 | 2.08 | 40.41 | 0.0167 |
| JanusNET | 28.16 | 105.79 | 17.29 | 94.92 | 37.35 | 0.0402 |

## Saturation-Controlled Evaluation

`evaluate_video_limit_VisDrone.py` adds explicit controls for the Fabricate component. It does not remove boxes after detection. Instead, when the number of newly created detections exceeds a predefined limit, it reduces the `G_f` alpha and regenerates the attacked frame.

| Option | Meaning |
|---|---|
| `--fabricate-alpha` | `G_f` strength for Fabricate mode |
| `--janusnet-fabricate-alpha` | `G_f` strength inside JanusNET |
| `--fabricate-area-ratio` | Fraction of non-object area where `G_f` is applied |
| `--max-new-per-frame` | New-detection saturation threshold |
| `--alpha-decay` | Multiplier used to reduce `G_f` alpha when the threshold is exceeded |
| `--min-fabricate-alpha` | Lower bound for adaptive alpha reduction |
| `--limit-attempts` | Maximum adaptive retries per frame |

Recommended saturation-control command:

```powershell
python evaluate_video_limit_VisDrone.py `
  --source .\Visdrone_Test_1.mp4 `
  --variants original fabricate vanish janusnet `
  --imgsz 640 `
  --save-overlay `
  --max-new-per-frame 80 `
  --janusnet-fabricate-alpha 0.7 `
  --fabricate-area-ratio 1.0 `
  --outdir .\video_outputs\VisDrone_Limit_80_Area100
```

Saturation-controlled result on `Visdrone_Test_1.mp4`:

| Mode | orig_det | adv_det | missed | new | PSNR | effective_alpha | limited_ratio | Speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 41.76 | 41.76 | 0.00 | 0.00 | inf | 0.000 | 0.00 | 0.0120 |
| Fabricate | 41.76 | 89.95 | 3.96 | 52.14 | 38.90 | 0.797 | 0.64 | 0.0436 |
| Vanish | 41.76 | 24.64 | 19.34 | 2.22 | 40.78 | 0.000 | 0.00 | 0.0194 |
| JanusNET | 41.76 | 48.19 | 22.62 | 29.05 | 37.69 | 0.665 | 0.22 | 0.0574 |

This controlled setting keeps JanusNET missed detections higher than Vanish while reducing excessive false-positive saturation. The limit was activated on about 22% of JanusNET frames, so it works as a saturation-control mechanism rather than constant post-processing.

## Result Summary

JanusNET reduces detector reliability by combining missed-detection and false-detection attacks. In VHRV image validation, it produced the largest drop in mAP50 and recall among the tested modes. In VisDrone video evaluation, `alpha_f=0.7` kept JanusNET missed detections higher than Vanish while maintaining a substantial number of new detections.

The saturation-controlled evaluation provides an additional way to address excessive Fabricate behavior. Instead of maximizing false positives without constraint, the code can limit false-positive saturation through adaptive alpha reduction while keeping the generator weights fixed.

## Notes

| Metric | Meaning |
|---|---|
| `mAP50` | Detector accuracy at IoU 0.50 |
| `Precision` | Correct detections among predicted detections |
| `Recall` | Detected objects among ground-truth objects |
| `orig_det` | Number of detections on the original frame |
| `adv_det` | Number of detections after attack |
| `missed` | Original detections removed after attack, per frame in video evaluation |
| `new` | New detections created after attack, per frame in video evaluation |
| `effective_alpha` | Actual Fabricate alpha used after adaptive limiting |
| `limited_ratio` | Fraction of frames where the saturation-control logic reduced alpha |
| `PSNR` | Pixel-level similarity; higher is less distorted |
| `SSIM` | Structural similarity; higher is less distorted |
| `LPIPS` | Perceptual difference; lower is less distorted |
| `Linf` | Maximum pixel perturbation magnitude |
