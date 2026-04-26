import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
import yaml

try:
    import lpips
except ImportError:
    lpips = None


# -----------------------------------------------------------------------------
# 검증 전용 스크립트
# - 목적: generate_variant_images.py로 만든 variant 데이터셋을 detector로 평가
# - 계산 지표: mAP50 / Precision / Recall / F1 / LPIPS / SSIM / PSNR / Speed
# - 출력: eval_variants/variant_summary.csv
# -----------------------------------------------------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    # CLI 입력 인자 정의
    p = argparse.ArgumentParser(description="Evaluate generated variant datasets and summarize metrics.")
    p.add_argument("--data", required=True, help="Original dataset yaml (used to infer split)")
    p.add_argument("--weights", required=True, help="Detector weights (.pt)")
    p.add_argument("--outdir", default="eval_variants", help="Output dir used by generate_variant_images.py")
    p.add_argument("--variants", nargs="+", default=["original", "fabricate", "vanish", "janusnet"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--device", default="", help="cuda:0 or cpu (empty = auto)")
    p.add_argument("--project", default="runs/variant_eval")
    p.add_argument("--name", default="exp")
    return p.parse_args()


def resolve_split(data_yaml: Path):
    # 원본 yaml에서 val/test split 이름 추정
    with open(data_yaml, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    split = "val"
    val_rel = str(d.get("val", "images/val"))
    test_rel = str(d.get("test", ""))
    if "test" in val_rel:
        split = "test"
    elif "val" in val_rel:
        split = "val"
    elif test_rel:
        split = "test"
    return split


def image_to_tensor(path: Path, device):
    # PIL 이미지를 [0,1] 텐서(B, C, H, W)로 변환
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def ssim_batch(x, y, window=11, c1=0.01 ** 2, c2=0.03 ** 2):
    # 간단한 SSIM 계산 (배치 평균)
    pad = window // 2
    mu_x = F.avg_pool2d(x, kernel_size=window, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, kernel_size=window, stride=1, padding=pad)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x2 = F.avg_pool2d(x * x, window, 1, pad) - mu_x2
    sigma_y2 = F.avg_pool2d(y * y, window, 1, pad) - mu_y2
    sigma_xy = F.avg_pool2d(x * y, window, 1, pad) - mu_xy
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12
    )
    return ssim_map.mean().item()


def psnr_batch(x, y):
    # PSNR 계산
    mse = F.mse_loss(x, y).item()
    return 10.0 * np.log10(1.0 / (mse + 1e-10))


def maybe_lpips(device):
    # LPIPS 패키지가 없으면 None 반환 (평가는 계속 가능)
    if lpips is None:
        return None
    return lpips.LPIPS(net="alex").to(device).eval()


def list_images(folder: Path):
    # 폴더 내 지원 확장자 이미지 수집
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])


def run_detector_val(detector_weights, data_yaml, imgsz, batch, device, project, name):
    # Ultralytics val 실행 후 핵심 탐지 지표 반환
    y = YOLO(detector_weights)
    r = y.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        device=device if device else None,
        project=project,
        name=name,
        verbose=False,
    )
    p = float(r.box.mp)
    rc = float(r.box.mr)
    f1 = (2 * p * rc / (p + rc)) if (p + rc) > 0 else 0.0
    map50 = float(r.box.map50)
    speed = r.speed if hasattr(r, "speed") else {}
    ms_img = float(speed.get("preprocess", 0.0) + speed.get("inference", 0.0) + speed.get("postprocess", 0.0))
    return map50, p, rc, f1, ms_img / 1000.0


def compute_perceptual_metrics(original_dir: Path, variant_dir: Path, device):
    # original 이미지와 variant 이미지를 파일명 매칭해
    # LPIPS/SSIM/PSNR 평균 계산
    lpips_model = maybe_lpips(device)
    orig_imgs = {p.name: p for p in list_images(original_dir)}
    var_imgs = {p.name: p for p in list_images(variant_dir)}
    keys = sorted(set(orig_imgs.keys()) & set(var_imgs.keys()))
    if not keys:
        raise RuntimeError(f"No matching images between {original_dir} and {variant_dir}")

    lpips_vals, ssim_vals, psnr_vals = [], [], []
    for name in keys:
        x = image_to_tensor(orig_imgs[name], device)
        y = image_to_tensor(var_imgs[name], device)
        ssim_vals.append(ssim_batch(y, x))
        psnr_vals.append(psnr_batch(y, x))
        if lpips_model is None:
            lpips_vals.append(float("nan"))
        else:
            with torch.no_grad():
                lpips_vals.append(lpips_model(y * 2 - 1, x * 2 - 1).mean().item())

    lpips_mean = float(np.nanmean(lpips_vals)) if not np.all(np.isnan(lpips_vals)) else float("nan")
    return lpips_mean, float(np.mean(ssim_vals)), float(np.mean(psnr_vals))


def main():
    # 실행 순서:
    # 1) variant yaml 기준 detector 성능 평가
    # 2) original 대비 지각 품질 지표 계산
    # 3) CSV 저장 + 콘솔 출력
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    data_yaml = Path(args.data).resolve()
    detector_weights = Path(args.weights).resolve()
    split = resolve_split(data_yaml)
    device = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device)

    rows = []
    original_dir = outdir / "original" / "images" / split
    if not original_dir.exists():
        raise FileNotFoundError(f"Original images not found: {original_dir}. Run generate_variant_images.py first.")

    for variant in args.variants:
        v = variant.lower()
        variant_yaml = outdir / f"{v}_{split}.yaml"
        if not variant_yaml.exists():
            raise FileNotFoundError(f"Variant yaml not found: {variant_yaml}")

        map50, p, rc, f1, speed = run_detector_val(
            detector_weights=detector_weights,
            data_yaml=variant_yaml,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=args.project,
            name=f"{args.name}_{v}",
        )

        if v == "original":
            # 원본은 자기 자신 대비이므로 품질 지표 고정
            lpips_v, ssim_v, psnr_v = 0.0, 1.0, float("nan")
        else:
            variant_dir = outdir / v / "images" / split
            lpips_v, ssim_v, psnr_v = compute_perceptual_metrics(original_dir, variant_dir, torch_device)

        rows.append({
            "Variant": v.capitalize(),
            "mAP50": map50,
            "Precision": p,
            "Recall": rc,
            "F1 Score": f1,
            "LPIPS": lpips_v,
            "SSIM": ssim_v,
            "PSNR": psnr_v,
            "Speed(s/img)": speed,
        })

    csv_path = outdir / "variant_summary.csv"
    fields = ["Variant", "mAP50", "Precision", "Recall", "F1 Score", "LPIPS", "SSIM", "PSNR", "Speed(s/img)"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n=== Variant Summary ===")
    for r in rows:
        psnr_str = "-" if np.isnan(r["PSNR"]) else f"{r['PSNR']:.4f}"
        lpips_str = "nan" if np.isnan(r["LPIPS"]) else f"{r['LPIPS']:.4f}"
        print(
            f"{r['Variant']:10s} | mAP50={r['mAP50']:.4f} | P={r['Precision']:.4f} | "
            f"R={r['Recall']:.4f} | F1={r['F1 Score']:.4f} | LPIPS={lpips_str} | "
            f"SSIM={r['SSIM']:.4f} | PSNR={psnr_str} | Speed={r['Speed(s/img)']:.4f}s/img"
        )
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
