import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from ultralytics import FastSAM, YOLO

import config
import model
from train import _plot_tensor, make_stats_panel, match_detections


VARIANTS = ["original", "fabricate", "vanish", "janusnet"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resolve_dataset_paths(data_yaml, split="val"):
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    base = Path(data.get("path", data_yaml.parent)).expanduser()
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()

    rel = data.get(split)
    if rel is None:
        raise KeyError(f"Split '{split}' is not defined in {data_yaml}")

    img_dir = (base / rel).resolve()
    lbl_dir = (base / "labels" / split).resolve()
    if not img_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {img_dir}")
    if not lbl_dir.exists():
        raise FileNotFoundError(f"Label dir not found: {lbl_dir}")

    names = data.get("names", [])
    nc = int(data.get("nc", len(names) if names else 0))
    return img_dir, lbl_dir, names, nc


def letterbox_to_square(pil_img, size):
    w, h = pil_img.size
    scale = size / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    resized = pil_img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    arr = np.asarray(canvas).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor, (w, h, nw, nh, pad_x, pad_y)


def transform_label_file(src_label_path, info, out_label_path, size):
    _, _, nw, nh, pad_x, pad_y = info
    boxes = []
    lines_out = []
    if src_label_path.exists():
        with src_label_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls, xc, yc, bw, bh = line.split()
                cls = int(float(cls))
                xc = float(xc)
                yc = float(yc)
                bw = float(bw)
                bh = float(bh)

                xc_n = (xc * nw + pad_x) / size
                yc_n = (yc * nh + pad_y) / size
                bw_n = (bw * nw) / size
                bh_n = (bh * nh) / size
                boxes.append([xc_n, yc_n, bw_n, bh_n])
                lines_out.append(f"{cls} {xc_n:.6f} {yc_n:.6f} {bw_n:.6f} {bh_n:.6f}")

    out_label_path.parent.mkdir(parents=True, exist_ok=True)
    with out_label_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))
    return boxes


def boxes_to_mask(boxes, size, device):
    mask = torch.zeros((1, 1, size, size), device=device)
    for xc, yc, bw, bh in boxes:
        x1 = int((xc - bw / 2) * size)
        y1 = int((yc - bh / 2) * size)
        x2 = int((xc + bw / 2) * size)
        y2 = int((yc + bh / 2) * size)
        mask[0, 0, max(0, y1):min(size, y2), max(0, x1):min(size, x2)] = 1.0
    return mask


def tensor_to_pil(t):
    t = t.detach().clamp(0, 1).cpu()
    arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def run_fastsam_union_mask(fastsam_model, img_tensor, device, size):
    img_np = (img_tensor[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    results = fastsam_model(source=img_np, imgsz=size, device=device, verbose=False)
    if not results or results[0].masks is None or results[0].masks.data is None:
        return torch.zeros((1, 1, size, size), device=img_tensor.device)
    masks = results[0].masks.data.float()  # (N, H, W)
    union = (masks.sum(dim=0, keepdim=True) > 0).float()  # (1, H, W)
    return union.unsqueeze(0).to(img_tensor.device)  # (1, 1, H, W)


def norm_for_vis(t):
    return (t - t.min()) / (t.max() - t.min() + 1e-8)


def image_pair_metrics(orig, adv, lpips_fn):
    # orig, adv: (1, 3, H, W), range [0,1]
    o = orig.detach().cpu().numpy()[0].transpose(1, 2, 0)
    a = adv.detach().cpu().numpy()[0].transpose(1, 2, 0)

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_ch = []
    for ch in range(3):
        x = o[:, :, ch]
        y = a[:, :, ch]
        mx = x.mean()
        my = y.mean()
        vx = ((x - mx) ** 2).mean()
        vy = ((y - my) ** 2).mean()
        vxy = ((x - mx) * (y - my)).mean()
        ssim_c = ((2 * mx * my + c1) * (2 * vxy + c2)) / (
            (mx * mx + my * my + c1) * (vx + vy + c2) + 1e-12
        )
        ssim_ch.append(float(ssim_c))
    ssim = float(np.mean(ssim_ch))

    mse = float(((o - a) ** 2).mean())
    psnr = float("inf") if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))
    lp = lpips_fn(orig * 2 - 1, adv * 2 - 1).item()
    return lp, ssim, psnr


def ensure_dataset_yaml(yaml_path, variant_root, names, nc):
    if isinstance(names, dict):
        names_dict = {int(k): v for k, v in names.items()}
    else:
        names_dict = {i: n for i, n in enumerate(names)}
    data = {
        "path": str(variant_root),
        "train": "images",
        "val": "images",
        "test": "images",
        "nc": nc,
        "names": names_dict,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def build_variant_image(variant, img, gt_mask, g_v, g_f, fastsam_model, device, size):
    if variant == "original":
        return img

    if variant == "fabricate":
        nf = g_f(img)
        nf_c = torch.clamp(nf, -config.EPSILON_F, config.EPSILON_F)
        return torch.clamp(img + nf_c * (1 - gt_mask), 0, 1)

    if variant == "vanish":
        nv = g_v(img)
        nv_c = torch.clamp(nv, -config.EPSILON_V, config.EPSILON_V)
        return torch.clamp(img + nv_c * gt_mask, 0, 1)

    if variant in {"janusnet", "hybrid_fastsam"}:
        # JanusNET is our method name here: G_v + G_f + FastSAM mask-based stitching.
        # It is not a separate external architecture or downloadable implementation.
        nv = g_v(img)
        nf = g_f(img)
        nv_c = torch.clamp(nv, -config.EPSILON_V, config.EPSILON_V)
        nf_c = torch.clamp(nf, -config.EPSILON_F, config.EPSILON_F)
        sam_obj = run_fastsam_union_mask(fastsam_model, img, device, size)
        return torch.clamp(img + nv_c * sam_obj + nf_c * (1 - sam_obj), 0, 1)

    raise ValueError(f"Unknown variant: {variant}")


def evaluate_single_variant(
    variant,
    img_files,
    val_lbl_dir,
    cache_root,
    examples_root,
    yolo,
    g_v,
    g_f,
    fastsam_model,
    lpips_fn,
    size,
    device,
    example_count=10,
):
    v_root = cache_root / variant
    v_img_dir = v_root / "images"
    v_lbl_dir = v_root / "labels"
    v_ex_dir = examples_root / variant
    v_img_dir.mkdir(parents=True, exist_ok=True)
    v_lbl_dir.mkdir(parents=True, exist_ok=True)
    save_examples = variant != "original"
    if save_examples:
        v_ex_dir.mkdir(parents=True, exist_ok=True)

    acc = {"lpips": 0.0, "ssim": 0.0, "psnr": 0.0, "speed": 0.0, "count": 0}
    saved_examples = 0

    loop = tqdm(img_files, desc=f"{variant}: build+measure")
    for img_path in loop:
        base = img_path.stem
        lbl_src = val_lbl_dir / f"{base}.txt"
        out_lbl = v_lbl_dir / f"{base}.txt"
        out_img = v_img_dir / f"{base}.png"

        pil = Image.open(img_path).convert("RGB")
        img_t, info = letterbox_to_square(pil, size)
        img = img_t.unsqueeze(0).to(device)
        boxes = transform_label_file(lbl_src, info, out_lbl, size)
        gt_mask = boxes_to_mask(boxes, size, device)

        with torch.no_grad():
            t0 = time.perf_counter()
            adv = build_variant_image(
                variant=variant,
                img=img,
                gt_mask=gt_mask,
                g_v=g_v,
                g_f=g_f,
                fastsam_model=fastsam_model,
                device=device,
                size=size,
            )
            _ = yolo(adv, verbose=False)
            dt = time.perf_counter() - t0

            if variant == "original":
                lp, ss, ps = 0.0, 1.0, float("inf")
            else:
                lp, ss, ps = image_pair_metrics(img, adv, lpips_fn)

        tensor_to_pil(adv[0]).save(out_img)
        if save_examples and saved_examples < example_count:
            noise_for_vis = None
            panel_attack_type = None
            if variant == "vanish":
                noise_for_vis = torch.clamp(g_v(img), -config.EPSILON_V, config.EPSILON_V)
                panel_attack_type = "vanish"
            elif variant == "fabricate":
                noise_for_vis = torch.clamp(g_f(img), -config.EPSILON_F, config.EPSILON_F)
                panel_attack_type = "fabricate"
            elif variant in {"janusnet", "hybrid_fastsam"}:
                # Keep train.py layout (noise panel + stats panel) while showing JanusNET perturbation.
                nv = torch.clamp(g_v(img), -config.EPSILON_V, config.EPSILON_V)
                nf = torch.clamp(g_f(img), -config.EPSILON_F, config.EPSILON_F)
                sam_obj = run_fastsam_union_mask(fastsam_model, img, device, size)
                noise_for_vis = nv * sam_obj + nf * (1 - sam_obj)
                panel_attack_type = "janus"

            res_orig = yolo(img, verbose=False)
            res_adv = yolo(adv, verbose=False)
            n_o = len(res_orig[0].boxes)
            n_a = len(res_adv[0].boxes)
            n_missed, n_new, n_chg = match_detections(res_orig[0], res_adv[0])

            orig_pred = _plot_tensor(res_orig[0], device)
            adv_pred = _plot_tensor(res_adv[0], device)
            panel = make_stats_panel(
                n_orig=n_o,
                n_adv=n_a,
                n_missed=n_missed,
                n_new=n_new,
                n_changed=n_chg,
                attack_type=panel_attack_type,
                device=device,
                size=size,
            )

            row1 = torch.cat([img[0], orig_pred, norm_for_vis(noise_for_vis)[0]], dim=2)
            row2 = torch.cat([adv[0], adv_pred, panel], dim=2)
            grid = torch.cat([row1, row2], dim=1)
            tensor_to_pil(grid).save(v_ex_dir / f"{saved_examples + 1:02d}_{base}.png")
            saved_examples += 1

        acc["lpips"] += lp
        acc["ssim"] += ss
        acc["psnr"] += ps
        acc["speed"] += dt
        acc["count"] += 1

    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="vhrv/vhrv.yaml", help="Dataset yaml")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--weights", type=str, default="vhrv_yolo11x.pt", help="Detector weights path")
    parser.add_argument("--gv", type=str, default="results/vhrv_vresults/G_v_best.pth", help="Vanish generator checkpoint")
    parser.add_argument("--gf", type=str, default="results/vhrv_fresults/G_f_best.pth", help="Fabricate generator checkpoint")
    parser.add_argument("--outdir", type=str, default="eval_outputs", help="Output directory")
    parser.add_argument("--imgsz", type=int, default=640, help="Square image size")
    parser.add_argument("--batch", type=int, default=1, help="YOLO val batch")
    parser.add_argument("--device", type=str, default="", help="cuda, cuda:0, or cpu; empty = auto")
    parser.add_argument("--variants", nargs="+", default=VARIANTS, help="Modes to evaluate")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of val images")
    parser.add_argument("--fastsam-weights", type=str, default="FastSAM-s.pt", help="FastSAM weights path")
    parser.add_argument("--examples-per-setting", type=int, default=10, help="Number of sample images per setting")
    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    size = args.imgsz
    batch = args.batch

    root = Path.cwd()
    data_yaml = Path(args.data).resolve()
    val_img_dir, val_lbl_dir, names, nc = resolve_dataset_paths(data_yaml, split=args.split)

    out_root = Path(args.outdir).resolve()
    cache_root = out_root / "generated_val_sets"
    examples_root = out_root / "examples"
    csv_path = out_root / "val_4mode_metrics.csv"
    out_root.mkdir(parents=True, exist_ok=True)

    # Load detector and generators (best checkpoints)
    yolo = YOLO(str(Path(args.weights).resolve()))
    yolo.model.to(device).eval()
    for p in yolo.model.parameters():
        p.requires_grad = False

    g_v = model.MobileUNetGenerator().to(device)
    g_f = model.MobileUNetGenerator().to(device)
    g_v.load_state_dict(torch.load(Path(args.gv).resolve(), map_location=device))
    g_f.load_state_dict(torch.load(Path(args.gf).resolve(), map_location=device))
    g_v.eval()
    g_f.eval()

    # LPIPS + FastSAM. FastSAM is only a local mask provider for our JanusNET method.
    import lpips

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()
    variants = [v.lower() for v in args.variants]
    fastsam = FastSAM(str(Path(args.fastsam_weights).resolve())) if any(v in {"janusnet", "hybrid_fastsam"} for v in variants) else None

    img_files = sorted([p for p in val_img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not img_files:
        raise RuntimeError(f"No validation images found in: {val_img_dir}")
    if args.max_images is not None:
        img_files = img_files[: max(args.max_images, 0)]

    rows = []
    for variant in variants:
        print(f"\n===== Run setting: {variant} =====")
        acc = evaluate_single_variant(
            variant=variant,
            img_files=img_files,
            val_lbl_dir=val_lbl_dir,
            cache_root=cache_root,
            examples_root=examples_root,
            yolo=yolo,
            g_v=g_v,
            g_f=g_f,
            fastsam_model=fastsam,
            lpips_fn=lpips_fn,
            size=size,
            device=device,
            example_count=args.examples_per_setting,
        )

        yaml_path = out_root / f"val_{variant}.yaml"
        ensure_dataset_yaml(yaml_path, cache_root / variant, names if names else yolo.names, nc)
        res = yolo.val(
            data=str(yaml_path),
            split="val",
            imgsz=size,
            batch=batch,
            device=device,
            verbose=False,
            plots=False,
        )
        p = float(res.box.mp)
        r = float(res.box.mr)
        map50 = float(res.box.map50)
        f1 = 2 * p * r / (p + r + 1e-12)

        c = max(acc["count"], 1)
        rows.append(
            {
                "variant": variant,
                "map50": map50,
                "precision": p,
                "recall": r,
                "f1": f1,
                "lpips": acc["lpips"] / c,
                "ssim": acc["ssim"] / c,
                "psnr": acc["psnr"] / c,
                "avg_sec_per_image": acc["speed"] / c,
            }
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "map50",
                "precision",
                "recall",
                "f1",
                "lpips",
                "ssim",
                "psnr",
                "avg_sec_per_image",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n===== Evaluation Completed =====")
    print(f"CSV saved: {csv_path}")
    print(f"Example folders saved under: {examples_root}")
    print("Columns:")
    print("  map50, precision, recall, f1")
    print("  lpips, ssim, psnr, avg_sec_per_image")
    print("\nPer-variant summary:")
    for r in rows:
        print(
            f"[{r['variant']:<14}] "
            f"mAP50={r['map50']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1']:.4f}  "
            f"LPIPS={r['lpips']:.4f}  SSIM={r['ssim']:.4f}  PSNR={r['psnr']:.2f}  "
            f"Speed={r['avg_sec_per_image']:.4f}s/img"
        )


if __name__ == "__main__":
    main()
