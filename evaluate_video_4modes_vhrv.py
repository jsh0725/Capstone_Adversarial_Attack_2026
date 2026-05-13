import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import FastSAM, YOLO

import config
import model
from train import match_detections


VARIANTS = ["original", "fabricate", "vanish", "janusnet"]


def frame_to_square_tensor(frame_bgr, size, device):
    h, w = frame_bgr.shape[:2]
    scale = size / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def tensor_to_bgr(tensor):
    t = tensor.detach().clamp(0, 1).cpu()[0]
    rgb = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def result_to_mask(result, size, device, conf_thres=0.25):
    mask = torch.zeros((1, 1, size, size), device=device)
    if result.boxes is None or len(result.boxes) == 0:
        return mask

    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    conf = result.boxes.conf.detach().cpu().numpy()
    for box, score in zip(xyxy, conf):
        if score < conf_thres:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(size, x1))
        x2 = max(0, min(size, x2))
        y1 = max(0, min(size, y1))
        y2 = max(0, min(size, y2))
        if x2 > x1 and y2 > y1:
            mask[:, :, y1:y2, x1:x2] = 1.0
    return mask


def run_fastsam_union_mask(fastsam_model, img_tensor, device, size):
    img_np = (img_tensor[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    results = fastsam_model(source=img_np, imgsz=size, device=device, verbose=False)
    if not results or results[0].masks is None or results[0].masks.data is None:
        return torch.zeros((1, 1, size, size), device=img_tensor.device)
    masks = results[0].masks.data.float()
    union = (masks.sum(dim=0, keepdim=True) > 0).float()
    return union.unsqueeze(0).to(img_tensor.device)


def build_janusnet_masks(fastsam_model, img_tensor, bbox_mask, device, size):
    sam_obj = run_fastsam_union_mask(fastsam_model, img_tensor, device, size)
    vanish_mask = sam_obj * bbox_mask
    fabricate_mask = 1 - bbox_mask
    return vanish_mask, fabricate_mask


def build_variant_image(variant, img, pseudo_mask, g_v, g_f, fastsam_model, device, size):
    if variant == "original":
        return img

    if variant == "fabricate":
        nf = g_f(img)
        nf_c = torch.clamp(nf, -config.EPSILON_F, config.EPSILON_F)
        return torch.clamp(img + nf_c * (1 - pseudo_mask), 0, 1)

    if variant == "vanish":
        nv = g_v(img)
        nv_c = torch.clamp(nv, -config.EPSILON_V, config.EPSILON_V)
        return torch.clamp(img + nv_c * pseudo_mask, 0, 1)

    if variant == "janusnet":
        nv = g_v(img)
        nf = g_f(img)
        nv_c = torch.clamp(nv, -config.EPSILON_V, config.EPSILON_V)
        nf_c = torch.clamp(nf, -config.EPSILON_F, config.EPSILON_F)
        vanish_mask, fabricate_mask = build_janusnet_masks(fastsam_model, img, pseudo_mask, device, size)
        return torch.clamp(img + nv_c * vanish_mask + nf_c * fabricate_mask, 0, 1)

    raise ValueError(f"Unknown variant: {variant}")


def image_pair_metrics(orig, adv):
    o = orig.detach().cpu().numpy()[0]
    a = adv.detach().cpu().numpy()[0]
    mse = float(((o - a) ** 2).mean())
    psnr = float("inf") if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))
    linf = float(np.abs(o - a).max())
    return psnr, linf


def draw_detections(img_bgr, result, names=None):
    out = img_bgr.copy()
    if result.boxes is None or len(result.boxes) == 0:
        cv2.putText(out, "detections: 0", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 255), 2)
        return out

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    clss = result.boxes.cls.detach().cpu().numpy().astype(int)
    for box, conf, cls_id in zip(boxes, confs, clss):
        x1, y1, x2, y2 = [int(v) for v in box]
        label_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        label = f"{label_name} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(out, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
    cv2.putText(out, f"detections: {len(boxes)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 255, 80), 2)
    return out


def draw_pair_frame(img, adv, result_orig, result_adv, size, names=None):
    orig_plot = draw_detections(tensor_to_bgr(img), result_orig, names=names)
    adv_plot = draw_detections(tensor_to_bgr(adv), result_adv, names=names)
    if orig_plot.shape[:2] != (size, size):
        orig_plot = cv2.resize(orig_plot, (size, size))
    if adv_plot.shape[:2] != (size, size):
        adv_plot = cv2.resize(adv_plot, (size, size))
    return np.concatenate([orig_plot, adv_plot], axis=1)


def open_writer(path, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VHRV video attacks for original/fabricate/vanish/janusnet modes.")
    p.add_argument("--source", required=True, help="Input video path")
    p.add_argument("--weights", default="vhrv_yolo11x.pt", help="Detector weights path")
    p.add_argument("--gv", default="results/vhrv_vresults/G_v_best.pth", help="Vanish generator checkpoint")
    p.add_argument("--gf", default="results/vhrv_fresults/G_f_best.pth", help="Fabricate generator checkpoint")
    p.add_argument("--fastsam-weights", default="FastSAM-s.pt", help="FastSAM weights path")
    p.add_argument("--variants", nargs="+", default=VARIANTS)
    p.add_argument("--outdir", default="video_outputs_vhrv")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="", help="cuda, cuda:0, or cpu; empty = auto")
    p.add_argument("--conf-thres", type=float, default=0.25, help="Pseudo-mask detector confidence threshold")
    p.add_argument("--max-frames", type=int, default=None, help="Limit processed frames")
    p.add_argument("--save-overlay", action="store_true", help="Save side-by-side original/attacked detector overlays")
    return p.parse_args()


def main():
    args = parse_args()
    source = Path(args.source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    raw_dir = outdir / "videos"
    overlay_dir = outdir / "overlays"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    size = args.imgsz
    variants = [v.lower() for v in args.variants]

    yolo = YOLO(str(Path(args.weights).resolve()))
    yolo.model.to(device).eval()
    for p in yolo.model.parameters():
        p.requires_grad = False

    g_v = model.MobileUNetGenerator().to(device).eval()
    g_f = model.MobileUNetGenerator().to(device).eval()
    g_v.load_state_dict(torch.load(Path(args.gv).resolve(), map_location=device))
    g_f.load_state_dict(torch.load(Path(args.gf).resolve(), map_location=device))

    fastsam = FastSAM(str(Path(args.fastsam_weights).resolve())) if "janusnet" in variants else None

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if args.max_frames is not None:
        total = min(total, max(args.max_frames, 0)) if total > 0 else args.max_frames

    writers = {
        v: open_writer(raw_dir / f"{source.stem}_{v}.mp4", fps, (size, size))
        for v in variants
    }
    overlay_writers = {}
    if args.save_overlay:
        overlay_writers = {
            v: open_writer(overlay_dir / f"{source.stem}_{v}_overlay.mp4", fps, (size * 2, size))
            for v in variants
        }

    frame_csv = outdir / "video_frame_metrics.csv"
    summary_csv = outdir / "video_summary.csv"
    frame_rows = []
    summary = {
        v: {
            "frames": 0,
            "orig_det": 0.0,
            "adv_det": 0.0,
            "missed": 0.0,
            "new": 0.0,
            "changed": 0.0,
            "psnr": 0.0,
            "linf": 0.0,
            "sec_per_frame": 0.0,
        }
        for v in variants
    }

    frame_idx = 0
    progress_total = total if total and total > 0 else None
    try:
        with tqdm(total=progress_total, desc="video frames") as pbar:
            while True:
                if args.max_frames is not None and frame_idx >= args.max_frames:
                    break
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                img = frame_to_square_tensor(frame_bgr, size, device)
                with torch.no_grad():
                    res_orig = yolo(img, verbose=False)
                    pseudo_mask = result_to_mask(res_orig[0], size, device, conf_thres=args.conf_thres)

                    for variant in variants:
                        t0 = time.perf_counter()
                        adv = build_variant_image(
                            variant=variant,
                            img=img,
                            pseudo_mask=pseudo_mask,
                            g_v=g_v,
                            g_f=g_f,
                            fastsam_model=fastsam,
                            device=device,
                            size=size,
                        )
                        res_adv = yolo(adv, verbose=False)
                        sec = time.perf_counter() - t0

                        psnr, linf = (float("inf"), 0.0) if variant == "original" else image_pair_metrics(img, adv)
                        n_orig = len(res_orig[0].boxes)
                        n_adv = len(res_adv[0].boxes)
                        missed, new, changed = match_detections(res_orig[0], res_adv[0])

                        writers[variant].write(tensor_to_bgr(adv))
                        if args.save_overlay:
                            overlay_writers[variant].write(
                                draw_pair_frame(img, adv, res_orig[0], res_adv[0], size, names=yolo.names)
                            )

                        row = {
                            "frame": frame_idx,
                            "variant": variant,
                            "orig_det": n_orig,
                            "adv_det": n_adv,
                            "missed": missed,
                            "new": new,
                            "changed": changed,
                            "psnr": psnr,
                            "linf": linf,
                            "sec_per_frame": sec,
                        }
                        frame_rows.append(row)

                        s = summary[variant]
                        s["frames"] += 1
                        for key in ["orig_det", "adv_det", "missed", "new", "changed", "psnr", "linf", "sec_per_frame"]:
                            if key == "psnr" and np.isinf(row[key]):
                                continue
                            s[key] += float(row[key])

                frame_idx += 1
                pbar.update(1)
    finally:
        cap.release()
        for writer in writers.values():
            writer.release()
        for writer in overlay_writers.values():
            writer.release()

    with open(frame_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["frame", "variant", "orig_det", "adv_det", "missed", "new", "changed", "psnr", "linf", "sec_per_frame"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(frame_rows)

    rows = []
    for variant, s in summary.items():
        n = max(s["frames"], 1)
        psnr_n = n if variant != "original" else 1
        rows.append({
            "variant": variant,
            "frames": s["frames"],
            "orig_det_per_frame": s["orig_det"] / n,
            "adv_det_per_frame": s["adv_det"] / n,
            "missed_per_frame": s["missed"] / n,
            "new_per_frame": s["new"] / n,
            "changed_per_frame": s["changed"] / n,
            "psnr": "inf" if variant == "original" else s["psnr"] / psnr_n,
            "linf": s["linf"] / n,
            "sec_per_frame": s["sec_per_frame"] / n,
        })

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "variant",
            "frames",
            "orig_det_per_frame",
            "adv_det_per_frame",
            "missed_per_frame",
            "new_per_frame",
            "changed_per_frame",
            "psnr",
            "linf",
            "sec_per_frame",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("\n===== Video Evaluation Completed =====")
    print(f"Processed frames: {frame_idx}")
    print(f"Videos: {raw_dir}")
    if args.save_overlay:
        print(f"Overlays: {overlay_dir}")
    print(f"Frame metrics: {frame_csv}")
    print(f"Summary: {summary_csv}")
    for r in rows:
        print(
            f"[{r['variant']:<10}] "
            f"orig_det={r['orig_det_per_frame']:.2f}  adv_det={r['adv_det_per_frame']:.2f}  "
            f"missed={r['missed_per_frame']:.2f}  new={r['new_per_frame']:.2f}  "
            f"PSNR={r['psnr']}  Linf={r['linf']:.4f}  Speed={r['sec_per_frame']:.4f}s/frame"
        )


if __name__ == "__main__":
    main()
