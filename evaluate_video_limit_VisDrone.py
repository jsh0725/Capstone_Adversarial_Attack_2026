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
DEFAULT_FABRICATE_ALPHA = 1.0
DEFAULT_JANUSNET_FABRICATE_ALPHA = 0.7
CLASS_COLORS = [
    (255, 99, 71),
    (60, 179, 113),
    (65, 105, 225),
    (255, 165, 0),
    (199, 21, 133),
    (0, 206, 209),
    (154, 205, 50),
    (255, 215, 0),
    (138, 43, 226),
    (70, 130, 180),
]
BBOX_LABEL_SCALE = 0.42
BBOX_LABEL_THICKNESS = 1


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


def limit_mask_by_score(mask, score, area_ratio):
    area_ratio = float(np.clip(area_ratio, 0.0, 1.0))
    if area_ratio >= 1.0:
        return mask
    if area_ratio <= 0.0:
        return torch.zeros_like(mask)

    valid = mask > 0
    valid_count = int(valid.sum().item())
    if valid_count <= 0:
        return mask

    keep_count = max(1, int(np.ceil(valid_count * area_ratio)))
    values = score[valid].flatten()
    if keep_count >= values.numel():
        return mask

    threshold = torch.topk(values, k=keep_count, largest=True).values.min()
    return mask * (score >= threshold).float()


def build_variant_image(
    variant,
    img,
    pseudo_mask,
    g_v,
    g_f,
    fastsam_model,
    device,
    size,
    fabricate_alpha,
    fabricate_area_ratio,
):
    if variant == "original":
        return img

    if variant == "fabricate":
        nf = g_f(img)
        nf_c = torch.clamp(nf, -config.EPSILON_F, config.EPSILON_F)
        fabricate_mask = 1 - pseudo_mask
        score = nf_c.abs().mean(dim=1, keepdim=True)
        fabricate_mask = limit_mask_by_score(fabricate_mask, score, fabricate_area_ratio)
        return torch.clamp(img + fabricate_alpha * nf_c * fabricate_mask, 0, 1)

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
        score = nf_c.abs().mean(dim=1, keepdim=True)
        fabricate_mask = limit_mask_by_score(fabricate_mask, score, fabricate_area_ratio)
        return torch.clamp(img + nv_c * vanish_mask + fabricate_alpha * nf_c * fabricate_mask, 0, 1)

    raise ValueError(f"Unknown variant: {variant}")


def initial_fabricate_alpha(variant, args):
    if variant == "fabricate":
        return args.fabricate_alpha
    if variant == "janusnet":
        return args.janusnet_fabricate_alpha
    return 0.0


def should_limit_variant(variant):
    return variant in {"fabricate", "janusnet"}


def build_and_detect_with_limit(
    variant,
    img,
    pseudo_mask,
    res_orig,
    yolo,
    g_v,
    g_f,
    fastsam_model,
    device,
    size,
    args,
):
    alpha = initial_fabricate_alpha(variant, args)
    max_new = args.max_new_per_frame
    attempts = 0
    limit_applied = False

    while True:
        attempts += 1
        adv = build_variant_image(
            variant=variant,
            img=img,
            pseudo_mask=pseudo_mask,
            g_v=g_v,
            g_f=g_f,
            fastsam_model=fastsam_model,
            device=device,
            size=size,
            fabricate_alpha=alpha,
            fabricate_area_ratio=args.fabricate_area_ratio,
        )
        res_adv = yolo(adv, verbose=False)
        missed, new, changed = match_detections(res_orig, res_adv[0])

        if max_new is None or not should_limit_variant(variant) or new <= max_new:
            return adv, res_adv, missed, new, changed, alpha, attempts, limit_applied
        if attempts >= args.limit_attempts or alpha <= args.min_fabricate_alpha:
            return adv, res_adv, missed, new, changed, alpha, attempts, limit_applied

        limit_applied = True
        alpha = max(args.min_fabricate_alpha, alpha * args.alpha_decay)


def image_pair_metrics(orig, adv):
    o = orig.detach().cpu().numpy()[0]
    a = adv.detach().cpu().numpy()[0]
    mse = float(((o - a) ** 2).mean())
    psnr = float("inf") if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))
    linf = float(np.abs(o - a).max())
    return psnr, linf


def class_color(cls_id):
    return CLASS_COLORS[int(cls_id) % len(CLASS_COLORS)]


def class_name(names, cls_id):
    if isinstance(names, dict):
        return names.get(int(cls_id), str(cls_id))
    if isinstance(names, (list, tuple)) and 0 <= int(cls_id) < len(names):
        return names[int(cls_id)]
    return str(cls_id)


def draw_panel_title(img_bgr, title):
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(out, title, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    return out


def detection_count_lines(result, names=None, max_chars=42):
    if result.boxes is None or len(result.boxes) == 0:
        return ["No objects"]

    clss = result.boxes.cls.detach().cpu().numpy().astype(int)
    counts = {}
    for cls_id in clss:
        counts[cls_id] = counts.get(cls_id, 0) + 1

    parts = [f"{class_name(names, cls_id)}: {counts[cls_id]}" for cls_id in sorted(counts)]
    lines = []
    line = ""
    for part in parts:
        sep = "  " if line else ""
        if line and len(line) + len(sep) + len(part) > max_chars:
            lines.append(line)
            line = part
        else:
            line += sep + part
    if line:
        lines.append(line)
    return lines


def draw_detection_summary(img_bgr, lines, top):
    out = img_bgr.copy()
    line_h = 19
    pad = 8
    bottom = top + pad * 2 + line_h * len(lines)
    cv2.rectangle(out, (0, top), (out.shape[1], bottom), (255, 255, 255), -1)
    for idx, line in enumerate(lines):
        y = top + pad + 14 + idx * line_h
        cv2.putText(out, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)
    return out, bottom


def draw_detections(img_bgr, result, names=None, title=None):
    out = img_bgr.copy()
    if title:
        out = draw_panel_title(out, title)

    summary_lines = detection_count_lines(result, names=names)
    summary_top = 42 if title else 0
    out, summary_bottom = draw_detection_summary(out, summary_lines, top=summary_top)

    if result.boxes is None or len(result.boxes) == 0:
        return out

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    clss = result.boxes.cls.detach().cpu().numpy().astype(int)

    for box, conf, cls_id in zip(boxes, confs, clss):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = class_color(cls_id)
        label_name = class_name(names, cls_id)
        label = f"{label_name} {conf:.2f}"
        label_y = max(summary_bottom + 18, y1 - 6)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, BBOX_LABEL_SCALE, BBOX_LABEL_THICKNESS)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(out, (x1, label_y - th - baseline), (x1 + tw + 4, label_y + baseline), (255, 255, 255), -1)
        cv2.putText(out, label, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, BBOX_LABEL_SCALE, (0, 0, 0), BBOX_LABEL_THICKNESS)
    return out


def draw_pair_frame(img, adv, result_orig, result_adv, size, names=None, variant="attacked"):
    orig_plot = draw_detections(tensor_to_bgr(img), result_orig, names=names, title="Original")
    adv_plot = draw_detections(tensor_to_bgr(adv), result_adv, names=names, title=f"Attacked: {variant}")
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
    p = argparse.ArgumentParser(description="Evaluate VisDrone video attacks with controllable Fabricate generation limits.")
    p.add_argument("--source", required=True, help="Input video path")
    p.add_argument("--weights", default="weights/detectors/visdrone_best.pt", help="VisDrone detector weights path")
    p.add_argument("--gv", default="outputs/training/generators/visdrone_vresults/G_v_best.pth", help="VisDrone vanish generator checkpoint")
    p.add_argument("--gf", default="outputs/training/generators/visdrone_fresults/G_f_best.pth", help="VisDrone fabricate generator checkpoint")
    p.add_argument("--fastsam-weights", default="weights/segmentation/FastSAM-s.pt", help="FastSAM weights path")
    p.add_argument("--variants", nargs="+", default=VARIANTS)
    p.add_argument("--outdir", default="outputs/video_eval/visdrone/limit")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="", help="cuda, cuda:0, or cpu; empty = auto")
    p.add_argument("--conf-thres", type=float, default=0.25, help="Pseudo-mask detector confidence threshold")
    p.add_argument("--max-frames", type=int, default=None, help="Limit processed frames")
    p.add_argument("--save-overlay", action="store_true", help="Save side-by-side original/attacked detector overlays")
    p.add_argument("--fabricate-alpha", type=float, default=DEFAULT_FABRICATE_ALPHA, help="Fabricate-mode G_f strength")
    p.add_argument("--janusnet-fabricate-alpha", type=float, default=DEFAULT_JANUSNET_FABRICATE_ALPHA, help="JanusNET G_f strength")
    p.add_argument("--fabricate-area-ratio", type=float, default=1.0, help="Fraction of non-object mask where G_f is applied")
    p.add_argument("--max-new-per-frame", type=int, default=None, help="Reduce G_f alpha when new detections exceed this cap")
    p.add_argument("--min-fabricate-alpha", type=float, default=0.1, help="Lowest adaptive G_f alpha")
    p.add_argument("--alpha-decay", type=float, default=0.8, help="Adaptive alpha multiplier when new detections exceed cap")
    p.add_argument("--limit-attempts", type=int, default=5, help="Maximum adaptive retries per frame and variant")
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
            "effective_alpha": 0.0,
            "limit_attempts": 0.0,
            "limited_frames": 0.0,
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
                        (
                            adv,
                            res_adv,
                            missed,
                            new,
                            changed,
                            effective_alpha,
                            limit_attempts,
                            limit_applied,
                        ) = build_and_detect_with_limit(
                            variant=variant,
                            img=img,
                            pseudo_mask=pseudo_mask,
                            res_orig=res_orig[0],
                            yolo=yolo,
                            g_v=g_v,
                            g_f=g_f,
                            fastsam_model=fastsam,
                            device=device,
                            size=size,
                            args=args,
                        )
                        sec = time.perf_counter() - t0

                        psnr, linf = (float("inf"), 0.0) if variant == "original" else image_pair_metrics(img, adv)
                        n_orig = len(res_orig[0].boxes)
                        n_adv = len(res_adv[0].boxes)

                        writers[variant].write(tensor_to_bgr(adv))
                        if args.save_overlay:
                            overlay_writers[variant].write(
                                draw_pair_frame(img, adv, res_orig[0], res_adv[0], size, names=yolo.names, variant=variant)
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
                            "effective_alpha": effective_alpha,
                            "limit_attempts": limit_attempts,
                            "limit_applied": int(limit_applied),
                        }
                        frame_rows.append(row)

                        s = summary[variant]
                        s["frames"] += 1
                        for key in [
                            "orig_det",
                            "adv_det",
                            "missed",
                            "new",
                            "changed",
                            "psnr",
                            "linf",
                            "sec_per_frame",
                            "effective_alpha",
                            "limit_attempts",
                        ]:
                            if key == "psnr" and np.isinf(row[key]):
                                continue
                            s[key] += float(row[key])
                        s["limited_frames"] += float(row["limit_applied"])

                frame_idx += 1
                pbar.update(1)
    finally:
        cap.release()
        for writer in writers.values():
            writer.release()
        for writer in overlay_writers.values():
            writer.release()

    with open(frame_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "frame",
            "variant",
            "orig_det",
            "adv_det",
            "missed",
            "new",
            "changed",
            "psnr",
            "linf",
            "sec_per_frame",
            "effective_alpha",
            "limit_attempts",
            "limit_applied",
        ]
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
            "effective_alpha": s["effective_alpha"] / n,
            "limit_attempts_per_frame": s["limit_attempts"] / n,
            "limited_frame_ratio": s["limited_frames"] / n,
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
            "effective_alpha",
            "limit_attempts_per_frame",
            "limited_frame_ratio",
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
            f"PSNR={r['psnr']}  Linf={r['linf']:.4f}  "
            f"alpha={r['effective_alpha']:.3f}  limited={r['limited_frame_ratio']:.2f}  "
            f"Speed={r['sec_per_frame']:.4f}s/frame"
        )


if __name__ == "__main__":
    main()
