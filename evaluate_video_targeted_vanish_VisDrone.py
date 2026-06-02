import argparse
import csv
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ultralytics import FastSAM, YOLO

import config
import model
from evaluate_video_4modes_VisDrone import (
    draw_detections,
    frame_to_square_tensor,
    image_pair_metrics,
    open_writer,
    result_to_mask,
    tensor_to_bgr,
)
from train import match_detections


def normalize_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def safe_filename_token(value):
    return re.sub(r"[^a-z0-9_-]+", "_", str(value).lower()).strip("_") or "target"


def names_dict(names):
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {idx: str(name) for idx, name in enumerate(names)}


def resolve_target_class_ids(names, target):
    available = names_dict(names)
    normalized = {normalize_name(name): cls_id for cls_id, name in available.items()}
    token = str(target).strip()
    if token.isdigit() and int(token) in available:
        return [int(token)], []
    cls_id = normalized.get(normalize_name(token))
    return ([cls_id], []) if cls_id is not None else ([], [token])


def class_options(names):
    return ", ".join(f"{cls_id}:{name}" for cls_id, name in names_dict(names).items())


def filter_result_by_classes(result, class_ids):
    if result.boxes is None or len(result.boxes) == 0 or not class_ids:
        return result[:0]
    cls = result.boxes.cls.detach()
    keep = torch.zeros_like(cls, dtype=torch.bool)
    for cls_id in class_ids:
        keep |= cls == int(cls_id)
    return result[keep]


def result_boxes(result, conf_thres, size):
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    out = []
    for box, conf in zip(boxes, confs):
        if float(conf) < conf_thres:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1, x2 = max(0, min(size, x1)), max(0, min(size, x2))
        y1, y2 = max(0, min(size, y1)), max(0, min(size, y2))
        if x2 > x1 and y2 > y1:
            out.append([x1, y1, x2, y2])
    return out


def union_result_masks(results, device, size):
    if not results or results[0].masks is None or results[0].masks.data is None:
        return torch.zeros((1, 1, size, size), device=device)
    masks = results[0].masks.data.float()
    if masks.shape[-2:] != (size, size):
        masks = F.interpolate(masks.unsqueeze(1), size=(size, size), mode="nearest").squeeze(1)
    return (masks.sum(dim=0, keepdim=True) > 0).float().unsqueeze(0).to(device)


def run_fastsam_prompt_mask(fastsam, img, target_boxes, device, size):
    img_np = (img[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    kwargs = dict(source=img_np, imgsz=size, device=device, verbose=False, retina_masks=True)
    if not target_boxes:
        return torch.zeros((1, 1, size, size), device=device)
    kwargs["bboxes"] = target_boxes
    results = fastsam(**kwargs)
    return union_result_masks(results, device=device, size=size)


def build_target_mask(fastsam, img, original_result, class_ids, conf_thres, device, size):
    target_result = filter_result_by_classes(original_result, class_ids)
    target_boxes = result_boxes(target_result, conf_thres=conf_thres, size=size)
    prompt_mask = run_fastsam_prompt_mask(
        fastsam=fastsam,
        img=img,
        target_boxes=target_boxes,
        device=device,
        size=size,
    )
    target_bbox_mask = result_to_mask(target_result, size, device, conf_thres=conf_thres)
    return prompt_mask * target_bbox_mask, target_result, len(target_boxes)


def build_targeted_vanish_image(img, target_mask, g_v):
    noise = torch.clamp(g_v(img), -config.EPSILON_V, config.EPSILON_V)
    return torch.clamp(img + noise * target_mask, 0, 1)


def draw_target_highlights(img, result, class_ids):
    out = img.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return out
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    clss = result.boxes.cls.detach().cpu().numpy().astype(int)
    for box, cls_id in zip(boxes, clss):
        if cls_id not in class_ids:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 215, 255), 3)
    return out


def draw_target_banner(pair, target, mask_ratio, target_box_count, target_orig_count, target_adv_count, attack_enabled):
    out = pair.copy()
    status = "ON" if attack_enabled else "OFF"
    text = (
        f"TARGET: {target} | ATTACK: {status} | "
        f"target detection: {target_orig_count} -> {target_adv_count} | "
        f"FastSAM masks: {target_box_count} | area: {mask_ratio * 100:.2f}%"
    )
    shortcuts = "Keys: 0-9 select class | A attack ON/OFF | SPACE pause | Q quit"
    y1 = out.shape[0] - 56
    cv2.rectangle(out, (0, y1), (out.shape[1], out.shape[0]), (0, 0, 0), -1)
    status_color = (0, 215, 255) if attack_enabled else (170, 170, 170)
    cv2.putText(out, text, (14, out.shape[0] - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.57, status_color, 1)
    cv2.putText(out, shortcuts, (14, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    return out


def draw_pair(
    img,
    adv,
    result_orig,
    result_adv,
    names,
    target,
    class_ids,
    mask_ratio,
    target_box_count,
    target_orig_count,
    target_adv_count,
    attack_enabled,
):
    left = draw_detections(tensor_to_bgr(img), result_orig, names=names, title="Original")
    right = draw_detections(tensor_to_bgr(adv), result_adv, names=names, title=f"Targeted Vanish: {target}")
    left = draw_target_highlights(left, result_orig, class_ids)
    right = draw_target_highlights(right, result_adv, class_ids)
    pair = np.concatenate([left, right], axis=1)
    return draw_target_banner(
        pair,
        target,
        mask_ratio,
        target_box_count,
        target_orig_count,
        target_adv_count,
        attack_enabled,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Interactive FastSAM-guided targeted Vanish demo for VisDrone classes."
    )
    p.add_argument("--source", required=True, help="Input video path")
    p.add_argument("--target", default="car", help="Initial VisDrone class name or class id")
    p.add_argument("--weights", default="weights/detectors/visdrone_best.pt", help="VisDrone detector weights")
    p.add_argument("--gv", default="outputs/training/generators/visdrone_vresults/G_v_best.pth", help="VisDrone Vanish generator")
    p.add_argument("--fastsam-weights", default="weights/segmentation/FastSAM-s.pt", help="FastSAM weights")
    p.add_argument("--outdir", default="outputs/video_eval/visdrone/targeted_vanish")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="", help="cuda, cuda:0, or cpu; empty = auto")
    p.add_argument("--conf-thres", type=float, default=0.25)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--save-overlay", action="store_true", help="Save side-by-side detector overlay")
    p.add_argument("--show", action="store_true", help="Display live side-by-side overlay")
    p.add_argument("--interactive", action="store_true", help="Enable live keyboard controls in the preview window")
    return p.parse_args()


def main():
    args = parse_args()
    source = Path(args.source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    outdir = Path(args.outdir).resolve()
    video_dir = outdir / "videos"
    overlay_dir = outdir / "overlays"
    video_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    size = args.imgsz

    yolo = YOLO(str(Path(args.weights).resolve()))
    yolo.model.to(device).eval()
    for parameter in yolo.model.parameters():
        parameter.requires_grad = False

    g_v = model.MobileUNetGenerator().to(device).eval()
    g_v.load_state_dict(torch.load(Path(args.gv).resolve(), map_location=device, weights_only=True))
    fastsam = FastSAM(str(Path(args.fastsam_weights).resolve()))

    target = args.target
    class_ids, unknown = resolve_target_class_ids(yolo.names, target)
    if unknown:
        raise ValueError(f"Unknown class: {unknown}. Available classes: {class_options(yolo.names)}")
    target = names_dict(yolo.names)[class_ids[0]]
    print(f"Available class keys: {class_options(yolo.names)}")
    print("Demo controls: 0-9 select class | A attack ON/OFF | SPACE pause | Q or ESC quit")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if args.max_frames is not None:
        total = min(total, max(args.max_frames, 0)) if total > 0 else args.max_frames

    target_token = safe_filename_token(target)
    raw_path = video_dir / f"{source.stem}_targeted_vanish_{target_token}.mp4"
    overlay_path = overlay_dir / f"{source.stem}_targeted_vanish_{target_token}_overlay.mp4"
    raw_writer = open_writer(raw_path, fps, (size, size))
    overlay_writer = open_writer(overlay_path, fps, (size * 2, size)) if args.save_overlay else None

    frame_csv = outdir / "targeted_vanish_frame_metrics.csv"
    rows = []
    frame_idx = 0
    stop = False
    paused = False
    attack_enabled = True
    show = args.show or args.interactive

    try:
        with tqdm(total=total if total > 0 else None, desc="targeted vanish frames") as pbar:
            while not stop:
                if args.max_frames is not None and frame_idx >= args.max_frames:
                    break
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                img = frame_to_square_tensor(frame_bgr, size, device)
                with torch.no_grad():
                    t0 = time.perf_counter()
                    result_orig = yolo(img, verbose=False)[0]
                    target_mask, target_orig, target_box_count = build_target_mask(
                        fastsam=fastsam,
                        img=img,
                        original_result=result_orig,
                        class_ids=class_ids,
                        conf_thres=args.conf_thres,
                        device=device,
                        size=size,
                    )
                    adv = build_targeted_vanish_image(img, target_mask, g_v) if attack_enabled else img
                    result_adv = yolo(adv, verbose=False)[0]
                    sec = time.perf_counter() - t0

                target_adv = filter_result_by_classes(result_adv, class_ids)
                missed, new, changed = match_detections(result_orig, result_adv)
                target_missed, target_new, target_changed = match_detections(target_orig, target_adv)
                psnr, linf = image_pair_metrics(img, adv)
                mask_ratio = float(target_mask.mean().item())

                raw_writer.write(tensor_to_bgr(adv))
                pair = draw_pair(
                    img,
                    adv,
                    result_orig,
                    result_adv,
                    yolo.names,
                    target,
                    class_ids,
                    mask_ratio,
                    target_box_count,
                    len(target_orig.boxes),
                    len(target_adv.boxes),
                    attack_enabled,
                )
                if overlay_writer is not None:
                    overlay_writer.write(pair)

                rows.append(
                    {
                        "frame": frame_idx,
                        "target": target,
                        "attack_enabled": int(attack_enabled),
                        "target_boxes": target_box_count,
                        "mask_area_ratio": mask_ratio,
                        "orig_det": len(result_orig.boxes),
                        "adv_det": len(result_adv.boxes),
                        "missed": missed,
                        "new": new,
                        "changed": changed,
                        "target_orig_det": len(target_orig.boxes),
                        "target_adv_det": len(target_adv.boxes),
                        "target_missed": target_missed,
                        "target_new": target_new,
                        "target_changed": target_changed,
                        "psnr": psnr,
                        "linf": linf,
                        "sec_per_frame": sec,
                    }
                )

                if show:
                    cv2.imshow("VisDrone Targeted Vanish Demo", pair)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        stop = True
                    elif args.interactive and key == ord("a"):
                        attack_enabled = not attack_enabled
                        print(f"Attack {'ON' if attack_enabled else 'OFF'}")
                    elif args.interactive and key == ord(" "):
                        paused = not paused
                        print("Paused" if paused else "Resumed")
                    elif args.interactive and ord("0") <= key <= ord("9"):
                        candidate_id = key - ord("0")
                        available = names_dict(yolo.names)
                        if candidate_id in available:
                            class_ids = [candidate_id]
                            target = available[candidate_id]
                            print(f"Target changed to: {candidate_id}:{target}")

                    while args.interactive and paused and not stop:
                        key = cv2.waitKey(50) & 0xFF
                        if key in (ord("q"), 27):
                            stop = True
                        elif key == ord(" "):
                            paused = False
                            print("Resumed")
                        elif key == ord("a"):
                            attack_enabled = not attack_enabled
                            print(f"Attack {'ON' if attack_enabled else 'OFF'}")
                        elif ord("0") <= key <= ord("9"):
                            candidate_id = key - ord("0")
                            available = names_dict(yolo.names)
                            if candidate_id in available:
                                class_ids = [candidate_id]
                                target = available[candidate_id]
                                print(f"Target changed to: {candidate_id}:{target}")

                frame_idx += 1
                pbar.update(1)
    finally:
        cap.release()
        raw_writer.release()
        if overlay_writer is not None:
            overlay_writer.release()
        if show:
            cv2.destroyAllWindows()

    fieldnames = [
        "frame",
        "target",
        "attack_enabled",
        "target_boxes",
        "mask_area_ratio",
        "orig_det",
        "adv_det",
        "missed",
        "new",
        "changed",
        "target_orig_det",
        "target_adv_det",
        "target_missed",
        "target_new",
        "target_changed",
        "psnr",
        "linf",
        "sec_per_frame",
    ]
    with frame_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== Targeted Vanish Demo Completed =====")
    print(f"Processed frames: {frame_idx}")
    print(f"Video: {raw_path}")
    if overlay_writer is not None:
        print(f"Overlay: {overlay_path}")
    print(f"Frame metrics: {frame_csv}")
    if rows:
        print(f"Target missed: {np.mean([r['target_missed'] for r in rows]):.2f}/frame")
        print(f"PSNR: {np.mean([r['psnr'] for r in rows]):.2f}")
        print(f"Speed: {np.mean([r['sec_per_frame'] for r in rows]):.4f}s/frame")


if __name__ == "__main__":
    main()
