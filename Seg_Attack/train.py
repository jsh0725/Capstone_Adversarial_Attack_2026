import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO
from torchvision.utils import save_image
from torchvision.ops import box_iou
import numpy as np
import os
import csv
from datetime import datetime
import config, model, dataset, utils


def collate_fn(batch):
    imgs, boxes = zip(*batch)
    return torch.stack(imgs), list(boxes)


def _t(arr, device):
    """HWC uint8 numpy → CHW float32 tensor"""
    return torch.from_numpy(arr.transpose(2, 0, 1).astype(np.float32) / 255.0).to(device)


def total_variation(x):
    """Anisotropic TV loss over (B, C, H, W) tensor."""
    return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + \
           (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()


def color_variance_penalty(noise, bg_mask):
    """
    Penalises cross-channel variance of the background noise.
    Forces the noise toward a grayscale tone, suppressing color artifacts.
    noise   : (B, 3, H, W)
    bg_mask : (B, 1, H, W)  — 1 where background (1 - object mask)
    """
    n = noise * bg_mask                        # zero out foreground
    mean_ch = n.mean(dim=[2, 3], keepdim=True) # per-channel spatial mean
    ch_var  = ((n - mean_ch) ** 2).mean(dim=[2, 3])   # (B, 3)
    # variance across the 3 channels (push them toward the same value)
    return ch_var.var(dim=1).mean()


def fabricate_loss(nf, adv_f, mask, yolo_model):
    """
    Multi-objective fabrication loss (background region: 1 - mask):
      Hinge   – push max logit up to HINGE_CAP (=1.0), then stop
                (prevents over-optimised, clumpy detections)
      L2      – penalise pixel magnitude         (λ = LAMBDA_L2  = 5.0)
      TV      – spatial smoothness               (γ = GAMMA_TV   = 0.5)
      Color   – cross-channel variance penalty   (λ = LAMBDA_COL = 1.0)
    """
    bg = 1 - mask
    lf   = utils.get_raw_logits(yolo_model, adv_f)
    aw_f = utils.mask_to_anchor_weights(bg)
    # Hinge: only penalise anchors whose max logit is below the cap
    hinge_part = (F.relu(config.HINGE_CAP - lf.max(dim=1)[0]) * aw_f).sum() \
                 / (aw_f.sum() + 1e-8)
    l2_part    = (nf * bg).pow(2).mean()
    tv_part    = total_variation(nf * bg)
    col_part   = color_variance_penalty(nf, bg)
    total = (hinge_part
             + config.LAMBDA_L2  * l2_part
             + config.GAMMA_TV   * tv_part
             + config.LAMBDA_COL * col_part)
    return total, hinge_part, l2_part, tv_part, col_part


def vanish_loss(nv, adv_v, mask, yolo_model):
    """
    Detection suppression loss with optional stealth regularizers.
    The regularizers default to 0.0 so existing behaviour is preserved
    until we explicitly tune them.
    """
    lv = utils.get_raw_logits(yolo_model, adv_v)
    aw_v = utils.mask_to_anchor_weights(mask)
    suppress_part = (F.softplus(lv).max(dim=1)[0] * aw_v).sum() / (aw_v.sum() + 1e-8)
    masked_noise = nv * mask
    l2_part = masked_noise.pow(2).mean()
    tv_part = total_variation(masked_noise)
    col_part = color_variance_penalty(nv, mask)
    total = (
        suppress_part
        + config.VANISH_LAMBDA_L2 * l2_part
        + config.VANISH_GAMMA_TV * tv_part
        + config.VANISH_LAMBDA_COL * col_part
    )
    return total, suppress_part, l2_part, tv_part, col_part


def _build_run_metadata(mode_label):
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_id": config.EXPERIMENT_ID,
        "mode": mode_label,
        "img_size": config.IMG_SIZE,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "epochs": config.EPOCHS,
        "early_stop_patience": config.EARLY_STOP_PATIENCE,
        "val_max_batches": config.VAL_MAX_BATCHES,
        "num_workers": config.NUM_WORKERS,
        "pin_memory": int(config.PIN_MEMORY),
        "prefetch_factor": config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else "",
        "persistent_workers": int(config.PERSISTENT_WORKERS) if config.NUM_WORKERS > 0 else "",
        "epsilon_v": config.EPSILON_V,
        "epsilon_f": config.EPSILON_F,
        "hinge_cap": config.HINGE_CAP,
        "lambda_l2": config.LAMBDA_L2,
        "gamma_tv": config.GAMMA_TV,
        "lambda_col": config.LAMBDA_COL,
        "vanish_lambda_l2": config.VANISH_LAMBDA_L2,
        "vanish_gamma_tv": config.VANISH_GAMMA_TV,
        "vanish_lambda_col": config.VANISH_LAMBDA_COL,
    }


def append_metrics_csv(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.exists(path)
    if file_exists:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            fieldnames = next(reader)
    else:
        fieldnames = list(row.keys())

    safe_row = {k: row.get(k, "") for k in fieldnames}
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(safe_row)


def log_validation_metrics(csv_path, run_meta, epoch, vm, run_v, run_f, es_v, es_f):
    row = dict(run_meta)
    row.update({
        "epoch": epoch,
        "run_v": int(run_v),
        "run_f": int(run_f),
        "v_best_loss": es_v.best if es_v is not None else "",
        "v_patience_counter": es_v.counter if es_v is not None else "",
        "v_stopped": int(es_v.stopped) if es_v is not None else "",
        "f_best_loss": es_f.best if es_f is not None else "",
        "f_patience_counter": es_f.counter if es_f is not None else "",
        "f_stopped": int(es_f.stopped) if es_f is not None else "",
    })
    row.update(vm)
    append_metrics_csv(csv_path, row)


def make_stats_panel(n_orig, n_adv, n_missed, n_new, n_changed,
                     attack_type, device, size=640):
    """
    Stats panel for a 2x3 grid cell (size x size).
    n_missed  : orig boxes with no IoU match in adv   (suppressed)
    n_new     : adv  boxes with no IoU match in orig  (hallucinated)
    n_changed : matched pairs where class label differs
    """
    from PIL import Image, ImageDraw, ImageFont
    canvas = Image.new("RGB", (size, size), (12, 12, 12))
    draw   = ImageDraw.Draw(canvas)

    try:
        f_h1  = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 34)
        f_lbl = ImageFont.truetype("C:/Windows/Fonts/arial.ttf",   24)
        f_val = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 44)
        f_sm  = ImageFont.truetype("C:/Windows/Fonts/arial.ttf",   20)
    except Exception:
        f_h1 = f_lbl = f_val = f_sm = ImageFont.load_default()

    cx, lx, rx = size // 2, 52, size - 52

    if attack_type == "vanish":
        title       = "VANISH  ATTACK"
        title_col   = (255, 90, 90)
        main_rate   = n_missed / max(n_orig, 1) * 100
        rate_label  = f"Suppression Rate:  {main_rate:.1f}%"
        rate_col    = (255, 80, 80) if n_missed > 0 else (120, 120, 120)
    else:
        title       = "FABRICATE  ATTACK"
        title_col   = (90, 200, 255)
        main_rate   = n_new / max(n_orig, 1) * 100
        rate_label  = f"Hallucination Rate:  {main_rate:.1f}%"
        rate_col    = (80, 230, 110) if n_new > 0 else (120, 120, 120)

    def div(y):
        draw.line([(lx, y), (rx, y)], fill=(55, 55, 55), width=1)

    def kv(y, label, val_str, val_col=(215, 215, 215)):
        draw.text((lx, y), label, fill=(140, 140, 140), font=f_lbl, anchor="lm")
        draw.text((rx,  y), val_str, fill=val_col,        font=f_lbl, anchor="rm")

    # ── Title ────────────────────────────────────────────
    y = 58
    draw.text((cx, y), title, fill=title_col, font=f_h1, anchor="mm")

    y += 42; div(y)

    # ── Detection counts ────────────────────────────────
    y += 38
    kv(y, "Original Det",  str(n_orig))
    y += 38
    kv(y, "After Attack",  str(n_adv))

    y += 30; div(y)

    # ── Delta breakdown ─────────────────────────────────
    y += 38
    miss_pct = f"{n_missed / max(n_orig, 1) * 100:.0f}%"
    miss_col = (255, 80,  80) if n_missed > 0 else (100, 100, 100)
    kv(y, "Missed  (suppressed)", f"-{n_missed}  ({miss_pct})", miss_col)

    y += 38
    new_pct  = f"{n_new / max(n_orig, 1) * 100:.0f}%"
    new_col  = (80, 230, 110) if n_new     > 0 else (100, 100, 100)
    kv(y, "New  (hallucinated)",  f"+{n_new}  ({new_pct})",     new_col)

    y += 38
    chg_col  = (255, 185, 50) if n_changed > 0 else (100, 100, 100)
    kv(y, "Class Changed",        str(n_changed),                chg_col)

    y += 30; div(y)

    # ── Summary rate ─────────────────────────────────────
    y += 42
    draw.text((cx, y), rate_label, fill=rate_col, font=f_sm, anchor="mm")

    return _t(np.array(canvas), device)


def _plot_tensor(result, device):
    """Ultralytics Result → CHW float32 tensor"""
    arr = result.plot()[:, :, ::-1].copy()   # BGR → RGB
    return _t(arr, device)


def match_detections(res_orig, res_adv, iou_thresh=0.45):
    """
    Greedy IoU matching between two Ultralytics Result objects.
    Returns (n_missed, n_new, n_class_changed).
      n_missed       : orig boxes with no match  in adv  (suppressed)
      n_new          : adv  boxes with no match  in orig (hallucinated)
      n_class_changed: matched pairs with different class label
    """
    boxes_o = res_orig.boxes.xyxy   # (N, 4)
    cls_o   = res_orig.boxes.cls    # (N,)
    boxes_a = res_adv.boxes.xyxy    # (M, 4)
    cls_a   = res_adv.boxes.cls     # (M,)

    n_o, n_a = len(boxes_o), len(boxes_a)

    if n_o == 0 and n_a == 0:
        return 0, 0, 0
    if n_o == 0:
        return 0, n_a, 0
    if n_a == 0:
        return n_o, 0, 0

    iou_mat = box_iou(boxes_o, boxes_a)          # (N, M)

    matched_o, matched_a = set(), set()
    n_changed = 0

    # collect (iou, i, j) sorted descending
    flat = iou_mat.cpu().numpy()
    pairs = sorted(
        ((flat[i, j], i, j) for i in range(n_o) for j in range(n_a)),
        reverse=True
    )
    for score, i, j in pairs:
        if score < iou_thresh:
            break
        if i in matched_o or j in matched_a:
            continue
        matched_o.add(i); matched_a.add(j)
        if int(cls_o[i].item()) != int(cls_a[j].item()):
            n_changed += 1

    return n_o - len(matched_o), n_a - len(matched_a), n_changed


def validate_and_visualize(yolo, G_v, G_f, val_loader, epoch,
                           run_v=True, run_f=True):
    """
    Saves up to two 2x3 grid images per epoch (one per active generator):
      Row 1: Original | Original Prediction | Noise Mask
      Row 2: Adv      | Adv Prediction      | Stats Panel
    """
    yolo.model.eval()
    if run_v: G_v.eval()
    if run_f: G_f.eval()
    os.makedirs("results", exist_ok=True)

    def norm(t):
        return (t - t.min()) / (t.max() - t.min() + 1e-8)

    with torch.no_grad():
        imgs, boxes_list = next(iter(val_loader))
        imgs = imgs.to(config.DEVICE)
        m    = utils.gt_to_mask(boxes_list, size=config.IMG_SIZE)
        img0 = imgs[0:1]
        m0   = m[0:1]

        res_orig  = yolo(img0, verbose=False)
        orig_pred = _plot_tensor(res_orig[0], config.DEVICE)
        n_o       = len(res_orig[0].boxes)
        vis_log   = []

        # ── Vanish ──────────────────────────────────────────
        if run_v:
            noise_v = G_v(img0)
            nv_c    = torch.clamp(noise_v, -config.EPSILON_V, config.EPSILON_V)
            adv_v   = torch.clamp(img0 + nv_c * m0, 0, 1)
            res_v   = yolo(adv_v, verbose=False)

            mv, nv_new, nv_chg = match_detections(res_orig[0], res_v[0])
            n_av    = len(res_v[0].boxes)
            panel_v = make_stats_panel(
                n_o, n_av, mv, nv_new, nv_chg, "vanish", config.DEVICE, size=config.IMG_SIZE
            )
            row1_v  = torch.cat([img0[0], orig_pred, norm(nv_c)[0]], dim=2)
            row2_v  = torch.cat([adv_v[0], _plot_tensor(res_v[0], config.DEVICE), panel_v], dim=2)
            save_image(torch.cat([row1_v, row2_v], dim=1), f"results/ep{epoch:03d}_vanish.png")
            vis_log.append(f"Vanish {n_o}->{n_av} (miss={mv} new={nv_new} chg={nv_chg})")

        # ── Fabricate ────────────────────────────────────────
        if run_f:
            noise_f = G_f(img0)
            nf_c    = torch.clamp(noise_f, -config.EPSILON_F, config.EPSILON_F)
            adv_f   = torch.clamp(img0 + nf_c * (1 - m0), 0, 1)
            res_f   = yolo(adv_f, verbose=False)

            mf, nf_new, nf_chg = match_detections(res_orig[0], res_f[0])
            n_af    = len(res_f[0].boxes)
            panel_f = make_stats_panel(
                n_o, n_af, mf, nf_new, nf_chg, "fabricate", config.DEVICE, size=config.IMG_SIZE
            )
            row1_f  = torch.cat([img0[0], orig_pred, norm(nf_c)[0]], dim=2)
            row2_f  = torch.cat([adv_f[0], _plot_tensor(res_f[0], config.DEVICE), panel_f], dim=2)
            save_image(torch.cat([row1_f, row2_f], dim=1), f"results/ep{epoch:03d}_fabricate.png")
            vis_log.append(f"Fabricate {n_o}->{n_af} (miss={mf} new={nf_new} chg={nf_chg})")

    print("  [Vis] " + " | ".join(vis_log))

# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=5, name=""):
        self.patience = patience
        self.name     = name
        self.best     = float('inf')
        self.counter  = 0
        self.stopped  = False

    def step(self, val_loss):
        if val_loss < self.best - 1e-6:
            self.best    = val_loss
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience and not self.stopped:
            self.stopped = True
            print(f"  [EarlyStop] {self.name} stopped "
                  f"(best={self.best:.4f}, no improvement for {self.patience} epochs)")
        return False


# ── Validation 지표 계산 ──────────────────────────────────────────────────────

def compute_val_metrics(yolo, G_v, G_f, val_loader, max_batches=None,
                        run_v=True, run_f=True):
    """
    Averages the following metrics over up to max_batches val batches:
      lv            - Vanish    val loss          (early-stop criterion)
      lf            - Fabricate val loss          (early-stop criterion)
      v_missed      - avg suppressed detections   (Vanish attack power)
      v_new         - avg hallucinated detections (Vanish side-effect)
      v_cls_chg     - avg class-changed matches   (Vanish)
      f_missed      - avg suppressed detections   (Fabricate side-effect)
      f_new         - avg hallucinated detections (Fabricate attack power)
      f_cls_chg     - avg class-changed matches   (Fabricate)
      psnr_v        - Vanish    image quality dB  (stealth)
      psnr_f        - Fabricate image quality dB  (stealth)
      linf_v        - Vanish    noise L-inf norm
      linf_f        - Fabricate noise L-inf norm
    """
    max_batches = max_batches or config.VAL_MAX_BATCHES
    yolo.model.eval()
    if run_v: G_v.eval()
    if run_f: G_f.eval()
    acc = dict(lv=0.0, lf=0.0,
               v_missed=0.0, v_new=0.0, v_cls_chg=0.0,
               f_missed=0.0, f_new=0.0, f_cls_chg=0.0,
               psnr_v=0.0,   psnr_f=0.0,
               linf_v=0.0,   linf_f=0.0)
    n = 0

    with torch.no_grad():
        for i, (imgs, boxes_list) in enumerate(val_loader):
            if i >= max_batches:
                break
            imgs = imgs.to(config.DEVICE)
            mask = utils.gt_to_mask(boxes_list, size=config.IMG_SIZE)
            r0   = yolo(imgs[0:1], verbose=False)

            if run_v:
                nv     = G_v(imgs)
                nv_c   = torch.clamp(nv, -config.EPSILON_V, config.EPSILON_V)
                adv_v  = torch.clamp(imgs + nv_c * mask, 0, 1)
                loss_v, _, _, _, _ = vanish_loss(nv_c, adv_v, mask, yolo.model)
                rv     = yolo(adv_v[0:1], verbose=False)
                vm, vn, vc = match_detections(r0[0], rv[0])
                psnr_v = 10 * np.log10(1.0 / (F.mse_loss(adv_v, imgs).item() + 1e-10))
                linf_v = (nv_c * mask).abs().amax().item()
                acc['lv'] += loss_v.item()
                acc['v_missed'] += vm; acc['v_new'] += vn; acc['v_cls_chg'] += vc
                acc['psnr_v'] += psnr_v; acc['linf_v'] += linf_v

            if run_f:
                nf     = G_f(imgs)
                nf_c   = torch.clamp(nf, -config.EPSILON_F, config.EPSILON_F)
                adv_f  = torch.clamp(imgs + nf_c * (1 - mask), 0, 1)
                loss_f, _, _, _, _ = fabricate_loss(nf_c, adv_f, mask, yolo.model)
                rf     = yolo(adv_f[0:1], verbose=False)
                fm, fn, fc = match_detections(r0[0], rf[0])
                psnr_f = 10 * np.log10(1.0 / (F.mse_loss(adv_f, imgs).item() + 1e-10))
                linf_f = (nf_c * (1 - mask)).abs().amax().item()
                acc['lf'] += loss_f.item()
                acc['f_missed'] += fm; acc['f_new'] += fn; acc['f_cls_chg'] += fc
                acc['psnr_f'] += psnr_f; acc['linf_f'] += linf_f

            n += 1

    return {k: v / n for k, v in acc.items()}


# ── Mode selector ─────────────────────────────────────────────────────────────

def select_mode() -> int:
    sep = "-" * 52
    print(f"\n{sep}")
    print("  VisDrone Adversarial Training")
    print(sep)
    print("  1) Vanish only    (G_v)")
    print("  2) Fabricate only (G_f)")
    print("  3) Both           (G_v + G_f)")
    print(sep)
    while True:
        raw = input("  Select mode [1/2/3]: ").strip()
        if raw in ("1", "2", "3"):
            return int(raw)
        print("  Invalid choice. Please enter 1, 2, or 3.")


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    yolo = YOLO(config.YOLO_PATH).to(config.DEVICE)
    yolo.model.eval()
    for p in yolo.model.parameters():
        p.requires_grad = False

    # ── Mode selection ───────────────────────────────────────────────────────
    mode = select_mode()
    run_v = mode in (1, 3)
    run_f = mode in (2, 3)
    mode_label = {1: "Vanish only", 2: "Fabricate only", 3: "Both"}[mode]

    # ── Instantiate only required generators ────────────────────────────────
    G_v   = model.MobileUNetGenerator().to(config.DEVICE) if run_v else None
    G_f   = model.MobileUNetGenerator().to(config.DEVICE) if run_f else None
    opt_v = torch.optim.AdamW(G_v.parameters(), lr=config.LEARNING_RATE) if run_v else None
    opt_f = torch.optim.AdamW(G_f.parameters(), lr=config.LEARNING_RATE) if run_f else None

    loader_kwargs = {
        "batch_size": config.BATCH_SIZE,
        "num_workers": config.NUM_WORKERS,
        "pin_memory": config.PIN_MEMORY,
        "collate_fn": collate_fn,
    }
    if config.NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = config.PREFETCH_FACTOR
        loader_kwargs["persistent_workers"] = config.PERSISTENT_WORKERS

    train_loader = DataLoader(
        dataset.VisDroneYOLODataset(config.TRAIN_DIR, size=config.IMG_SIZE),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        dataset.VisDroneYOLODataset(config.VAL_DIR, size=config.IMG_SIZE),
        shuffle=False,
        **loader_kwargs,
    )

    es_v = EarlyStopping(patience=config.EARLY_STOP_PATIENCE, name="G_v (Vanish)")    if run_v else None
    es_f = EarlyStopping(patience=config.EARLY_STOP_PATIENCE, name="G_f (Fabricate)") if run_f else None
    run_meta = _build_run_metadata(mode_label)

    sep = "-" * 72
    print(f"\n>>> VisDrone HBB Attack | Mode: {mode_label} | "
          f"All classes ({len(yolo.names)} cls)")
    print(f">>> Experiment ID: {config.EXPERIMENT_ID}")
    print(f">>> CSV Log Path : {config.RESULTS_CSV}")

    for epoch in range(1, config.EPOCHS + 1):
        v_done = (not run_v) or es_v.stopped
        f_done = (not run_f) or es_f.stopped
        if v_done and f_done:
            print(f"\n{sep}\nAll active generators early-stopped. Training complete.\n{sep}")
            break

        if run_v and not es_v.stopped: G_v.train()
        if run_f and not es_f.stopped: G_f.train()
        loop = tqdm(train_loader, desc=f"Ep{epoch:03d}")

        for imgs, boxes_list in loop:
            imgs = imgs.to(config.DEVICE)
            mask = utils.gt_to_mask(boxes_list, size=config.IMG_SIZE)
            pf   = {}

            # ── Vanish branch (logic and hyperparameters unchanged) ───────
            if run_v and not es_v.stopped:
                nv     = G_v(imgs)
                nv_c   = torch.clamp(nv, -config.EPSILON_V, config.EPSILON_V)
                adv_v  = torch.clamp(imgs + nv_c * mask, 0, 1)
                loss_v, suppress_v, l2_v, tv_v, col_v = vanish_loss(nv_c, adv_v, mask, yolo.model)
                opt_v.zero_grad(); loss_v.backward(); opt_v.step()
                pf['V_loss'] = f"{loss_v.item():.3f}"
                pf['V_sup'] = f"{suppress_v.item():.3f}"
                pf['V_tv'] = f"{tv_v.item():.4f}"
                pf['V_Linf'] = f"{(nv_c * mask).abs().amax().item():.3f}"

            # ── Fabricate branch (background-only, stealth-first) ─────────
            if run_f and not es_f.stopped:
                nf    = G_f(imgs)
                nf_c  = torch.clamp(nf, -config.EPSILON_F, config.EPSILON_F)
                adv_f = torch.clamp(imgs + nf_c * (1 - mask), 0, 1)
                loss_f, hinge, l2, tv, col = fabricate_loss(nf_c, adv_f, mask, yolo.model)
                opt_f.zero_grad(); loss_f.backward(); opt_f.step()
                pf['F_loss']  = f"{loss_f.item():.3f}"
                pf['F_hinge'] = f"{hinge.item():.3f}"
                pf['F_col']   = f"{col.item():.4f}"
                pf['F_Linf']  = f"{(nf_c * (1 - mask)).abs().amax().item():.3f}"

            loop.set_postfix(**pf)

        # ── Validation ──────────────────────────────────────────────────────
        print(f"\n{sep}")
        print(f"  Epoch {epoch:03d}  |  Validation  [{mode_label}]")
        print(sep)
        vm = compute_val_metrics(yolo, G_v, G_f, val_loader, run_v=run_v, run_f=run_f)

        if run_v:
            imp_v = es_v.step(vm['lv']) if not es_v.stopped else False
            tag_v = " * BEST" if imp_v else f"  patience {es_v.counter}/{es_v.patience}"
            print(f"  [Vanish   ]{tag_v}")
            print(f"    loss={vm['lv']:.4f}  missed={vm['v_missed']:.1f}  "
                  f"new={vm['v_new']:.1f}  cls_chg={vm['v_cls_chg']:.1f}  "
                  f"PSNR={vm['psnr_v']:.1f}dB  Linf={vm['linf_v']:.4f}")
            if imp_v: torch.save(G_v.state_dict(), "G_v_best.pth")

        if run_f:
            imp_f = es_f.step(vm['lf']) if not es_f.stopped else False
            tag_f = " * BEST" if imp_f else f"  patience {es_f.counter}/{es_f.patience}"
            print(f"  [Fabricate]{tag_f}")
            print(f"    loss={vm['lf']:.4f}  missed={vm['f_missed']:.1f}  "
                  f"new={vm['f_new']:.1f}  cls_chg={vm['f_cls_chg']:.1f}  "
                  f"PSNR={vm['psnr_f']:.1f}dB  Linf={vm['linf_f']:.4f}")
            if imp_f: torch.save(G_f.state_dict(), "G_f_best.pth")

        log_validation_metrics(
            config.RESULTS_CSV,
            run_meta,
            epoch,
            vm,
            run_v,
            run_f,
            es_v,
            es_f,
        )

        print(sep)

        # ── Visualisation ────────────────────────────────────────────────────
        validate_and_visualize(yolo, G_v, G_f, val_loader, epoch,
                               run_v=run_v, run_f=run_f)

        if epoch % 10 == 0:
            if run_v: torch.save(G_v.state_dict(), f"G_v_ep{epoch:03d}.pth")
            if run_f: torch.save(G_f.state_dict(), f"G_f_ep{epoch:03d}.pth")


if __name__ == "__main__":
    train()
