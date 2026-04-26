import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import yaml

import config
import model


# -----------------------------------------------------------------------------
# 생성 전용 스크립트
# - 목적: Original/Fabricate/Vanish/Janusnet 공격 이미지를 variant별 폴더로 생성
# - 출력: eval_variants/<variant>/images/<split>, labels/<split>, <variant>_<split>.yaml
# - 주의: 이 파일은 "이미지 생성"만 수행하며 detector 성능 평가는 하지 않음
# -----------------------------------------------------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    # CLI 입력 인자 정의
    p = argparse.ArgumentParser(description="Generate perturbed images for Original/Fabricate/Vanish/Janusnet.")
    p.add_argument("--data", required=True, help="Path to dataset yaml (e.g., vhrv/vhrv.yaml)")
    p.add_argument("--gv", default="", help="Vanish generator checkpoint (.pth)")
    p.add_argument("--gf", default="", help="Fabricate generator checkpoint (.pth)")
    p.add_argument("--eps-v", type=float, default=12 / 255.0)
    p.add_argument("--eps-f", type=float, default=4 / 255.0)
    p.add_argument("--variants", nargs="+", default=["original", "fabricate", "vanish", "janusnet"])
    p.add_argument("--device", default="", help="cuda:0 or cpu (empty = auto)")
    p.add_argument("--outdir", default="eval_variants")
    return p.parse_args()


def resolve_data_paths(data_yaml: Path):
    # 원본 데이터 yaml에서 val/test split 경로를 파싱해 실제 이미지/라벨 경로 결정
    with open(data_yaml, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    base = Path(d.get("path", data_yaml.parent)).expanduser()
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()

    val_rel = d.get("val", "images/val")
    test_rel = d.get("test", "")
    val_images = (base / val_rel).resolve()
    test_images = (base / test_rel).resolve() if test_rel else None

    split = "val"
    images_dir = val_images
    if not images_dir.exists() and test_images and test_images.exists():
        split = "test"
        images_dir = test_images
    if not images_dir.exists():
        raise FileNotFoundError(f"Could not find val/test image dir from yaml: {data_yaml}")

    labels_dir = (base / "labels" / split).resolve()
    if not labels_dir.exists():
        raise FileNotFoundError(f"Label dir not found: {labels_dir}")

    names = d.get("names", [])
    nc = int(d.get("nc", len(names) if names else 0))
    return split, images_dir, labels_dir, nc, names


def list_images(images_dir: Path):
    # 지원 확장자만 재귀적으로 수집
    imgs = [p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    return sorted(imgs)


def image_to_tensor(path: Path, device):
    # PIL 이미지를 [0,1] 텐서(B, C, H, W)로 변환
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def tensor_to_image(t):
    # 텐서를 uint8 이미지로 복원
    t = t.squeeze(0).detach().cpu().clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def read_yolo_labels(label_path: Path):
    # YOLO txt 라벨(class xc yc bw bh)을 파싱
    rows = []
    if not label_path.exists():
        return rows
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = line.split()
            if len(vals) < 5:
                continue
            cls, xc, yc, bw, bh = vals[:5]
            rows.append((int(float(cls)), float(xc), float(yc), float(bw), float(bh)))
    return rows


def boxes_to_mask(label_rows, h, w, device):
    # 라벨 bbox를 객체 마스크(1:객체, 0:배경)로 변환
    m = torch.zeros((1, 1, h, w), device=device)
    for _, xc, yc, bw, bh in label_rows:
        x1 = int((xc - bw / 2.0) * w)
        y1 = int((yc - bh / 2.0) * h)
        x2 = int((xc + bw / 2.0) * w)
        y2 = int((yc + bh / 2.0) * h)
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        if x2 > x1 and y2 > y1:
            m[:, :, y1:y2, x1:x2] = 1.0
    return m


def pad_to_multiple(t, mult=4):
    # U-Net skip 연결 shape mismatch 방지를 위해 배수 패딩
    _, _, h, w = t.shape
    pad_h = (mult - (h % mult)) % mult
    pad_w = (mult - (w % mult)) % mult
    if pad_h == 0 and pad_w == 0:
        return t, (0, 0)
    t_pad = F.pad(t, (0, pad_w, 0, pad_h), mode="replicate")
    return t_pad, (pad_h, pad_w)


def crop_from_padding(t, pad_hw):
    # 패딩했던 영역을 원본 해상도로 복원
    pad_h, pad_w = pad_hw
    if pad_h > 0:
        t = t[:, :, :-pad_h, :]
    if pad_w > 0:
        t = t[:, :, :, :-pad_w]
    return t


def make_variant_root(outdir: Path, variant: str, split: str):
    # variant별 출력 디렉터리 생성
    root = outdir / variant
    images = root / "images" / split
    labels = root / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    return root, images, labels


def write_variant_yaml(path: Path, root: Path, split: str, nc: int, names):
    # 생성된 variant 데이터셋을 detector val에 바로 넣을 수 있도록 yaml 생성
    d = {
        "path": root.as_posix(),
        "train": f"images/{split}",
        "val": f"images/{split}",
        "test": f"images/{split}",
        "nc": nc,
        "names": names if names else [str(i) for i in range(nc)],
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, sort_keys=False)


def save_label_copy(src: Path, dst: Path):
    # 원본 라벨을 variant 라벨 폴더로 복사 (없으면 빈 파일 생성)
    if src.exists():
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        dst.write_text("", encoding="utf-8")


def generate_variant_dataset(variant, images, labels_dir, outdir, split, g_v, g_f, eps_v, eps_f, device):
    # 핵심 생성 루프:
    # - original: 원본 유지
    # - vanish: 객체영역(mask)에 G_v 노이즈 적용
    # - fabricate: 배경영역(1-mask)에 G_f 노이즈 적용
    # - janusnet: 객체는 G_v, 배경은 G_f를 합성 적용
    root, img_out, lbl_out = make_variant_root(outdir, variant, split)

    for img_path in images:
        rel = img_path.name
        lbl_src = labels_dir / (img_path.stem + ".txt")
        lbl_dst = lbl_out / (img_path.stem + ".txt")
        save_label_copy(lbl_src, lbl_dst)

        orig = image_to_tensor(img_path, device)
        _, _, h, w = orig.shape
        label_rows = read_yolo_labels(lbl_src)
        mask = boxes_to_mask(label_rows, h, w, device)

        orig_pad, pad_hw = pad_to_multiple(orig, mult=4)
        mask_pad, _ = pad_to_multiple(mask, mult=4)

        adv = orig.clone()
        with torch.no_grad():
            if variant == "vanish":
                if g_v is None:
                    raise RuntimeError("Vanish variant requested but --gv is not provided.")
                nv = g_v(orig_pad)
                nv_c = torch.clamp(nv, -eps_v, eps_v)
                adv_pad = torch.clamp(orig_pad + nv_c * mask_pad, 0, 1)
                adv = crop_from_padding(adv_pad, pad_hw)
            elif variant == "fabricate":
                if g_f is None:
                    raise RuntimeError("Fabricate variant requested but --gf is not provided.")
                nf = g_f(orig_pad)
                nf_c = torch.clamp(nf, -eps_f, eps_f)
                adv_pad = torch.clamp(orig_pad + nf_c * (1 - mask_pad), 0, 1)
                adv = crop_from_padding(adv_pad, pad_hw)
            elif variant == "janusnet":
                if g_v is None or g_f is None:
                    raise RuntimeError("Janusnet variant requires both --gv and --gf.")
                nv = g_v(orig_pad)
                nf = g_f(orig_pad)
                nv_c = torch.clamp(nv, -eps_v, eps_v)
                nf_c = torch.clamp(nf, -eps_f, eps_f)
                adv_pad = torch.clamp(orig_pad + nv_c * mask_pad + nf_c * (1 - mask_pad), 0, 1)
                adv = crop_from_padding(adv_pad, pad_hw)
            elif variant == "original":
                adv = orig
            else:
                raise ValueError(f"Unknown variant: {variant}")

        # 최종 공격 이미지를 variant 폴더에 저장
        tensor_to_image(adv).save(img_out / rel)

    return root


def main():
    # 실행 순서:
    # 1) 데이터/디바이스/체크포인트 로딩
    # 2) variant별 이미지 생성
    # 3) variant별 yaml 생성
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    data_yaml = Path(args.data).resolve()
    split, images_dir, labels_dir, nc, names = resolve_data_paths(data_yaml)
    images = list_images(images_dir)
    if not images:
        raise RuntimeError(f"No images found in {images_dir}")

    device = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device)

    config.EPSILON_V = float(args.eps_v)
    config.EPSILON_F = float(args.eps_f)

    g_v = None
    g_f = None
    if args.gv:
        g_v = model.MobileUNetGenerator().to(torch_device).eval()
        g_v.load_state_dict(torch.load(args.gv, map_location=torch_device))
    if args.gf:
        g_f = model.MobileUNetGenerator().to(torch_device).eval()
        g_f.load_state_dict(torch.load(args.gf, map_location=torch_device))

    for variant in args.variants:
        v = variant.lower()
        root = generate_variant_dataset(
            variant=v,
            images=images,
            labels_dir=labels_dir,
            outdir=outdir,
            split=split,
            g_v=g_v,
            g_f=g_f,
            eps_v=float(args.eps_v),
            eps_f=float(args.eps_f),
            device=torch_device,
        )
        yaml_path = outdir / f"{v}_{split}.yaml"
        write_variant_yaml(yaml_path, root, split, nc, names)
        print(f"[Done] {v}: images={root / 'images' / split} | yaml={yaml_path}")


if __name__ == "__main__":
    main()
