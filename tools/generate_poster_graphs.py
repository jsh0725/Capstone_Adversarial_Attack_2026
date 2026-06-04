from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "graphs" / "poster"
IMAGE_CSV = ROOT / "outputs" / "image_eval" / "visdrone" / "val_4mode_metrics.csv"
VIDEO_CSV = ROOT / "outputs" / "video_eval" / "visdrone" / "color_overlay" / "video_summary.csv"

VARIANT_ORDER = ["original", "fabricate", "vanish", "janusnet"]
VARIANT_LABELS = {
    "original": "Original",
    "fabricate": "Fabricate",
    "vanish": "Vanish",
    "janusnet": "JanusNet",
}
COLORS = {
    "original": "#7f7f7f",
    "fabricate": "#b8b8b8",
    "vanish": "#d0d0d0",
    "janusnet": "#f2c94c",
}


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(path)


def annotate(ax, bars, fmt="{:.2f}", pad_ratio=0.012):
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * pad_ratio
    for bar in bars:
        h = bar.get_height()
        if not np.isfinite(h):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + pad,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#111111",
        )


def load_ordered(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["variant"] = df["variant"].str.lower()
    df = df[df["variant"].isin(VARIANT_ORDER)].copy()
    df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)
    df = df.sort_values("variant").reset_index(drop=True)
    return df


def plot_image_detection(df):
    metrics = ["map50", "precision", "recall", "f1"]
    labels = ["mAP50 ↑", "Precision ↑", "Recall ↑", "F1 ↑"]
    x = np.arange(len(metrics))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for i, variant in enumerate(VARIANT_ORDER):
        row = df[df["variant"] == variant].iloc[0]
        values = [float(row[m]) for m in metrics]
        bars = ax.bar(
            x + (i - 1.5) * width,
            values,
            width,
            label=VARIANT_LABELS[variant],
            color=COLORS[variant],
        )
        annotate(ax, bars, fmt="{:.2f}", pad_ratio=0.006)

    ax.set_title("VisDrone Image Evaluation: Detection Performance")
    ax.set_ylabel("Score (higher is better ↑)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.68)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=9)
    save(fig, "visdrone_image_detection_comparison.png")


def plot_image_quality(df):
    attacks = ["fabricate", "vanish", "janusnet"]
    labels = [VARIANT_LABELS[v] for v in attacks]

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.2))
    panels = [
        ("lpips", "LPIPS ↓", "{:.2f}", None),
        ("ssim", "SSIM ↑", "{:.2f}", (0.98, 1.002)),
        ("psnr", "PSNR (dB) ↑", "{:.2f}", None),
        ("avg_sec_per_image", "Speed (s/img) ↓", "{:.2f}", None),
    ]

    for ax, (col, title, fmt, ylim) in zip(axes, panels):
        values = [float(df[df["variant"] == v].iloc[0][col]) for v in attacks]
        bars = ax.bar(labels, values, color=[COLORS[v] for v in attacks])
        annotate(ax, bars, fmt=fmt)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=18)

    fig.suptitle("VisDrone Image Evaluation: Visual Quality and Runtime", y=1.05)
    save(fig, "visdrone_image_quality_runtime_comparison.png")


def plot_video_detection(df):
    metrics = ["orig_det_per_frame", "adv_det_per_frame", "missed_per_frame", "new_per_frame"]
    labels = ["Original det", "After attack", "Missed ↑", "New ↑"]
    x = np.arange(len(metrics))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    for i, variant in enumerate(VARIANT_ORDER):
        row = df[df["variant"] == variant].iloc[0]
        values = [float(row[m]) for m in metrics]
        bars = ax.bar(
            x + (i - 1.5) * width,
            values,
            width,
            label=VARIANT_LABELS[variant],
            color=COLORS[variant],
        )
        annotate(ax, bars, fmt="{:.2f}", pad_ratio=0.005)

    ax.set_title("VisDrone Video Evaluation: Detection Changes")
    ax.set_ylabel("Detections per frame")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=9)
    save(fig, "visdrone_video_detection_comparison.png")


def plot_video_quality(df):
    attacks = ["fabricate", "vanish", "janusnet"]
    labels = [VARIANT_LABELS[v] for v in attacks]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))
    panels = [
        ("psnr", "PSNR (dB) ↑", "{:.2f}"),
        ("linf", "Linf ↓", "{:.2f}"),
        ("sec_per_frame", "Speed (s/frame) ↓", "{:.2f}"),
    ]

    for ax, (col, title, fmt) in zip(axes, panels):
        values = [float(df[df["variant"] == v].iloc[0][col]) for v in attacks]
        bars = ax.bar(labels, values, color=[COLORS[v] for v in attacks])
        annotate(ax, bars, fmt=fmt)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=18)

    fig.suptitle("VisDrone Video Evaluation: Visual Quality and Runtime", y=1.05)
    save(fig, "visdrone_video_quality_runtime_comparison.png")


def main():
    if not IMAGE_CSV.exists():
        raise FileNotFoundError(f"Image CSV not found: {IMAGE_CSV}")
    if not VIDEO_CSV.exists():
        raise FileNotFoundError(f"Video CSV not found: {VIDEO_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_df = load_ordered(IMAGE_CSV)
    video_df = load_ordered(VIDEO_CSV)

    image_df.to_csv(OUT_DIR / "visdrone_image_metrics_used.csv", index=False)
    video_df.to_csv(OUT_DIR / "visdrone_video_metrics_used.csv", index=False)

    plot_image_detection(image_df)
    plot_image_quality(image_df)
    plot_video_detection(video_df)
    plot_video_quality(video_df)


if __name__ == "__main__":
    main()
