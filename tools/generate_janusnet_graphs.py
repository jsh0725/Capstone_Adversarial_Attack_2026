from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "graphs" / "janusnet"


def _read_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _safe_float(value):
    if isinstance(value, str) and value.lower() == "inf":
        return np.inf
    return float(value)


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(path)


def _annotate_bars(ax, bars, fmt="{:.3f}", y_pad=0.01):
    ylim = ax.get_ylim()
    pad = (ylim[1] - ylim[0]) * y_pad
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + pad,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def collect_image_janusnet_rows():
    candidates = [
        ("VHRV", ROOT / "outputs" / "image_eval" / "vhrv" / "val_4mode_metrics.csv"),
        ("VisDrone", ROOT / "outputs" / "image_eval" / "visdrone" / "val_4mode_metrics.csv"),
    ]
    rows = []
    for dataset, path in candidates:
        if not path.exists():
            continue
        df = _read_csv(path)
        if "variant" not in df.columns:
            continue
        hit = df[df["variant"].astype(str).str.lower() == "janusnet"]
        if hit.empty:
            continue
        row = hit.iloc[0].copy()
        row["dataset"] = dataset
        row["source_csv"] = str(path.relative_to(ROOT))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_image_metrics(image_df):
    if image_df.empty:
        return

    image_df = image_df.copy()
    for col in ["map50", "precision", "recall", "f1", "lpips", "ssim", "psnr", "avg_sec_per_image"]:
        if col in image_df.columns:
            image_df[col] = image_df[col].map(_safe_float)

    x = np.arange(len(image_df))
    width = 0.18
    metrics = ["map50", "precision", "recall", "f1"]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for idx, metric in enumerate(metrics):
        bars = ax.bar(x + (idx - 1.5) * width, image_df[metric], width, label=metric.upper())
        _annotate_bars(ax, bars)
    ax.set_title("JanusNet Image Evaluation: Detection Metrics")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(image_df["dataset"])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=8)
    _save(fig, "janusnet_image_detection_metrics.png")

    metrics = ["lpips", "ssim", "psnr"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    for ax, metric in zip(axes, metrics):
        bars = ax.bar(image_df["dataset"], image_df[metric], color="#3b6ea8")
        _annotate_bars(ax, bars, fmt="{:.2f}" if metric == "psnr" else "{:.3f}")
        ax.set_title(metric.upper())
        ax.grid(axis="y", alpha=0.25)
        if metric == "ssim":
            ax.set_ylim(0.95, 1.005)
    fig.suptitle("JanusNet Image Evaluation: Visual Quality Metrics", y=1.03)
    _save(fig, "janusnet_image_quality_metrics.png")


def collect_video_janusnet_rows():
    rows = []
    for path in sorted((ROOT / "outputs" / "video_eval").rglob("video_summary.csv")):
        df = _read_csv(path)
        if "variant" not in df.columns:
            continue
        hit = df[df["variant"].astype(str).str.lower() == "janusnet"]
        if hit.empty:
            continue
        row = hit.iloc[0].copy()
        row["run"] = path.parent.name
        row["source_csv"] = str(path.relative_to(ROOT))
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    numeric_cols = [
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
    for col in numeric_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda v: _safe_float(v) if pd.notna(v) else np.nan)
    return out


def plot_video_summary(video_df):
    if video_df.empty:
        return

    df = video_df.sort_values(["run"]).reset_index(drop=True)
    labels = df["run"].astype(str).tolist()
    x = np.arange(len(df))
    width = 0.22

    fig, ax = plt.subplots(figsize=(max(9, len(df) * 1.15), 5.2))
    for idx, metric in enumerate(["orig_det_per_frame", "adv_det_per_frame", "missed_per_frame", "new_per_frame"]):
        bars = ax.bar(x + (idx - 1.5) * width, df[metric], width, label=metric.replace("_per_frame", ""))
        if len(df) <= 6:
            _annotate_bars(ax, bars, fmt="{:.1f}", y_pad=0.004)
    ax.set_title("JanusNet Video Evaluation: Detection Changes")
    ax.set_ylabel("Detections per frame")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=8)
    _save(fig, "janusnet_video_detection_summary.png")

    fig, ax1 = plt.subplots(figsize=(max(9, len(df) * 1.15), 5.2))
    bars = ax1.bar(x - 0.15, df["psnr"], 0.3, label="PSNR", color="#3b6ea8")
    ax1.set_ylabel("PSNR (dB)")
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    line1 = ax2.plot(x + 0.15, df["sec_per_frame"], marker="o", label="sec/frame", color="#d95f02")
    ax2.set_ylabel("Seconds per frame")
    ax1.set_title("JanusNet Video Evaluation: Quality and Runtime")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    _annotate_bars(ax1, bars, fmt="{:.1f}", y_pad=0.005)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=8)
    _save(fig, "janusnet_video_quality_runtime_summary.png")

    if "limited_frame_ratio" in df.columns and df["limited_frame_ratio"].notna().any():
        limited = df[df["limited_frame_ratio"].notna()].copy()
        if not limited.empty:
            fig, ax = plt.subplots(figsize=(max(7, len(limited) * 1.2), 4.6))
            bars = ax.bar(limited["run"], limited["limited_frame_ratio"], color="#7b9e3f")
            _annotate_bars(ax, bars, fmt="{:.2f}")
            ax.set_title("JanusNet Saturation Control: Limited Frame Ratio")
            ax.set_ylabel("Ratio")
            ax.set_ylim(0, min(1.0, max(0.1, limited["limited_frame_ratio"].max() * 1.25)))
            ax.tick_params(axis="x", labelrotation=35)
            ax.grid(axis="y", alpha=0.25)
            _save(fig, "janusnet_limit_ratio_summary.png")


def plot_latest_frame_trend():
    frame_csvs = sorted(
        (ROOT / "outputs" / "video_eval").rglob("video_frame_metrics.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not frame_csvs:
        return

    path = frame_csvs[0]
    df = _read_csv(path)
    if "variant" not in df.columns:
        return
    df = df[df["variant"].astype(str).str.lower() == "janusnet"].copy()
    if df.empty:
        return

    for col in ["frame", "orig_det", "adv_det", "missed", "new", "psnr", "sec_per_frame"]:
        if col in df.columns:
            df[col] = df[col].map(_safe_float)
    df = df.sort_values("frame")
    window = max(5, min(50, int(len(df) * 0.04)))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for col in ["orig_det", "adv_det", "missed", "new"]:
        axes[0].plot(df["frame"], df[col].rolling(window, min_periods=1).mean(), label=col)
    axes[0].set_title(f"JanusNet Per-frame Detection Trend ({path.parent.name})")
    axes[0].set_ylabel(f"Rolling mean, window={window}")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8)

    axes[1].plot(df["frame"], df["psnr"].rolling(window, min_periods=1).mean(), color="#3b6ea8", label="PSNR")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    _save(fig, "janusnet_latest_video_frame_trend.png")


def plot_poster_ready_single_results(image_df, video_df):
    if not image_df.empty:
        vis = image_df[image_df["dataset"].astype(str).str.lower() == "visdrone"]
        if not vis.empty:
            row = vis.iloc[0]
            metrics = ["map50", "precision", "recall", "f1"]
            values = [_safe_float(row[m]) for m in metrics]
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            bars = ax.bar([m.upper() for m in metrics], values, color=["#2b7bba", "#58a14e", "#f28e2b", "#d64f4f"])
            _annotate_bars(ax, bars)
            ax.set_title("JanusNet on VisDrone: Detection Metrics")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 0.55)
            ax.grid(axis="y", alpha=0.25)
            _save(fig, "janusnet_visdrone_image_detection_metrics.png")

            q_metrics = ["lpips", "ssim", "psnr", "avg_sec_per_image"]
            q_values = [_safe_float(row[m]) for m in q_metrics]
            fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.8))
            for ax, metric, value in zip(axes, q_metrics, q_values):
                bars = ax.bar([metric.upper().replace("AVG_SEC_PER_IMAGE", "SEC/IMG")], [value], color="#3b6ea8")
                _annotate_bars(ax, bars, fmt="{:.2f}" if metric == "psnr" else "{:.4f}")
                ax.grid(axis="y", alpha=0.25)
                if metric == "ssim":
                    ax.set_ylim(0.95, 1.005)
            fig.suptitle("JanusNet on VisDrone: Quality and Runtime", y=1.04)
            _save(fig, "janusnet_visdrone_image_quality_runtime.png")

    if not video_df.empty:
        # Prefer the latest saturation-controlled VisDrone run if it exists.
        preferred = video_df[video_df["run"].astype(str).str.contains("Limit_80_Area100", case=False, na=False)]
        row = preferred.iloc[-1] if not preferred.empty else video_df.iloc[-1]

        metrics = ["orig_det_per_frame", "adv_det_per_frame", "missed_per_frame", "new_per_frame"]
        labels = ["Original det", "After attack", "Missed", "New"]
        values = [_safe_float(row[m]) for m in metrics]
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        bars = ax.bar(labels, values, color=["#4e79a7", "#f28e2b", "#59a14f", "#e15759"])
        _annotate_bars(ax, bars, fmt="{:.2f}")
        ax.set_title(f"JanusNet Video Result: Detection Changes ({row['run']})")
        ax.set_ylabel("Detections per frame")
        ax.grid(axis="y", alpha=0.25)
        _save(fig, "janusnet_visdrone_video_detection_metrics.png")

        metrics = ["psnr", "linf", "sec_per_frame"]
        labels = ["PSNR", "Linf", "sec/frame"]
        values = [_safe_float(row[m]) for m in metrics]
        fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.8))
        for ax, label, value in zip(axes, labels, values):
            bars = ax.bar([label], [value], color="#3b6ea8")
            _annotate_bars(ax, bars, fmt="{:.4f}" if label != "PSNR" else "{:.2f}")
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"JanusNet Video Result: Quality and Runtime ({row['run']})", y=1.04)
        _save(fig, "janusnet_visdrone_video_quality_runtime.png")


def plot_original_vs_janusnet_visdrone():
    path = ROOT / "outputs" / "image_eval" / "visdrone" / "val_4mode_metrics.csv"
    if not path.exists():
        return

    df = _read_csv(path)
    df["variant_norm"] = df["variant"].astype(str).str.lower()
    df = df[df["variant_norm"].isin(["original", "janusnet"])].copy()
    if set(df["variant_norm"]) != {"original", "janusnet"}:
        return

    order = ["original", "janusnet"]
    df["variant_norm"] = pd.Categorical(df["variant_norm"], categories=order, ordered=True)
    df = df.sort_values("variant_norm")

    for col in ["map50", "precision", "recall", "f1", "lpips", "ssim", "psnr", "avg_sec_per_image"]:
        if col in df.columns:
            df[col] = df[col].map(_safe_float)

    metrics = ["map50", "precision", "recall", "f1"]
    labels = ["mAP50", "Precision", "Recall", "F1"]
    original = df[df["variant"].str.lower() == "original"].iloc[0]
    janusnet = df[df["variant"].str.lower() == "janusnet"].iloc[0]

    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    bars_o = ax.bar(x - width / 2, [original[m] for m in metrics], width, label="Original", color="#4e79a7")
    bars_j = ax.bar(x + width / 2, [janusnet[m] for m in metrics], width, label="JanusNet", color="#e15759")
    _annotate_bars(ax, bars_o, fmt="{:.3f}")
    _annotate_bars(ax, bars_j, fmt="{:.3f}")
    ax.set_title("Original vs JanusNet on VisDrone")
    ax.set_ylabel("Detection score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(0.62, float(df[metrics].to_numpy().max()) * 1.2))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save(fig, "visdrone_original_vs_janusnet_detection_metrics.png")

    drops = []
    for metric in metrics:
        base = float(original[metric])
        adv = float(janusnet[metric])
        drops.append((base - adv) / (base + 1e-12) * 100.0)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bars = ax.bar(labels, drops, color="#d95f02")
    _annotate_bars(ax, bars, fmt="{:.1f}%")
    ax.set_title("JanusNet Performance Drop Relative to Original")
    ax.set_ylabel("Relative drop (%)")
    ax.set_ylim(0, max(100, max(drops) * 1.18))
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "visdrone_janusnet_relative_drop.png")

    summary = pd.DataFrame(
        {
            "metric": labels,
            "original": [original[m] for m in metrics],
            "janusnet": [janusnet[m] for m in metrics],
            "absolute_drop": [original[m] - janusnet[m] for m in metrics],
            "relative_drop_percent": drops,
        }
    )
    summary.to_csv(OUT_DIR / "visdrone_original_vs_janusnet_detection_metrics.csv", index=False)


def main():
    image_df = collect_image_janusnet_rows()
    video_df = collect_video_janusnet_rows()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not image_df.empty:
        image_df.to_csv(OUT_DIR / "janusnet_image_sources.csv", index=False)
    if not video_df.empty:
        video_df.to_csv(OUT_DIR / "janusnet_video_sources.csv", index=False)

    plot_image_metrics(image_df)
    plot_video_summary(video_df)
    plot_latest_frame_trend()
    plot_poster_ready_single_results(image_df, video_df)
    plot_original_vs_janusnet_visdrone()


if __name__ == "__main__":
    main()
