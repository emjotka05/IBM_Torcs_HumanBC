"""
Offline validation for TORCS behavioral cloning models.

This script compares a trained model against recorded human actions without
running TORCS. It is meant to separate two failure modes:
  1. the model does not copy the dataset well offline,
  2. the model copies offline but fails online because its own errors move it
     outside the dataset distribution.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "driving_data.json"
DEFAULT_MODEL = ROOT / "bc_model.pth"
DEFAULT_CORRECTIONS = ROOT / "correction_data.json"
DEFAULT_OUT_DIR = ROOT / "offline_validation"
ACTIONS = ("steer", "accel", "brake")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Validate a TORCS BC model against recorded driving data."
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA),
                        help="Dataset JSON file. Default: driving_data.json")
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help="Model file. Default: bc_model.pth")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="Output directory for summaries and CSV logs.")
    parser.add_argument("--corrections", default=str(DEFAULT_CORRECTIONS),
                        help="Correction dataset used only with --include-corrections.")
    parser.add_argument("--include-corrections", action="store_true",
                        help="Also validate on correction_data.json if present.")
    parser.add_argument("--sector-size", type=float, default=100.0,
                        help="Sector size in meters for per-track analysis.")
    parser.add_argument("--val-fraction", type=float, default=0.20,
                        help="Fraction of lap-like groups held out for validation.")
    parser.add_argument("--min-lap-samples", type=int, default=500,
                        help="Minimum group size to be treated as a lap-like group.")
    parser.add_argument("--top-sectors", type=int, default=12,
                        help="How many worst sectors to print.")
    parser.add_argument("--plot", action="store_true",
                        help="Try to generate PNG plots with matplotlib.")
    return parser.parse_args(argv)


def import_project_modules():
    # train.py rewrites sys.argv at import time, so protect this script's args.
    saved_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        import train  # pylint: disable=import-error,import-outside-toplevel
        from model import Actor  # pylint: disable=import-error,import-outside-toplevel
    finally:
        sys.argv = saved_argv
    return train, Actor


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_torch_model(model_path, Actor):
    import torch  # pylint: disable=import-outside-toplevel

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")

    state_dict = checkpoint.get("actor", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("Model file does not contain a PyTorch state_dict.")

    if "fc.0.weight" not in state_dict:
        # Accept simple DataParallel-style prefixes if they ever appear.
        stripped = {}
        for key, value in state_dict.items():
            stripped[key.replace("module.", "", 1)] = value
        state_dict = stripped

    if "fc.0.weight" not in state_dict:
        raise ValueError("Cannot infer model input dimension: missing fc.0.weight.")

    input_dim = int(state_dict["fc.0.weight"].shape[1])
    actor = Actor(input_dim)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor, input_dim


def prepare_current_dim(raw_data, source_name, train):
    prepared, group_ids, legacy_count = train.prepare_training_samples(raw_data, source_name)
    rows = []
    for idx, row in enumerate(prepared):
        rows.append({
            "index": idx,
            "source": row.get("source", source_name),
            "state": [float(x) for x in row["state"]],
            "action": [float(x) for x in row["action"]],
            "group_id": group_ids[idx],
        })
    return rows, legacy_count, 0


def prepare_legacy_dim(raw_data, source_name, train):
    rows = []
    skipped = 0
    group_id = 0
    prev_lap_pos = None

    for idx, sample in enumerate(raw_data):
        state = [float(x) for x in sample.get("state", [])]
        action = [float(x) for x in sample.get("action", [])]
        if len(state) != train.LEGACY_STATE_DIM or len(action) != train.ACTION_DIM:
            skipped += 1
            continue

        lap_pos = state[5] if len(state) > 5 else 0.0
        if prev_lap_pos is not None and lap_pos + 0.10 < prev_lap_pos:
            group_id += 1

        rows.append({
            "index": idx,
            "source": source_name,
            "state": state,
            "action": action,
            "group_id": f"{source_name}:{group_id}",
        })
        prev_lap_pos = lap_pos

    return rows, len(rows), skipped


def prepare_rows_for_model(raw_data, source_name, model_input_dim, train):
    if model_input_dim == train.STATE_DIM:
        rows, legacy_count, skipped = prepare_current_dim(raw_data, source_name, train)
        return rows, legacy_count, skipped
    if model_input_dim == train.LEGACY_STATE_DIM:
        return prepare_legacy_dim(raw_data, source_name, train)
    raise ValueError(
        f"Unsupported model input dim {model_input_dim}. "
        f"Expected {train.STATE_DIM} or legacy {train.LEGACY_STATE_DIM}."
    )


def split_rows(rows, train, min_lap_samples, val_fraction):
    group_ids = [row["group_id"] for row in rows]
    train_idx, val_idx, group_counts, val_groups = train.split_by_lap_groups(
        group_ids,
        min_lap_samples=min_lap_samples,
        val_fraction=val_fraction,
    )
    for idx in train_idx:
        rows[idx]["split"] = "train"
    for idx in val_idx:
        rows[idx]["split"] = "val"
    if not val_idx:
        for row in rows:
            row.setdefault("split", "train")
    return train_idx, val_idx, group_counts, val_groups


def predict_rows(rows, actor, batch_size=4096):
    import torch  # pylint: disable=import-outside-toplevel

    states = torch.FloatTensor([row["state"] for row in rows])
    preds = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            batch = states[start:start + batch_size]
            preds.extend(actor(batch).cpu().tolist())
    for row, pred in zip(rows, preds):
        row["pred"] = [float(x) for x in pred]
        row["err"] = [row["pred"][i] - row["action"][i] for i in range(3)]


def mean(values):
    return sum(values) / max(1, len(values))


def metrics_for(rows):
    if not rows:
        return None
    mse = []
    mae = []
    bias = []
    for i in range(3):
        errors = [row["err"][i] for row in rows]
        mse.append(mean([e * e for e in errors]))
        mae.append(mean([abs(e) for e in errors]))
        bias.append(mean(errors))
    return {
        "n": len(rows),
        "mse": mse,
        "mae": mae,
        "bias": bias,
        "total_mse": mean(mse),
        "total_mae": mean(mae),
    }


def state_meta(row, lap_threshold):
    state = row["state"]
    lap_pos = state[5] if len(state) > 5 else 0.0
    lap_distance = (lap_pos % 1.0) * lap_threshold
    return {
        "lap_pos": lap_pos,
        "lap_distance": lap_distance,
        "track_pos": state[1] if len(state) > 1 else 0.0,
        "speed_x": (state[2] * 300.0) if len(state) > 2 else 0.0,
        "speed_y": (state[3] * 300.0) if len(state) > 3 else 0.0,
        "angle": (state[0] * math.pi) if len(state) > 0 else 0.0,
        "front_track": (state[12] * 200.0) if len(state) > 12 else 0.0,
    }


def dataset_health(rows, lap_threshold):
    nonfinite = 0
    bad_action_range = 0
    off_track = 0
    near_edge = 0
    gas_brake = 0
    brake = 0
    hard_steer = 0
    for row in rows:
        for value in row["state"] + row["action"]:
            if not math.isfinite(float(value)):
                nonfinite += 1
        action = row["action"]
        if not (-1.0001 <= action[0] <= 1.0001 and
                -0.0001 <= action[1] <= 1.0001 and
                -0.0001 <= action[2] <= 1.0001):
            bad_action_range += 1
        meta = state_meta(row, lap_threshold)
        if abs(meta["track_pos"]) > 1.0:
            off_track += 1
        if 0.85 <= abs(meta["track_pos"]) <= 1.0:
            near_edge += 1
        if action[1] > 0.1 and action[2] > 0.1:
            gas_brake += 1
        if action[2] > 0.05:
            brake += 1
        if abs(action[0]) > 0.5:
            hard_steer += 1
    n = max(1, len(rows))
    return {
        "nonfinite_values": nonfinite,
        "bad_action_range_rows": bad_action_range,
        "off_track_rows": off_track,
        "near_edge_rows": near_edge,
        "gas_brake_rows": gas_brake,
        "brake_rows": brake,
        "hard_steer_rows": hard_steer,
        "off_track_pct": 100.0 * off_track / n,
        "near_edge_pct": 100.0 * near_edge / n,
        "brake_pct": 100.0 * brake / n,
        "hard_steer_pct": 100.0 * hard_steer / n,
    }


def sector_metrics(rows, lap_threshold, sector_size):
    sectors = {}
    for row in rows:
        meta = state_meta(row, lap_threshold)
        sector_idx = int(meta["lap_distance"] // sector_size)
        sector_start = sector_idx * sector_size
        sector_end = sector_start + sector_size
        key = (sector_start, sector_end, row.get("split", "all"))
        sectors.setdefault(key, []).append(row)

    results = []
    for (start, end, split), sector_rows in sectors.items():
        m = metrics_for(sector_rows)
        metas = [state_meta(row, lap_threshold) for row in sector_rows]
        results.append({
            "split": split,
            "sector_start_m": start,
            "sector_end_m": end,
            "n": len(sector_rows),
            "mse_steer": m["mse"][0],
            "mse_accel": m["mse"][1],
            "mse_brake": m["mse"][2],
            "mae_steer": m["mae"][0],
            "mae_accel": m["mae"][1],
            "mae_brake": m["mae"][2],
            "total_mse": m["total_mse"],
            "mean_speed_x": mean([x["speed_x"] for x in metas]),
            "mean_abs_track_pos": mean([abs(x["track_pos"]) for x in metas]),
            "mean_front_track": mean([x["front_track"] for x in metas]),
            "brake_rows": sum(1 for row in sector_rows if row["action"][2] > 0.05),
            "hard_steer_rows": sum(1 for row in sector_rows if abs(row["action"][0]) > 0.5),
        })
    return sorted(results, key=lambda r: (r["split"], r["sector_start_m"]))


def write_predictions_csv(path, rows, lap_threshold):
    fields = [
        "index", "source", "group_id", "split", "lap_distance_m",
        "track_pos", "speed_x", "speed_y", "angle", "front_track",
        "human_steer", "pred_steer", "err_steer",
        "human_accel", "pred_accel", "err_accel",
        "human_brake", "pred_brake", "err_brake",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            meta = state_meta(row, lap_threshold)
            writer.writerow({
                "index": row["index"],
                "source": row["source"],
                "group_id": row["group_id"],
                "split": row.get("split", "train"),
                "lap_distance_m": f"{meta['lap_distance']:.3f}",
                "track_pos": f"{meta['track_pos']:.6f}",
                "speed_x": f"{meta['speed_x']:.6f}",
                "speed_y": f"{meta['speed_y']:.6f}",
                "angle": f"{meta['angle']:.6f}",
                "front_track": f"{meta['front_track']:.6f}",
                "human_steer": f"{row['action'][0]:.8f}",
                "pred_steer": f"{row['pred'][0]:.8f}",
                "err_steer": f"{row['err'][0]:.8f}",
                "human_accel": f"{row['action'][1]:.8f}",
                "pred_accel": f"{row['pred'][1]:.8f}",
                "err_accel": f"{row['err'][1]:.8f}",
                "human_brake": f"{row['action'][2]:.8f}",
                "pred_brake": f"{row['pred'][2]:.8f}",
                "err_brake": f"{row['err'][2]:.8f}",
            })


def write_sector_csv(path, sectors):
    fields = [
        "split", "sector_start_m", "sector_end_m", "n",
        "mse_steer", "mse_accel", "mse_brake",
        "mae_steer", "mae_accel", "mae_brake",
        "total_mse", "mean_speed_x", "mean_abs_track_pos",
        "mean_front_track", "brake_rows", "hard_steer_rows",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sectors:
            writer.writerow(row)


def copy_latest(timestamped_path, latest_path):
    latest_path.write_bytes(timestamped_path.read_bytes())


def format_action_values(values):
    return " ".join(f"{name}={values[i]:.6f}" for i, name in enumerate(ACTIONS))


def maybe_make_plots(rows, sectors, out_dir, stamp):
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # pragma: no cover - optional dependency
        return [f"Plots skipped: matplotlib unavailable ({exc})"]

    messages = []
    val_rows = [row for row in rows if row.get("split") == "val"] or rows
    max_points = 2500
    if len(val_rows) > max_points:
        step = max(1, len(val_rows) // max_points)
        plot_rows = val_rows[::step]
    else:
        plot_rows = val_rows

    xs = list(range(len(plot_rows)))
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for i, name in enumerate(ACTIONS):
        axes[i].plot(xs, [row["action"][i] for row in plot_rows],
                     label=f"human {name}", linewidth=1.0)
        axes[i].plot(xs, [row["pred"][i] for row in plot_rows],
                     label=f"pred {name}", linewidth=1.0, alpha=0.8)
        axes[i].set_ylabel(name)
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.25)
    axes[-1].set_xlabel("validation sample order")
    fig.tight_layout()
    plot_path = out_dir / f"offline_{stamp}_actions.png"
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    copy_latest(plot_path, out_dir / "offline_latest_actions.png")
    messages.append(f"Action plot: {plot_path}")

    val_sectors = [s for s in sectors if s["split"] == "val"] or sectors
    val_sectors = sorted(val_sectors, key=lambda r: r["sector_start_m"])
    fig, ax = plt.subplots(figsize=(14, 5))
    labels = [int(s["sector_start_m"]) for s in val_sectors]
    ax.plot(labels, [s["mse_steer"] for s in val_sectors], label="steer mse")
    ax.plot(labels, [s["mse_accel"] for s in val_sectors], label="accel mse")
    ax.plot(labels, [s["mse_brake"] for s in val_sectors], label="brake mse")
    ax.set_xlabel("sector start [m]")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    sector_plot_path = out_dir / f"offline_{stamp}_sector_errors.png"
    fig.savefig(sector_plot_path, dpi=130)
    plt.close(fig)
    copy_latest(sector_plot_path, out_dir / "offline_latest_sector_errors.png")
    messages.append(f"Sector plot: {sector_plot_path}")
    return messages


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    train, Actor = import_project_modules()

    data_path = Path(args.data)
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    actor, model_input_dim = load_torch_model(model_path, Actor)
    raw_data = load_json(data_path)
    rows, legacy_count, skipped = prepare_rows_for_model(
        raw_data, data_path.name, model_input_dim, train)

    if args.include_corrections:
        corr_path = Path(args.corrections)
        if corr_path.exists():
            corr_raw = load_json(corr_path)
            corr_rows, corr_legacy, corr_skipped = prepare_rows_for_model(
                corr_raw, corr_path.name, model_input_dim, train)
            rows.extend(corr_rows)
            legacy_count += corr_legacy
            skipped += corr_skipped
        else:
            print(f"Correction file not found, skipped: {corr_path}")

    if not rows:
        raise ValueError("No compatible samples found for this model input dimension.")

    train_idx, val_idx, group_counts, val_groups = split_rows(
        rows, train, args.min_lap_samples, args.val_fraction)
    predict_rows(rows, actor)

    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    all_metrics = metrics_for(rows)
    train_metrics = metrics_for(train_rows)
    val_metrics = metrics_for(val_rows) if val_rows else None
    health = dataset_health(rows, train.LAP_THRESHOLD)
    sectors = sector_metrics(rows, train.LAP_THRESHOLD, args.sector_size)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = out_dir / f"offline_{stamp}_predictions.csv"
    sector_path = out_dir / f"offline_{stamp}_sector_metrics.csv"
    summary_path = out_dir / f"offline_{stamp}_summary.txt"
    write_predictions_csv(pred_path, rows, train.LAP_THRESHOLD)
    write_sector_csv(sector_path, sectors)
    copy_latest(pred_path, out_dir / "offline_latest_predictions.csv")
    copy_latest(sector_path, out_dir / "offline_latest_sector_metrics.csv")

    split_for_top = "val" if val_rows else "train"
    top_candidates = [s for s in sectors if s["split"] == split_for_top and s["n"] >= 10]
    top_total = sorted(top_candidates, key=lambda r: r["total_mse"], reverse=True)[:args.top_sectors]
    top_steer = sorted(top_candidates, key=lambda r: r["mse_steer"], reverse=True)[:args.top_sectors]
    top_brake = sorted(top_candidates, key=lambda r: r["mse_brake"], reverse=True)[:args.top_sectors]

    lines = []
    lines.append("Offline BC Validation")
    lines.append("=" * 60)
    lines.append(f"Dataset: {data_path}")
    lines.append(f"Model:   {model_path}")
    lines.append(f"Rows: {len(rows)} | skipped incompatible: {skipped}")
    lines.append(f"Model input dim: {model_input_dim}")
    lines.append(f"Current code STATE_DIM: {train.STATE_DIM} | legacy: {train.LEGACY_STATE_DIM}")
    if model_input_dim != train.STATE_DIM:
        lines.append(
            "WARNING: model input dim differs from current play/train STATE_DIM. "
            "This model is useful for offline diagnosis, but current train.py play "
            "expects a freshly trained current-dim model."
        )
    if legacy_count:
        lines.append(f"Legacy-compatible samples used/converted: {legacy_count}")
    lines.append(f"Lap-like groups: {len(group_counts)} | validation groups: {len(val_groups)}")
    if val_groups:
        lines.append("Validation group sizes: " +
                     ", ".join(str(group_counts[g]) for g in val_groups))
    else:
        lines.append("Validation disabled: not enough lap-like groups.")
    lines.append("")
    lines.append("Dataset health:")
    for key, value in health.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("Metrics:")
    lines.append(f"  all   n={all_metrics['n']} mse({format_action_values(all_metrics['mse'])}) "
                 f"mae({format_action_values(all_metrics['mae'])}) "
                 f"bias({format_action_values(all_metrics['bias'])})")
    lines.append(f"  train n={train_metrics['n']} mse({format_action_values(train_metrics['mse'])}) "
                 f"mae({format_action_values(train_metrics['mae'])}) "
                 f"bias({format_action_values(train_metrics['bias'])})")
    if val_metrics:
        lines.append(f"  val   n={val_metrics['n']} mse({format_action_values(val_metrics['mse'])}) "
                     f"mae({format_action_values(val_metrics['mae'])}) "
                     f"bias({format_action_values(val_metrics['bias'])})")
    lines.append("")

    def add_top(title, items, metric_name):
        lines.append(title)
        if not items:
            lines.append("  no sectors")
            return
        for item in items:
            lines.append(
                f"  {item['sector_start_m']:6.0f}-{item['sector_end_m']:6.0f}m "
                f"n={item['n']:5d} {metric_name}={item[metric_name]:.6f} "
                f"mse_s/a/b={item['mse_steer']:.6f}/"
                f"{item['mse_accel']:.6f}/{item['mse_brake']:.6f} "
                f"speed={item['mean_speed_x']:.1f} "
                f"|tpos|={item['mean_abs_track_pos']:.3f} "
                f"front={item['mean_front_track']:.1f}"
            )

    add_top(f"Worst {split_for_top} sectors by total_mse:", top_total, "total_mse")
    lines.append("")
    add_top(f"Worst {split_for_top} sectors by mse_steer:", top_steer, "mse_steer")
    lines.append("")
    add_top(f"Worst {split_for_top} sectors by mse_brake:", top_brake, "mse_brake")
    lines.append("")
    lines.append(f"Predictions CSV: {pred_path}")
    lines.append(f"Sector CSV:      {sector_path}")

    if args.plot:
        lines.extend(maybe_make_plots(rows, sectors, out_dir, stamp))

    summary = "\n".join(lines) + "\n"
    summary_path.write_text(summary, encoding="utf-8")
    (out_dir / "offline_latest_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
