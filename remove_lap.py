import argparse
import json
import os
import shutil
from datetime import datetime


def load_samples(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} nie zawiera listy probek JSON.")
    return data


def lap_pos(sample):
    state = sample.get("state", [])
    if len(state) < 6:
        raise ValueError("Probka ma za krotki state; nie moge odczytac lap_pos.")
    return float(state[5])


def split_segments(samples):
    if not samples:
        return []

    segments = []
    start = 0
    prev = lap_pos(samples[0])
    for i in range(1, len(samples)):
        cur = lap_pos(samples[i])
        if cur < prev - 0.45:
            segments.append((start, i - 1))
            start = i
        prev = cur
    segments.append((start, len(samples) - 1))
    return segments


def describe_segment(samples, start, end):
    rows = end - start + 1
    lps = [lap_pos(samples[i]) for i in range(start, end + 1)]
    track_pos = [float(samples[i]["state"][1]) for i in range(start, end + 1)]
    off_track = sum(1 for x in track_pos if abs(x) > 1.0)
    near_edge = sum(1 for x in track_pos if abs(x) > 0.85)
    return {
        "rows": rows,
        "start_lap_pos": min(lps),
        "end_lap_pos": max(lps),
        "last_lap_pos": lps[-1],
        "off_track": off_track,
        "near_edge": near_edge,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Usuwa wybrane okrazenie/segment z driving_data.json z automatycznym backupem."
    )
    parser.add_argument("--file", default="driving_data.json", help="Plik z probkami JSON.")
    parser.add_argument("--drop-lap", type=int, required=True, help="Numer segmentu/okrazenia do usuniecia, liczony od 1.")
    parser.add_argument(
        "--drop-tail",
        action="store_true",
        help="Usun tez koncowy niepelny segment po ostatniej mecie, jesli istnieje.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tylko pokaz co zostaloby usuniete, bez zapisu pliku.",
    )
    args = parser.parse_args()

    path = args.file
    samples = load_samples(path)
    segments = split_segments(samples)

    if args.drop_lap < 1 or args.drop_lap > len(segments):
        raise SystemExit(f"Nie ma segmentu {args.drop_lap}. Wykryto segmentow: {len(segments)}.")

    drop_indexes = {args.drop_lap - 1}
    if args.drop_tail and len(segments) > 1:
        tail_idx = len(segments) - 1
        tail = describe_segment(samples, *segments[tail_idx])
        if tail["end_lap_pos"] < 0.95:
            drop_indexes.add(tail_idx)

    print(f"Plik: {path}")
    print(f"Probki przed: {len(samples)}")
    print(f"Wykryte segmenty: {len(segments)}")
    for idx, (start, end) in enumerate(segments, 1):
        desc = describe_segment(samples, start, end)
        marker = " <- USUN" if (idx - 1) in drop_indexes else ""
        print(
            f"  {idx:02d}: rows={desc['rows']:5d} "
            f"lap_pos={desc['start_lap_pos']:.3f}->{desc['end_lap_pos']:.3f} "
            f"last={desc['last_lap_pos']:.3f} "
            f"off={desc['off_track']} near={desc['near_edge']}{marker}"
        )

    keep = []
    removed = 0
    for idx, (start, end) in enumerate(segments):
        if idx in drop_indexes:
            removed += end - start + 1
            continue
        keep.extend(samples[start : end + 1])

    print(f"Probki do usuniecia: {removed}")
    print(f"Probki po: {len(keep)}")

    if args.dry_run:
        print("DRY RUN: nic nie zapisano.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root, ext = os.path.splitext(path)
    backup = f"{root}_backup_before_remove_lap_{stamp}{ext}"
    shutil.copy2(path, backup)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(keep, f)

    print(f"Backup: {backup}")
    print(f"Zapisano oczyszczony plik: {path}")


if __name__ == "__main__":
    main()
