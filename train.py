"""
TORCS pipeline — record human demos, clone them, then let the net drive.

The policy network emits steering, throttle and brake every frame; only the
gear is chosen by hand-written code. Main entry points:

    python train.py human_collect   # drive yourself and record demos
    python train.py bc               # fit the imitation policy
    python train.py play             # hand the wheel to the trained policy
"""

import numpy as np
import os
import json
from datetime import datetime

import sys
COMMAND_ARGS = sys.argv[1:]
COMMAND = COMMAND_ARGS[0] if COMMAND_ARGS else None
COMMAND_FLAGS = set(COMMAND_ARGS[1:])
sys.argv = [sys.argv[0]]


def cli_int_flag(name, default):
    args = COMMAND_ARGS[1:]
    prefix = name + "="
    for i, arg in enumerate(args):
        if arg.startswith(prefix):
            try:
                return int(arg[len(prefix):])
            except ValueError:
                return default
        if arg == name and i + 1 < len(args):
            try:
                return int(args[i + 1])
            except ValueError:
                return default
    return default

import snakeoil3_gym as snakeoil3

PI = 3.14159265359

# -------------------------------------------------------------------------
# Constants & file paths
# -------------------------------------------------------------------------
FEATURE_DIM     = 17
LEGACY_FEATURE_DIM = 20
CONTROL_DIM    = 3       # steer, accel, brake
LAP_LENGTH_M = 3600
DATASET_PATH     = "driving_data.json"
CORRECTIONS_PATH = "correction_data.json"
CORRECTIONS_PENDING_PATH = "correction_data_toCombine.json"
CORRECTION_OVERSAMPLE = 5
RUN_LOG_DIR  = "play_logs"
SCR_PORT          = 3001


# Per-frame steering-change limit used to smooth model steering in play.
STEER_SLEW_LIMIT = 0.035


def build_observation(S, lap_start_dist=0.0):
    track = S.get('track', [100.0] * 19)

    dist_raced = float(S.get('distRaced', 0.0))
    if lap_start_dist is None:
        lap_start_dist = 0.0
    lap_distance = dist_raced - float(lap_start_dist)
    if not np.isfinite(lap_distance):
        lap_distance = 0.0
    # TORCS can report distFromStart near the lap end while the car is on the
    # grid. Episode-relative distRaced keeps lap_pos=0 at launch.
    if lap_distance < 0.0:
        lap_distance = max(0.0, dist_raced)

    lap_pos = (lap_distance % LAP_LENGTH_M) / LAP_LENGTH_M
    lap_angle = 2.0 * PI * lap_pos

    return np.array([
        float(S.get('angle', 0.0)) / PI,
        float(S.get('trackPos', 0.0)),
        float(S.get('speedX', 0.0)) / 300.0,
        float(S.get('speedY', 0.0)) / 300.0,
        float(S.get('rpm', 0.0)) / 10000.0,
        lap_pos,
        np.sin(lap_angle),
        np.cos(lap_angle),
        float(track[0]) / 200.0,
        float(track[3]) / 200.0,
        float(track[5]) / 200.0,
        float(track[7]) / 200.0,
        float(track[9]) / 200.0,
        float(track[11]) / 200.0,
        float(track[13]) / 200.0,
        float(track[16]) / 200.0,
        float(track[18]) / 200.0,
    ], dtype=np.float32)


def normalize_dataset(raw_data, source_name):
    """Coerce 17D (and legacy 20D) rows into the current 17-feature layout."""
    prepared = []
    group_ids = []
    legacy_count = 0
    group_id = 0
    group_start_lap_pos = None
    prev_lap_pos = None

    for idx, sample in enumerate(raw_data):
        state = [float(x) for x in sample.get('state', [])]
        action = [float(x) for x in sample.get('action', [])]
        state_dim = len(state)

        if state_dim not in (FEATURE_DIM, LEGACY_FEATURE_DIM):
            raise ValueError(
                f"{source_name}[{idx}] has state_dim={state_dim}, "
                f"expected {FEATURE_DIM} or legacy {LEGACY_FEATURE_DIM}."
            )
        if len(action) != CONTROL_DIM:
            raise ValueError(
                f"{source_name}[{idx}] has action_dim={len(action)}, "
                f"expected {CONTROL_DIM}."
            )

        lap_pos_raw = state[5] if len(state) > 5 else 0.0
        if prev_lap_pos is not None and lap_pos_raw + 0.10 < prev_lap_pos:
            group_id += 1
            group_start_lap_pos = None

        if state_dim == LEGACY_FEATURE_DIM:
            legacy_count += 1
            if group_start_lap_pos is None:
                group_start_lap_pos = lap_pos_raw
            lap_pos = (lap_pos_raw - group_start_lap_pos) % 1.0
            state = state[:FEATURE_DIM]
            state[5] = lap_pos
            state[6] = float(np.sin(2.0 * PI * lap_pos))
            state[7] = float(np.cos(2.0 * PI * lap_pos))
        elif group_start_lap_pos is None:
            group_start_lap_pos = 0.0

        prepared.append({
            'state': state,
            'action': action,
            'source': source_name,
        })
        group_ids.append(f"{source_name}:{group_id}")
        prev_lap_pos = lap_pos_raw

    return prepared, group_ids, legacy_count


def holdout_lap_groups(group_ids, min_lap_samples=500, val_fraction=0.20):
    """Reserve whole laps for validation so adjacent frames cannot leak."""
    group_order = []
    group_counts = {}
    for gid in group_ids:
        if gid not in group_counts:
            group_order.append(gid)
            group_counts[gid] = 0
        group_counts[gid] += 1

    eligible = [gid for gid in group_order if group_counts[gid] >= min_lap_samples]
    if not eligible and len(group_order) > 1:
        eligible = group_order
    if not eligible:
        return list(range(len(group_ids))), [], group_counts, []

    n_val = max(1, int(round(len(eligible) * val_fraction)))
    val_groups = set(eligible[-n_val:])
    train_idx = [i for i, gid in enumerate(group_ids) if gid not in val_groups]
    val_idx = [i for i, gid in enumerate(group_ids) if gid in val_groups]

    if not train_idx:
        split = max(1, int(len(group_ids) * (1.0 - val_fraction)))
        train_idx = list(range(split))
        val_idx = list(range(split, len(group_ids)))
        val_groups = set(group_ids[i] for i in val_idx)

    return train_idx, val_idx, group_counts, list(val_groups)


# -------------------------------------------------------------------------
# Gear selection (shared by play + human recording)
# -------------------------------------------------------------------------
def pick_gear(speed_x, R):
    gear = 1
    if speed_x > 40:  gear = 2
    if speed_x > 70:  gear = 3
    if speed_x > 100: gear = 4
    if speed_x > 135: gear = 5
    if speed_x > 170: gear = 6
    R['gear'] = gear


# -------------------------------------------------------------------------
# Imitation training (behavioral cloning)
# -------------------------------------------------------------------------
def fit_behavior_clone(epochs=500, batch_size=256):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from model import ClonePolicy

    print("\n" + "=" * 60)
    print("  Imitation training")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        print(f"  No data! Run: python train.py collect")
        return

    with open(DATASET_PATH, 'r') as f:
        base_data = json.load(f)
    if not base_data:
        print(f"  Empty data file: {DATASET_PATH}")
        return

    try:
        base_prepared, base_group_ids, base_legacy = normalize_dataset(
            base_data, DATASET_PATH)
    except ValueError as e:
        print(f"  Incompatible data: {e}")
        return

    train_idx, val_idx, group_counts, val_groups = holdout_lap_groups(
        base_group_ids)
    train_data = [base_prepared[i] for i in train_idx]
    val_data = [base_prepared[i] for i in val_idx]

    print(f"  Loaded {len(base_prepared)} base samples")
    print(f"  Current state dim: {FEATURE_DIM} (legacy {LEGACY_FEATURE_DIM}D samples are converted)")
    if base_legacy:
        print(f"  Converted legacy base samples: {base_legacy} (dropped prev_action, rebased lap_pos)")
    print(f"  Lap-like groups: {len(group_counts)} | validation groups: {len(val_groups)}")
    if val_groups:
        val_sizes = [group_counts[g] for g in val_groups]
        print(f"  Validation samples: {len(val_data)} | group sizes: {val_sizes}")
    else:
        print("  Validation disabled: not enough lap groups")

    if os.path.exists(CORRECTIONS_PATH):
        with open(CORRECTIONS_PATH, 'r') as f:
            correction_data = json.load(f)
        if correction_data:
            try:
                correction_prepared, _, correction_legacy = normalize_dataset(
                    correction_data, CORRECTIONS_PATH)
            except ValueError as e:
                print(f"  Incompatible corrections: {e}")
                return
            train_data.extend(correction_prepared * CORRECTION_OVERSAMPLE)
            print(f"  Loaded {len(correction_prepared)} correction samples")
            if correction_legacy:
                print(f"  Converted legacy correction samples: {correction_legacy}")
            print(f"  Correction weight: x{CORRECTION_OVERSAMPLE}")
            print(f"  Effective training samples: {len(train_data)}")
        else:
            print(f"  Correction file is empty: {CORRECTIONS_PATH}")
    else:
        print(f"  No correction file found: {CORRECTIONS_PATH}")

    if os.path.exists(CORRECTIONS_PENDING_PATH):
        print(f"  Pending corrections found: {CORRECTIONS_PENDING_PATH}")
        print(f"  Run: python combine_corrections.py before final correction training")

    print(f"  Action dim: {len(train_data[0]['action'])} (steer, accel, brake)")

    train_states  = torch.FloatTensor([d['state']  for d in train_data])
    train_actions = torch.FloatTensor([d['action'] for d in train_data])
    val_states = torch.FloatTensor([d['state'] for d in val_data]) if val_data else None
    val_actions = torch.FloatTensor([d['action'] for d in val_data]) if val_data else None

    # ── Loss weighting ────────────────────────────────────────────────────
    # Human braking is rare (~6% of frames) but sharp (~0.66 when present), so
    # plain per-frame MSE averages it toward zero and the brake head goes dead
    # (it predicts ~0 through corner-entry braking zones -> excess -> off).
    # Two corrections:
    #   1. per-sample weight rises with brake magnitude, so braking frames are
    #      not drowned by the mass of straight-line throttle frames;
    #   2. per-output weight emphasizes the chronically under-predicted brake
    #      head relative to steer/accel.
    brake_sample_weight = cli_int_flag("--brake-weight", 10)
    brake_output_weight = cli_int_flag("--brake-out-weight", 3)
    output_weights = torch.FloatTensor([1.0, 1.0, float(brake_output_weight)])

    train_weights = 1.0 + float(brake_sample_weight) * train_actions[:, 2]
    braking_mask = train_actions[:, 2] > 0.05
    n_braking = int(braking_mask.sum().item())
    n_total = len(train_actions)
    mean_w_brake = float(train_weights[braking_mask].mean()) if n_braking else 1.0
    mean_w_rest = float(train_weights[~braking_mask].mean()) if n_braking < n_total else 1.0
    print(f"  Loss weighting: per-sample w = 1 + {brake_sample_weight}*brake "
          f"| brake output x{brake_output_weight}")
    print(f"  Braking frames (brake>0.05): {n_braking}/{n_total} "
          f"({100.0 * n_braking / max(1, n_total):.1f}%) "
          f"| mean weight braking={mean_w_brake:.2f} non-braking={mean_w_rest:.2f}")

    actor     = ClonePolicy(FEATURE_DIM)
    optimizer = optim.Adam(actor.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)
    best_score = float('inf')

    def evaluate(states_t, actions_t):
        actor.eval()
        sum_sq = torch.zeros(CONTROL_DIM)
        count = 0
        with torch.no_grad():
            for i in range(0, len(states_t), 4096):
                batch_s = states_t[i:i + 4096]
                batch_a = actions_t[i:i + 4096]
                diff_sq = (actor(batch_s) - batch_a) ** 2
                sum_sq += diff_sq.sum(dim=0).cpu()
                count += diff_sq.shape[0]
        comp = sum_sq / max(1, count)
        return float(comp.mean().item()), [float(x) for x in comp.tolist()]

    def fmt_components(values):
        return f"steer={values[0]:.6f} accel={values[1]:.6f} brake={values[2]:.6f}"

    ow = [float(x) for x in output_weights.tolist()]

    def weighted_score(comp):
        # Model-selection metric: weight the brake component so we don't save a
        # checkpoint that minimizes average MSE by quietly never braking.
        return sum(c * w for c, w in zip(comp, ow)) / sum(ow)

    for epoch in range(epochs):
        actor.train()
        perm      = torch.randperm(len(train_states))
        states_s  = train_states[perm]
        actions_s = train_actions[perm]
        weights_s = train_weights[perm]
        total_loss = 0
        n_batches  = 0

        for i in range(0, len(states_s), batch_size):
            batch_s = states_s[i:i + batch_size]
            batch_a = actions_s[i:i + batch_size]
            batch_w = weights_s[i:i + batch_size]
            pred    = actor(batch_s)
            se          = (pred - batch_a) ** 2 * output_weights   # [B, 3] per-output weighted
            per_sample  = se.mean(dim=1)                           # [B]
            loss        = (per_sample * batch_w).sum() / batch_w.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        batch_loss = total_loss / n_batches

        train_loss, train_comp = evaluate(train_states, train_actions)
        train_score = weighted_score(train_comp)
        if val_states is not None:
            val_loss, val_comp = evaluate(val_states, val_actions)
            val_score = weighted_score(val_comp)
            score = val_score
        else:
            val_loss, val_comp = None, None
            score = train_score

        if epoch % 10 == 0 or score < best_score:
            marker = ""
            if score < best_score:
                best_score = score
                torch.save(actor.state_dict(), "bc_model.pth")
                marker = " <- BEST saved"
            msg = (
                f"  epoch {epoch:4d} | batch={batch_loss:.6f} "
                f"| train={train_loss:.6f} ({fmt_components(train_comp)})"
            )
            if val_loss is not None:
                msg += f" | val={val_loss:.6f} ({fmt_components(val_comp)})"
            msg += f" | sel_score={score:.6f}"
            print(msg + marker)

    score_name = "validation" if val_states is not None else "training"
    print(f"\n  BC complete! Best weighted {score_name} score={best_score:.6f} -> bc_model.pth")


# -------------------------------------------------------------------------
# Inference loop — the policy drives; code only shifts gears
# -------------------------------------------------------------------------
def run_policy():
    import torch
    from model import ClonePolicy

    print("\n" + "=" * 60)
    print("  Inference — the policy is driving")
    print("=" * 60)

    actor = ClonePolicy(FEATURE_DIM)
    if not os.path.exists("bc_model.pth"):
        print("  No trained policy found on disk.")
        return
    try:
        actor.load_state_dict(
            torch.load("bc_model.pth", map_location='cpu', weights_only=False))
    except RuntimeError as e:
        print("  Incompatible model file: bc_model.pth")
        print(f"  Current FEATURE_DIM={FEATURE_DIM}. Retrain with fresh data: python train.py bc")
        print(f"  Details: {e}")
        return
    model_file = "bc_model.pth"
    print(f"  Loaded: {model_file}")
    actor.eval()

    use_steer_smoothing = "--raw-steer" not in COMMAND_FLAGS

    C = snakeoil3.Client(p=SCR_PORT)
    C.MAX_STEPS = 50000
    C.get_servers_input()
    S = C.S.d

    dist_start = float(S.get('distRaced', 0))
    lap_start_dist = dist_start
    steps = 0
    prev_steer = 0.0
    prev_cmd = [0.0, 0.0, 0.0]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(RUN_LOG_DIR, exist_ok=True)
    play_log_path = os.path.join(RUN_LOG_DIR, f"play_{run_id}.jsonl")
    latest_log_path = os.path.join(RUN_LOG_DIR, "play_latest.jsonl")
    play_log_files = [
        open(play_log_path, "w", encoding="utf-8", buffering=1),
        open(latest_log_path, "w", encoding="utf-8", buffering=1),
    ]

    def write_play_log(event):
        event = dict(event)
        event.setdefault("time", datetime.now().isoformat(timespec="milliseconds"))
        line = json.dumps(event, ensure_ascii=False)
        for handle in play_log_files:
            handle.write(line + "\n")

    def close_play_logs():
        for handle in play_log_files:
            if not handle.closed:
                handle.close()

    import atexit
    atexit.register(close_play_logs)

    write_play_log({
        "type": "meta",
        "run_id": run_id,
        "command": " ".join(["train.py"] + COMMAND_ARGS),
        "state_dim": FEATURE_DIM,
        "model_file": model_file,
        "flags": sorted(COMMAND_FLAGS),
        "steer_smoothing": use_steer_smoothing,
        "base_steer_rate": STEER_SLEW_LIMIT,
        "lap_threshold": LAP_LENGTH_M,
    })

    print(f"  Steer smoothing: {'ON' if use_steer_smoothing else 'off'}")
    print(f"  Play log: {os.path.abspath(play_log_path)}")
    print("  Rolling out...")

    dist_ep = 0.0
    stop_reason = "normal"

    while True:
        prev_cmd_used = [float(x) for x in prev_cmd]
        state = build_observation(S, lap_start_dist)
        with torch.no_grad():
            a = actor(torch.FloatTensor(state).unsqueeze(0)).numpy()[0]

        speed_x = float(S.get('speedX', 0))
        speed_y = float(S.get('speedY', 0))
        track_pos = float(S.get('trackPos', 0))
        angle = float(S.get('angle', 0))
        rpm = float(S.get('rpm', 0))
        dist_raced = float(S.get('distRaced', 0))
        dist_from_start = float(S.get('distFromStart', 0))
        cur_lap_time = float(S.get('curLapTime', 0))
        dist_ep_before = dist_raced - dist_start
        raw_track = S.get('track', [])
        if not isinstance(raw_track, (list, tuple)):
            raw_track = []
        track = [float(x) for x in raw_track]
        abs_track_pos = abs(track_pos)
        off_track = abs_track_pos > 1.0
        near_edge = 0.85 <= abs_track_pos <= 1.0
        if track_pos > 0.05:
            track_side = "right"
        elif track_pos < -0.05:
            track_side = "left"
        else:
            track_side = "center"

        model_steer = float(np.clip(a[0], -1, 1))
        raw_steer = model_steer
        accel_cmd = float(np.clip(a[1], 0, 1))
        brake_cmd = float(np.clip(a[2], 0, 1))

        steer_cmd = raw_steer
        if use_steer_smoothing:
            steer_delta = np.clip(raw_steer - prev_steer, -STEER_SLEW_LIMIT, STEER_SLEW_LIMIT)
            steer_cmd = float(np.clip(prev_steer + steer_delta, -1, 1))
        prev_steer = steer_cmd
        prev_cmd = [steer_cmd, accel_cmd, brake_cmd]

        R = C.R.d
        R['steer'] = steer_cmd
        R['accel'] = accel_cmd
        R['brake'] = brake_cmd

        # Gears only — AI controls everything else
        gear = 1
        if speed_x > 40:  gear = 2
        if speed_x > 70:  gear = 3
        if speed_x > 100: gear = 4
        if speed_x > 135: gear = 5
        if speed_x > 170: gear = 6
        R['gear'] = gear

        write_play_log({
            "type": "step",
            "step": steps + 1,
            "dist_ep": dist_ep_before,
            "dist_raced": dist_raced,
            "dist_from_start": dist_from_start,
            "cur_lap_time": cur_lap_time,
            "speed_x": speed_x,
            "speed_y": speed_y,
            "track_pos": track_pos,
            "off_track": off_track,
            "near_edge": near_edge,
            "track_side": track_side,
            "angle": angle,
            "rpm": rpm,
            "track": track,
            "state": state.tolist(),
            "prev_cmd": prev_cmd_used,
            "model_steer": model_steer,
            "raw_steer": raw_steer,
            "cmd_steer": steer_cmd,
            "accel": accel_cmd,
            "brake": brake_cmd,
            "gear": gear,
        })

        C.respond_to_server()
        C.get_servers_input()
        S = C.S.d

        dist_ep = float(S.get('distRaced', 0)) - dist_start
        steps += 1

        if steps <= 300 and (steps <= 30 or steps % 25 == 0):
            print(f"  dbg {steps:3d} | dist={dist_ep_before:6.1f}m | "
                  f"v={speed_x:6.1f} | tpos={track_pos:+.3f} | "
                  f"ang={angle:+.3f} | model_steer={model_steer:+.3f} | "
                  f"cmd_steer={steer_cmd:+.3f} | acc={accel_cmd:.2f} | brk={brake_cmd:.2f}")

        if steps % 100 == 0:
            print(f"    {steps:5d} | {dist_ep:6.0f}m | "
                  f"{float(S.get('speedX', 0)):5.1f}km/h | "
                  f"lap={float(S.get('curLapTime', 0)):.1f}s")

        if dist_ep > LAP_LENGTH_M * 2:
            print(f"\n  Reached target distance: {dist_ep:.0f}m")
            break
        if steps > 30000:
            break

    write_play_log({
        "type": "stop",
        "reason": stop_reason,
        "steps": steps,
        "distance": dist_ep,
    })
    C.shutdown()
    close_play_logs()
    print(f"  Play log saved: {os.path.abspath(play_log_path)}")
    print(f"  Latest log alias: {os.path.abspath(latest_log_path)}")


# -------------------------------------------------------------------------
# Command-line entry
# -------------------------------------------------------------------------
if __name__ == "__main__":
    if COMMAND == "human_collect":
        from human_drive import human_collect_data
        human_collect_data(
            num_laps=cli_int_flag("--laps", 50),
            auto_reset_each_lap=("--reset-each-lap" in COMMAND_FLAGS),
        )
    elif COMMAND == "correction_collect":
        from human_drive import human_collect_data
        correction_output = (
            CORRECTIONS_PENDING_PATH if os.path.exists(CORRECTIONS_PATH)
            else CORRECTIONS_PATH
        )
        human_collect_data(
            output_file=correction_output,
            restart_on_save=True,
            allow_low_progress=("--allow-start" in COMMAND_FLAGS),
        )
    elif COMMAND == "bc":
        fit_behavior_clone(epochs=500)
    elif COMMAND == "offline_validate":
        from offline_validate import main as offline_validate_main
        offline_validate_main(COMMAND_ARGS[1:])
    elif COMMAND == "play":
        try:
            run_policy()
        except KeyboardInterrupt:
            print("\n  Play interrupted by user.")
            print(f"  Latest log alias: {os.path.abspath(os.path.join(RUN_LOG_DIR, 'play_latest.jsonl'))}")
    else:
        print("BrokeCoders — TORCS imitation pipeline")
        print("=" * 45)
        print("  python train.py human_collect   # Phase 1: collect data manually (Play the game!)")
        print("  python train.py human_collect --reset-each-lap # Phase 1: reset car after every completed lap")
        print("  python train.py human_collect --laps=10 --reset-each-lap # Phase 1: collect 10 clean reset laps")
        print("  python train.py correction_collect # Phase 4: collect targeted correction samples")
        print("  python train.py correction_collect --allow-start # Corrections that include 0-15 km/h start")
        print("  python train.py bc              # Phase 2: Behavioral Cloning")
        print("  python train.py offline_validate # Offline model-vs-human validation")
        print("  python train.py play            # Run trained model")
        print("  python train.py play --raw-steer # Run trained model without steering smoothing")
        print("  python train.py bc --brake-weight=10 --brake-out-weight=3 # Tune brake loss weighting")
        print("  Note: current model expects 17-dim states; legacy 20-dim data is converted during training.")
        print()
        print("Flow: human_collect -> bc -> play -> correction_collect -> combine_corrections.py -> bc")
