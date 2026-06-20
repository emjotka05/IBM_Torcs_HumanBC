# Team Sqro — IBM AI Racing League

An AI racing driver for the TORCS **Corkscrew** track, trained by **Behavioral Cloning (BC)** from human driving demonstrations.

A human drives laps in TORCS; a small neural network learns to imitate them; the trained network then drives the car on its own. The AI controls **steering, throttle and brake** — gear changes are handled by code.

---

## How it works

The whole project is a human-in-the-loop imitation-learning pipeline:

1. **Collect** — you drive laps yourself in TORCS. Each frame, the car's state and your `steer / accel / brake` inputs are recorded as a training sample.
2. **Train (BC)** — a neural network is trained to map state → your action. No reinforcement learning, no reward — it just imitates you.
3. **Play** — the trained network drives the car. At inference the model outputs steering, throttle and brake every frame; the code only picks the gear.
4. **Correct (optional)** — where the model fails, you record short *correction* clips (e.g. braking late into a corner). These are oversampled during the next training round so the model learns to recover.

```
human_collect ──▶ driving_data.json ──▶  bc  ──▶ bc_model.pth ──▶ play
      ▲                                                              │
      └──────────────── correction_collect ◀───────────────────────┘
                         (fix what the model gets wrong)
```

### State, model, outputs

- **Input — 17 features** (`get_state` in `train.py`): heading angle, track position, longitudinal/lateral speed, RPM, lap progress (`lap_pos` + its sin/cos), and 9 forward track range-finder sensors. All normalized.
- **Model** (`model.py`) — an MLP `17 → 256 → 128 → 64` (ReLU) with three output heads:
  - `steer` → `tanh` (−1…1)
  - `accel` → `sigmoid` (0…1)
  - `brake` → `sigmoid` (0…1)
- **Gears** are not learned — a fixed `speed → gear` table sets them at play time.

### Brake-weighted training (the key trick)

Human braking is **rare but sharp** — only ~6% of frames have meaningful brake, yet when you do brake it averages ~0.66. Plain MSE averages that spike toward zero, so the brake head goes dead: the car under-brakes, arrives at corners far too fast, and flies off (the first hard corner was a guaranteed crash).

The loss fixes this:

- **per-sample weighting** — each frame's loss is scaled by `1 + brake_weight × brake`, so the rare braking frames actually drive the gradient instead of being averaged away;
- **per-output weighting** — the brake error term is weighted higher than steer/accel;
- **model selection** uses a brake-weighted score, so a "never-brake" checkpoint can't win on raw average MSE.

Tunable: `python train.py bc --brake-weight=10 --brake-out-weight=3`.

With this, the model brakes hard (up to 1.0) into the first corner on its own and completes clean Corkscrew laps under pure BC.

---

## Requirements & setup

- **Python 3** with the packages in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```
- **TORCS** patched with the **SCR (Simulated Car Racing) server**. The agent connects to a running TORCS instance over a socket on **port 3001** (via `snakeoil3_gym.py`). Start TORCS and have it waiting for a client before running `human_collect` or `play`. On Linux, `autostart.sh` drives the TORCS menus with `xte`.

---

## Usage

Everything runs through `train.py <command>` (run `python train.py` with no args for the full help):

```bash
# 1. Collect human demonstrations (opens a pygame window — you drive)
python train.py human_collect --laps=5 --reset-each-lap

# 2. Train the BC model  ->  bc_model.pth
python train.py bc

# 3. Check the model against your recorded actions, offline (no TORCS)
python train.py offline_validate

# 4. Let the trained model drive
python train.py play

# 5. Record targeted recoveries where it fails, then retrain
python train.py correction_collect
python combine_corrections.py
python train.py bc
```

**Driving controls** (during `human_collect` / `correction_collect`): `WASD` or arrow keys to drive · `F8` save segment + reset · `F9` save and quit · `Backspace` discard current pass.

### Play modes

`play` always uses `bc_model.pth` as the primary driver. Flags add optional rule-based safety/assist layers on top of the model's output:

| Flag | Effect |
|------|--------|
| *(none)* | Pure BC. Model drives; only gears + mild steering smoothing are code. |
| `--raw-steer` | Disable steering smoothing (raw model steer). |
| `--hybrid` | Sector-based launch + first-corner speed support. |
| `--racing-line-guardian` / `--rlg` | Rule-based track supervisor using the human envelope. |
| `--guardian-assist` | Late-intervention guardian state machine. |
| `--global-assist` | Whole-track stability assist. |
| `--corner-safety` | Speed/lane safety profiles. |
| `--guardian-collect` | (with a guardian mode) save interventions for retraining. |

> The assist layers only *correct* the model when it leaves the human driving envelope — the BC network is always the primary driver. With a well-trained brake-weighted model, pure `play` already laps cleanly.

---

## Data & training notes

- `driving_data.json` is the main dataset (`{state: [17 floats], action: [steer, accel, brake]}` per sample). It is **not committed** (see below).
- `human_collect` writes a new session to `driving_data_toCombine.json` when a main dataset already exists; merge it in with `python combine_data.py`. Corrections follow the same pattern via `combine_corrections.py`.
- Validation holds out whole lap-like groups (not random frames) so it doesn't leak adjacent frames.
- `remove_lap.py` deletes a bad lap from a dataset (it writes a timestamped backup first).
- Each `play` run is logged to `play_logs/play_<timestamp>.jsonl` (and `play_latest.jsonl`) for after-the-fact analysis.

### What's in this repository

The code, the trained model (`bc_model.pth`) and the precomputed `human_racing_envelope.json` are included. The large recorded datasets, run logs, offline-validation outputs and model backups are intentionally **excluded** via `.gitignore` (the raw datasets exceed GitHub's 100 MB file limit). To train, collect your own laps with `human_collect`.

---

## Project layout

| File | Role |
|------|------|
| `train.py` | Main pipeline: state building, BC training, the `play` loop, all assist/guardian layers. |
| `model.py` | The `Actor` network (and an unused `Critic`). |
| `human_drive.py` | Human data collection — pygame window, smooth physics-based car controller. |
| `racing_line_guardian.py` | Rule-based intervention state machine used by guardian/RLG modes. |
| `offline_validate.py` | Compare a trained model against recorded human actions without TORCS. |
| `snakeoil3_gym.py` | TORCS SCR socket client (port 3001). |
| `combine_data.py`, `combine_corrections.py`, `remove_lap.py` | Dataset maintenance helpers. |
| `practice.xml`, `autostart.sh` | TORCS race config and (Linux) menu autostart. |

> `gym_torcs.py`, `torcs_env.py`, `jmcncarai.py`, `sample_agent.py` are legacy scaffolding from the original gym_torcs / SCR base and are not part of the active pipeline.
