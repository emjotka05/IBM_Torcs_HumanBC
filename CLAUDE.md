# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Behavioral Cloning (BC) AI driver for the TORCS racing simulator, built for the IBM AI Racing League. A human plays the game to record demonstrations; a small neural net is trained to imitate them; the trained net then drives the car. The AI controls `steer`, `accel`, and `brake`; gear shifting is handled by code, not the model.

There is no build system, package manifest, or test suite. It is a set of scripts run directly with `python`. Runtime dependencies (not pinned anywhere): `torch`, `numpy`, `pygame`, plus a running TORCS instance patched with the SCR (Simulated Car Racing) server. The AI process connects to TORCS over a socket via `snakeoil3_gym.py` on `SCR_PORT = 3001`.

## Commands

Everything funnels through `train.py <command> [flags]` (the bare `python train.py` prints the full help/order):

```bash
python train.py human_collect --laps=5 --reset-each-lap   # record human driving (opens a pygame window)
python train.py correction_collect [--allow-start]        # record targeted recovery/correction samples
python train.py bc [--brake-weight=N]                     # train the BC model -> bc_model.pth
python train.py offline_validate                          # compare model vs recorded human actions (no TORCS)
python train.py play [mode flags]                         # run the trained model in TORCS
python train.py collect                                   # legacy: collect data from the built-in aggressive bot
```

Helper scripts (run standalone):
- `python combine_data.py` — merge `driving_data_toCombine.json` into the main `driving_data.json`.
- `python combine_corrections.py` — merge `correction_data_toCombine.json` into `correction_data.json`.
- `python offline_validate.py [--data ... --model ... --include-corrections]` — same as `train.py offline_validate`, with more flags.
- `python remove_lap.py` — interactively delete a lap segment from a dataset (writes a timestamped `*_backup_before_remove_lap_*.json` first).
- `python collect_clean_laps.py` — thin wrapper around `human_collect_data` with `auto_reset_each_lap=True`.

There is no "run a single test" — validation is `offline_validate` (offline, metric-based) and `play` (online, in TORCS).

### `play` mode flags

`run_policy()` loads `bc_model.pth` and drives directly: the model's steer/accel/brake go straight to TORCS, with only gear selection and a steering rate-limiter added by code. One flag:
- `--raw-steer` — disable the rate-limiter (send the model's steering unsmoothed).

(Earlier rule-based assist layers — hybrid / racing-line-guardian / guardian-assist / global-assist / corner-safety / launch helpers — were removed once the brake-weighted model drove cleanly on its own.)

## Architecture

### The model (`model.py`)
`ClonePolicy` is an MLP: `FEATURE_DIM (17) -> 256 -> 128 -> 64` (ReLU), then three separate output heads — `steer` (tanh), `accel` (sigmoid), `brake` (sigmoid). `ValueHead` exists but is unused (BC only, no RL).

### The state vector (`train.py:build_observation`)
17 features, in order: `angle/PI`, `trackPos`, `speedX/300`, `speedY/300`, `rpm/10000`, `lap_pos`, `sin(lap_angle)`, `cos(lap_angle)`, then 9 normalized track range-finder sensors at indices `[0,3,5,7,9,11,13,16,18]/200`. `lap_pos` is episode-relative distance modulo `LAP_LENGTH_M = 3600` (track-specific). There are **no autoregressive `prev_action` features** in the live model — earlier 20-dim datasets included them.

### `train.py` is the spine
~875 lines: CLI parsing, state construction, the scripted data-collection bot, BC training, and the play loop. A few things to know before editing:
- **Custom arg parsing at module top.** `COMMAND`, `COMMAND_FLAGS`, and `COMMAND_ARGS` are captured from `sys.argv`, then `sys.argv` is reset to `[sys.argv[0]]` so imported libraries don't see the flags. Use `cli_int_flag(...)` / `COMMAND_FLAGS` rather than reintroducing `argparse` at the top level.
- **Legacy data conversion.** `normalize_dataset` accepts both 17-dim and legacy 20-dim samples; 20-dim ones are downconverted (drop `prev_action`, rebase `lap_pos`). Keep this path working — most existing datasets predate the 17-dim format.
- **Lap-group validation.** `holdout_lap_groups` holds out whole lap-like groups (not random frames) so validation isn't leaking adjacent frames. `fit_behavior_clone` reports per-output MSE and saves the best checkpoint to `bc_model.pth`.
- **Data weighting.** Correction samples are oversampled `×CORRECTION_OVERSAMPLE (5)` and concatenated into the training set.
- **Brake-weighted loss.** Human braking is rare (~6% of frames) but sharp, so plain MSE drives the brake head to ~0 (under-braking → overspeed → off-track). The loss is a weighted MSE: per-sample weight `1 + (--brake-weight, default 10)*brake`, plus a per-output weight on brake (`--brake-out-weight`, default 3). Model selection uses a brake-weighted `sel_score` (not raw MSE) so a "never-brake" checkpoint can't win. `evaluate()` still reports honest unweighted per-output MSE.

### IBM Granite tooling
`granite_engineer.py` / `ask_granite_brake.py` send real braking telemetry to a locally-run IBM Granite model and save its analysis under `granite_analysis/`. Used to document the brake-weighted-loss decision; not part of the driving loop.

### Data collection (`human_drive.py`)
Opens a pygame window with a physics-based smooth controller (steering spring-damper, progressive throttle ramp, traction-circle stability control). Controls: **WASD** (or arrows) to drive, **F8** = save segment + reset (correction mode), **F9** = save and quit, **BACKSPACE** = discard current pass and reset. On Windows it reads keys globally via `GetAsyncKeyState` so the TORCS window can stay focused. Note: the UI strings and many comments here are in **Polish**.

## Data flow and files

- `driving_data.json` — main training set (`{state: [17 floats], action: [steer, accel, brake]}` per sample). This is what `bc` trains on.
- `human_collect` writes to `driving_data.json` directly only if it doesn't exist yet; otherwise it writes the new session to `driving_data_toCombine.json`, which you then merge with `combine_data.py`. Same two-file pattern for corrections (`correction_data.json` + `correction_data_toCombine.json` + `combine_corrections.py`).
- `bc_model.pth` — the active model. **`play` and `offline_validate` load this hardcoded filename**, so to test a different checkpoint, copy it over `bc_model.pth` (the other `bc_model_*.pth` files and `model_backups/` are experiments/backups).
- `play_logs/play_<timestamp>.jsonl` and `play_latest.jsonl` — per-step inference logs from each `play` run.
- `offline_validation/` — summaries and CSVs from `offline_validate`.

### Recommended iteration loop
`human_collect` → `combine_data.py` → `bc` → `offline_validate` → `play` → `correction_collect` → `combine_corrections.py` → `bc` → `play`. Good correction data covers `trackPos` 0.5–0.9, edge recoveries, slower corner entry, and mild line mistakes with safe recovery.

## TORCS connection notes

`snakeoil3_gym.py` is the active TORCS client (a Python3 port of the SCR `snakeoil` library); `train.py` and `human_drive.py` both use it on port 3001. A TORCS instance with the SCR server must already be running and waiting for a client. `autostart.sh` drives the TORCS menus via `xte` (Linux/X11 only) — it is a Linux convenience, while the human-driving key handling targets Windows.

`gym_torcs.py`, `torcs_env.py`, `jmcncarai.py`, and `sample_agent.py` are legacy scaffolding from the original gym_torcs / SCR base and are **not** imported by the active `train.py` / `human_drive.py` pipeline.
