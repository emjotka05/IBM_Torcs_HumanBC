# BrokeCoders · TORCS IBM 2026

My entry for the **IBM AI Racing League** — an autonomous TORCS driver I built by *behavioral cloning*. 

---

## Idea

I decided that humans are still — for now — better at this than AI, so instead of explaining cornering theory to the network, I took matters into my own hands. First I learned to drive the track properly myself; then I recorded many of my own laps and used them as training data.

The network receives the car's situation as input and my driving actions as the expected output — steering, throttle, and braking. This is **behavioral cloning**: a supervised-learning approach where the model doesn't learn to drive from scratch, but instead learns to imitate a human who already knows what they're doing.

Three moving parts:

| Stage | What happens | Entry point |
|-------|--------------|-------------|
| **Record** | I drive; every frame's situation plus my steering / throttle / brake are saved. | `human_collect` |
| **Clone** | A network is trained to reproduce my inputs from that situation. | `bc` |
| **Drive** | The network takes the wheel; my code only shifts gears. | `play` |

When the car develops a bad habit, a fourth stage closes the loop: I record short **corrections** at exactly the spots it fails, and they get extra weight in the next training pass.

---

## Under the hood

**What the network sees** — a 17-number snapshot I build every frame (`build_observation`):

- where the car points and sits on track (heading angle, track position)
- how fast it's going (forward + sideways speed, engine RPM)
- where it is around the lap (progress + its sine/cosine, so the loop has no seam)
- what's ahead — 9 forward distance sensors fanned across the road

**What the network decides** — three independent pedal/wheel outputs from a compact multilayer perceptron (`17 → 256 → 128 → 64`):

```
                 ┌─▶ steer   (tanh,    −1 … +1)
state(17) ─▶ MLP ─┼─▶ accel   (sigmoid,  0 … 1)
                 └─▶ brake   (sigmoid,  0 … 1)
```

I don't learn gears — a simple speed lookup handles them, so the model spends all its capacity on the hard part: where to point the car and how hard to slow it.

---

## The braking problem

The first version had one fatal flaw: braking.

The cause is statistical. In a clean human lap, braking is **rare and spiky** — only about **6%** of frames brake at all, but those frames brake *hard* (≈0.66 on average). Train with vanilla mean-squared error and the optimizer discovers an easy win: predict "barely any brake" everywhere and eat the tiny penalty on those few frames. The brake output flatlines near zero, the car never sheds speed, and physics does the rest.

My loss refuses that shortcut:

- **frames are weighted by how hard I braked** (`1 + brake_weight × brake`) — so the handful of braking moments carry real gradient instead of drowning under a sea of throttle frames;
- **the brake channel is weighted above steer/accel**, since it's the one that was being neglected;
- **the saved checkpoint is chosen by a brake-aware score**, so a lazy "never-brake" model can't sneak through on a flattering average.

Both knobs are tunable:

```bash
python train.py bc --brake-weight=10 --brake-out-weight=3
```

After this change the network slams the brakes (up to full lock) into the first corner by itself and turns clean laps with no rule-based crutches.

### Where IBM Granite came in

I handed my real braking telemetry to a locally-run **IBM Granite** model — the model's flat-zero brake through the corner versus the human's hard braking, plus the dataset's action statistics — and asked it to diagnose and fix the training, with no hint at a solution (`granite_engineer.py` / `ask_granite_brake.py`; output saved under `granite_analysis/`).

Granite independently found the same root cause (equal-weight MSE ignores the rare braking frames) and recommended re-weighting the loss toward braking. I evaluated its advice rather than copying it: I adopted its top two ideas — a heavier weight on the brake output and per-frame weighting of braking samples — in a cleaner form (its proposed formula carried an indicator term that cancels out), and deliberately dropped its third suggestion (input normalization + extra features), since my observations are already normalized and I wanted to keep the architecture fixed. That weighing of the AI's own advice is the brake-weighted loss described above.

---

## How to run it

**1. Python dependencies**

```bash
pip install -r requirements.txt    # torch, numpy, pygame
```

**2. TORCS + SCR server**

The agent talks to a live **TORCS** instance patched with the **SCR (Simulated Car Racing)** server, over a socket on **port 3001** (`snakeoil3_gym.py`). Launch TORCS and have it waiting for a client before collecting data or driving. On Linux, `autostart.sh` clicks through the menus automatically via `xte`.

---

## Command reference

Everything is a subcommand of `train.py` (run it bare to print the menu):

```bash
python train.py human_collect --laps=5 --reset-each-lap   # drive & record
python train.py bc                                        # train -> bc_model.pth
python train.py offline_validate                          # grade against recorded laps, no sim
python train.py play                                      # let the model drive
python train.py correction_collect                        # record targeted recoveries
```

After a correction session, fold the clips into the dataset and retrain:

```bash
python combine_corrections.py
python train.py bc
```

**Keys while recording:** `WASD` / arrows to drive · `F8` save + reset · `F9` save + quit · `Backspace` scrap the current attempt.

## Repo map

| Path | What it does |
|------|--------------|
| `train.py` | The whole pipeline — state encoding, the brake-weighted training loss, and the live `play` loop. |
| `model.py` | The `ClonePolicy` network. |
| `human_drive.py` | Recording mode: pygame HUD + a physics-feel keyboard controller. |
| `granite_engineer.py` · `ask_granite_brake.py` | Feed telemetry to IBM Granite for analysis (output in `granite_analysis/`). |
| `offline_validate.py` | Score a model against recorded laps without launching TORCS. |
| `snakeoil3_gym.py` | TORCS SCR socket client. |
| `combine_data.py` · `combine_corrections.py` · `remove_lap.py` | Dataset upkeep. |
| `practice.xml` · `autostart.sh` | Race configuration and (Linux) menu autostart. |

---

## Attribution & licensing

This project builds on standard TORCS / SCR Python tooling. Third-party code is documented here as required by the competition rules, and to make clear what is my own work.

**Third-party code (not written by me):**

| File(s) | Origin | License |
|---------|--------|---------|
| `snakeoil3_gym.py`, `jmcncarai.py` | *Snake Oil* SCR TORCS client — Chris X Edwards (`snakeoil@xed.ch`, scr.geccocompetitions.com) | as distributed with the SCR tooling (see file headers) |
| `gym_torcs.py`, `torcs_env.py`, `sample_agent.py` | *gym_torcs* — Naoto Yoshida | MIT — see `LICENSE` |
| `practice.xml`, `autostart.sh` | TORCS race configuration + Linux menu launcher | TORCS distribution (GPL) |

The repository `LICENSE` file is the MIT license of the upstream *gym_torcs* project (© 2016 Naoto Yoshida), retained to satisfy its attribution requirement — not a license grant over my own code.

**Original work, written by me:** `train.py` (the full BC pipeline, the brake-weighted loss, and the `play` loop), `model.py` (the `ClonePolicy` network), `human_drive.py` (human data collection), the IBM Granite analysis tooling (`granite_engineer.py`, `ask_granite_brake.py`), and the validation/dataset tooling (`offline_validate.py`, `combine_data.py`, `combine_corrections.py`, `remove_lap.py`, `collect_clean_laps.py`). The trained `bc_model.pth` is derived from my own recorded laps.

Per the competition rules, only the AI Python code is modified — the TORCS car physics and the Corkscrew track definition are untouched.

---

<div align="center">

**BrokeCoders** — IBM AI Racing League

</div>
