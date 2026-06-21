# BrokeCoders · TORCS IBM 2026


This is **BrokeCoders'** entry for the IBM AI Racing League. Instead of hand-coding racing logic or grinding reinforcement learning, we take the shortest path between a human and an autonomous lap: **record a person driving, then clone their reflexes into a small network.** The result steers, accelerates and brakes entirely on its own around the TORCS *Corkscrew* circuit.

---

## Idea

Driving is mostly muscle memory. So rather than describe how to take a corner, we just capture *a human taking it* — thousands of times — and let a network absorb the pattern. This is **behavioral cloning**: supervised learning where the input is the car's situation and the label is what the driver did about it.

Three moving parts:

| Stage | What happens | Entry point |
|-------|--------------|-------------|
| **Record** | You drive; every frame's situation + your pedals/wheel are saved. | `human_collect` |
| **Clone** | A network is fit to reproduce your inputs from the situation. | `bc` |
| **Drive** | The network takes the wheel; code only shifts gears. | `play` |

When the car develops a bad habit, a fourth stage closes the loop: you record short **corrections** at exactly the spots it fails, and they get extra weight in the next training pass.

---

## Under the hood

**What the network sees** — a 17-number snapshot built every frame (`get_state`):

- where the car points and sits on track (heading angle, track position)
- how fast it's going (forward + sideways speed, engine RPM)
- where it is around the lap (progress + its sine/cosine, so the loop has no seam)
- what's ahead — 9 forward distance sensors fanned across the road

**What the network decides** — three independent pedals/wheel outputs from a compact multilayer perceptron (`17 → 256 → 128 → 64`):

```
                 ┌─▶ steer   (tanh,    −1 … +1)
state(17) ─▶ MLP ─┼─▶ accel   (sigmoid,  0 … 1)
                 └─▶ brake   (sigmoid,  0 … 1)
```

Gears aren't learned — a simple speed lookup handles them, so the model spends all its capacity on the hard part: where to point the car and how hard to slow it.

---

## The braking problem

The first version had one fatal flaw: braking.

The cause is statistical. In a clean human lap, braking is **rare and spiky** — only about **6%** of frames brake at all, but those frames brake *hard* (≈0.66 on average). Train with vanilla mean-squared error and the optimizer discovers an easy win: predict "barely any brake" everywhere and eat the tiny penalty on those few frames. The brake output flatlines near zero, the car never sheds speed, and physics does the rest.

Our loss refuses that shortcut:

- **frames are weighted by how much you braked** (`1 + brake_weight × brake`) — so the handful of braking moments carry real gradient instead of drowning under a sea of throttle frames;
- **the brake channel is weighted above steer/accel**, since it's the one that was being neglected;
- **the saved checkpoint is chosen by a brake-aware score**, so a lazy "never-brake" model can't sneak through on a flattering average.

Both knobs are tunable:

```bash
python train.py bc --brake-weight=10 --brake-out-weight=3
```

After this change the network slams the brakes (up to full lock) into the first corner by itself and turns clean laps with no rule-based crutches.

---

## How to run it

**1. Python dependencies**

```bash
pip install -r requirements.txt    # torch, numpy, pygame
```

**2. TORCS + SCR server**

The agent talks to a live **TORCS** instance patched with the **SCR (Simulated Car Racing)** server, over a socket on **port 3001** (`snakeoil3_gym.py`). Launch TORCS and have it waiting for a client before you collect data or drive. On Linux, `autostart.sh` clicks through the menus for you via `xte`.

---

## Command reference

Everything is a subcommand of `train.py` (run it bare to print the menu):

```bash
python train.py human_collect --laps=5 --reset-each-lap   # drive & record
python train.py bc                                        # train -> bc_model.pth
python train.py offline_validate                          # grade vs. your laps, no sim
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
| `train.py` | The whole pipeline — state encoding, training, the live `play` loop, every assist layer. |
| `model.py` | The `Actor` network. |
| `human_drive.py` | Recording mode: pygame HUD + a physics-feel keyboard controller. |
| `racing_line_guardian.py` | The intervention state machine behind guardian/RLG modes. |
| `offline_validate.py` | Score a model against recorded laps without launching TORCS. |
| `snakeoil3_gym.py` | TORCS SCR socket client. |
| `combine_data.py` · `combine_corrections.py` · `remove_lap.py` | Dataset upkeep. |
| `practice.xml` · `autostart.sh` | Race configuration and (Linux) menu autostart. |

---

## Attribution & licensing

This project builds on standard TORCS / SCR Python tooling. Third-party code is documented here as required by the competition rules, and to make clear what is our own work.

**Third-party code (not written by BrokeCoders):**

| File(s) | Origin | Licence |
|---------|--------|---------|
| `snakeoil3_gym.py`, `jmcncarai.py` | *Snake Oil* SCR TORCS client — Chris X Edwards (`snakeoil@xed.ch`, scr.geccocompetitions.com) | as distributed with the SCR tooling (see file headers) |
| `gym_torcs.py`, `torcs_env.py`, `sample_agent.py` | *gym_torcs* — Naoto Yoshida | MIT — see `LICENSE` |
| `practice.xml`, `autostart.sh` | TORCS race configuration + Linux menu launcher | TORCS distribution (GPL) |

The repository `LICENSE` file is the MIT licence of the upstream *gym_torcs* project (© 2016 Naoto Yoshida), retained to satisfy its attribution requirement — not a licence grant over BrokeCoders' own code.

**Original work by BrokeCoders:** `train.py` (the full BC pipeline, brake-weighted loss, `play` loop and assist layers), `model.py` (the `Actor` network), `human_drive.py` (human data collection), `racing_line_guardian.py` (rule-based intervention), and the validation/dataset tooling (`offline_validate.py`, `combine_data.py`, `combine_corrections.py`, `remove_lap.py`, `collect_clean_laps.py`). The trained `bc_model.pth` and `human_racing_envelope.json` are derived from our own recorded laps.

Per the competition rules, only the AI Python code is modified — the TORCS car physics and the Corkscrew track definition are untouched.

---

<div align="center">

**BrokeCoders** — IBM AI Racing League

</div>
