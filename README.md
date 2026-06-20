# Team Sqro - IBM AI Racing League

AI racing driver for the TORCS Corkscrew track using Behavioral Cloning.

## How it works
- `human_collect` - collect human driving demonstrations.
- `correction_collect` - collect targeted recovery/correction samples.
- `bc` - train the neural network via Behavioral Cloning.
- `play` - run the trained model; code handles gears only.
- `offline_validate.py` - compare a trained model with recorded human actions.

## Architecture
- Input: 17 features (angle, track position, speed, lap progress, 9 track sensors).
- No autoregressive `prev_action` features in the model input.
- Hidden layers: 256 -> 128 -> 64 neurons (ReLU).
- Output: steer (tanh), accel (sigmoid), brake (sigmoid).
- Training: Adam optimizer, lap-group validation, per-output MSE logs.

## Recommended workflow
- `python train.py human_collect --laps=5 --reset-each-lap`
- `python train.py bc`
- `python offline_validate.py`
- `python train.py play`
- `python train.py correction_collect`
- `python combine_corrections.py`
- `python train.py bc`

Correction data should include trackPos 0.5-0.9, edge recoveries, slower corner entry, and mild line mistakes with safe recovery.
