"""
ask_granite_brake.py — run the braking-problem analysis on a locally
downloaded IBM Granite model and save Granite's reply as an artefact.

It mirrors the loading pattern in C:\\granite41\\run_granite.py and uses the
exact, NEUTRAL evidence prompt produced by granite_engineer.py from our real
data (dataset action stats + the pre-fix turn-1 telemetry where the model's
brake output was flat 0.00). The prompt does not suggest a solution, so any
recommendation to weight/oversample the braking frames is Granite's own.

Run it with the venv that already has transformers + torch, from the project
folder so the artefact lands in granite_analysis/:

    cd C:\\Users\\emjot\\Downloads\\IBM\\TORCS_Human_BC
    C:\\granite41\\venv\\Scripts\\python.exe ask_granite_brake.py
"""

import os
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("GRANITE_HF_MODEL", "ibm-granite/granite-4.1-3b")
OUT_DIR = "granite_analysis"

SYSTEM = ("You are an experienced machine-learning engineer reviewing an "
          "imitation-learning (behavioral cloning) setup. Reason from the "
          "evidence; do not assume a fix in advance. Be specific and technical.")

USER = """We trained a neural network to drive a TORCS race car by behavioral cloning from ~50 human laps. It maps a 17-feature state to three outputs: steer, accel, brake. Training is supervised regression.

ORIGINAL TRAINING SETUP (the version that produced the problem):
- Loss: plain mean-squared error, applied EQUALLY to every frame and to every output (steer/accel/brake weighted the same; every frame weighted the same).
- Optimizer: Adam, lr 1e-3.

DATASET STATISTICS (from 246684 frames):
- steer: mean 0.048, std 0.253
- accel: mean 0.628, std 0.398
- brake: mean 0.042, std 0.183
- frames with brake > 0.05: 6.4%   (brake > 0.2: 5.4%)
- mean brake value on the frames that DO brake: 0.661

SYMPTOM AT THE FIRST HARD CORNER (per 20 m of track):
Human driver (the training target):
  dist(m) | speed(km/h) | brake | accel
   320    |     221     | 0.00  | 0.83
   340    |     220     | 0.18  | 0.31
   360    |     194     | 0.69  | 0.00
   380    |     147     | 0.66  | 0.00
   400    |     115     | 0.12  | 0.05
Trained model in the simulator (same zone):
  dist(m) | speed(km/h) | brake
   320    |     227     | 0.00
   340    |     231     | 0.00
   360    |     234     | 0.00
   380    |     236     | 0.00
   400    |     229     | 0.00
  -> the car then leaves the track around 466 m (146/736 frames off track).

QUESTIONS:
1. Given the data and the loss, what is the most likely root cause that the model barely brakes at the corner?
2. What concrete changes to the loss function or training procedure would fix it? Be specific (formulas / pseudocode welcome) and list them in priority order.
3. Any risks or tuning advice for your top recommendation."""


def main():
    print(f"Loading {MODEL_PATH} on CPU (first run can take a while)...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="cpu")
    model.eval()

    chat = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER}]
    prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    print("Generating (greedy, up to 1500 tokens)...")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1500, do_sample=False)
    reply = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_DIR, f"brake_analysis_{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Granite braking analysis — {stamp}\n\n")
        f.write(f"_Model: `{MODEL_PATH}` (run locally via Hugging Face transformers)_\n\n")
        f.write(f"## Prompt sent to Granite\n\n**System:** {SYSTEM}\n\n```\n{USER}\n```\n\n")
        f.write(f"## Granite response\n\n{reply}\n")

    print("\n=== Granite said ===\n")
    print(reply)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
