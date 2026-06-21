"""
granite_engineer.py — let IBM Granite diagnose the braking problem.

This tool builds an *evidence package* from our real data:
  * the action distribution in the dataset (how rare/strong braking is), and
  * the first-corner telemetry: what the human did vs. what the trained model
    actually did (from play_logs),
and asks an IBM Granite model to (1) diagnose why the model barely brakes and
(2) recommend concrete fixes to the training procedure.

The prompt is deliberately NEUTRAL: it describes the *original* training setup
(plain MSE, every frame weighted equally) and the symptoms, and does NOT
suggest a solution — so any recommendation to weight/oversample the braking
frames is Granite's own conclusion.

Backends (auto-detected in this order):
  1. Hugging Face — a Granite model downloaded from Hugging Face, run locally.
                   Set GRANITE_HF_MODEL to the local model folder or the HF id
                   (e.g. ibm-granite/granite-3.3-8b-instruct).
                   Needs: pip install transformers accelerate
  2. watsonx.ai  — set WATSONX_API_KEY and WATSONX_PROJECT_ID
                   (optional: WATSONX_URL, WATSONX_MODEL_ID)
  3. Ollama      — local; have `ollama serve` running with a Granite model
                   pulled, e.g. `ollama pull granite3.3`
                   (optional: OLLAMA_MODEL, OLLAMA_URL)
  4. manual      — no backend reachable: the prompt is saved to a file so you
                   can paste it into the watsonx Prompt Lab / SkillsBuild
                   Granite chat, then keep the reply.

Output: granite_analysis/brake_analysis_<timestamp>.md
Usage:  python granite_engineer.py
"""

import json
import os
import sys
import glob
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime

import numpy as np

DATA_FILE = "driving_data.json"
PLAY_LOG_DIR = "play_logs"
OUT_DIR = "granite_analysis"
LAP_THRESHOLD = 3600.0
FIRST_CORNER = (300.0, 470.0)        # turn-1 braking zone (episode metres)
BINS = list(range(320, 420, 20))     # representative 20 m bins for the prompt


# ──────────────────────────────────────────────────────────────────────────
#  Evidence: dataset action distribution + first-corner human profile
# ──────────────────────────────────────────────────────────────────────────
def dataset_evidence():
    if not os.path.exists(DATA_FILE):
        return None
    data = json.load(open(DATA_FILE))
    A = np.array([s["action"] for s in data], dtype=np.float32)
    S = np.array([s["state"] for s in data], dtype=np.float32)
    steer, accel, brake = A[:, 0], A[:, 1], A[:, 2]
    dist = S[:, 5] * LAP_THRESHOLD           # lap_pos -> metres
    braking = brake > 0.05

    human_bins = []
    for lo in BINS:
        m = (dist >= lo) & (dist < lo + 20)
        if m.sum() > 10:
            human_bins.append((lo, float(np.mean(S[m, 2] * 300.0)),
                               float(brake[m].mean()), float(accel[m].mean())))
    return {
        "n": len(data),
        "steer": (float(steer.mean()), float(steer.std())),
        "accel": (float(accel.mean()), float(accel.std())),
        "brake": (float(brake.mean()), float(brake.std())),
        "brake_frac_005": float(braking.mean()),
        "brake_frac_02": float((brake > 0.2).mean()),
        "mean_brake_when_braking": float(brake[braking].mean()) if braking.any() else 0.0,
        "human_bins": human_bins,
    }


# ──────────────────────────────────────────────────────────────────────────
#  Evidence: what the trained model actually did at turn 1 (pure-BC play log)
# ──────────────────────────────────────────────────────────────────────────
def model_evidence(explicit=None):
    # `explicit` lets you point at a specific run (e.g. the pre-fix log that
    # actually shows the braking failure). Otherwise pick the newest pure-BC run.
    if explicit:
        chosen = explicit if os.path.exists(explicit) else os.path.join(PLAY_LOG_DIR, explicit)
        if not os.path.exists(chosen):
            print(f"  (play log not found: {explicit})")
            return None
    else:
        logs = sorted(glob.glob(os.path.join(PLAY_LOG_DIR, "play_2*.jsonl")))
        chosen = None
        for path in reversed(logs):                  # newest first
            with open(path, encoding="utf-8") as f:
                meta = json.loads(f.readline())
            if meta.get("flags") == []:              # pure BC, no assists
                chosen = path
                break
        if chosen is None:
            return None

    steps = [json.loads(l) for l in open(chosen, encoding="utf-8") if l.strip()]
    steps = [r for r in steps if r.get("type") == "step"]

    model_bins = []
    for lo in BINS:
        m = [r for r in steps if lo <= r["dist_ep"] < lo + 20]
        if m:
            model_bins.append((lo, float(np.mean([r["speed_x"] for r in m])),
                               float(np.mean([r["brake"] for r in m]))))
    off = [r["dist_ep"] for r in steps if r.get("off_track")]
    return {
        "log": os.path.basename(chosen),
        "model_bins": model_bins,
        "off_track_frames": len(off),
        "total_frames": len(steps),
        "first_off_track_dist": min(off) if off else None,
    }


# ──────────────────────────────────────────────────────────────────────────
#  Build the (neutral) prompt
# ──────────────────────────────────────────────────────────────────────────
def build_prompt(d, m):
    lines = []
    lines.append(
        "We trained a neural network to drive a TORCS race car by behavioral "
        "cloning from ~50 human laps. It maps a 17-feature state to three "
        "outputs: steer, accel, brake. Training is supervised regression.\n")
    lines.append("ORIGINAL TRAINING SETUP (the version that produced the problem):")
    lines.append("- Loss: plain mean-squared error, applied EQUALLY to every "
                 "frame and to every output (steer/accel/brake weighted the "
                 "same; every frame weighted the same).")
    lines.append("- Optimizer: Adam, lr 1e-3.\n")

    if d:
        lines.append(f"DATASET STATISTICS (from {d['n']} frames):")
        lines.append(f"- steer: mean {d['steer'][0]:.3f}, std {d['steer'][1]:.3f}")
        lines.append(f"- accel: mean {d['accel'][0]:.3f}, std {d['accel'][1]:.3f}")
        lines.append(f"- brake: mean {d['brake'][0]:.3f}, std {d['brake'][1]:.3f}")
        lines.append(f"- frames with brake > 0.05: {100*d['brake_frac_005']:.1f}%   "
                     f"(brake > 0.2: {100*d['brake_frac_02']:.1f}%)")
        lines.append(f"- mean brake value on the frames that DO brake: "
                     f"{d['mean_brake_when_braking']:.3f}\n")

    lines.append("SYMPTOM AT THE FIRST HARD CORNER (per 20 m of track):")
    if d and d["human_bins"]:
        lines.append("Human driver (the training target):")
        lines.append("  dist(m) | speed(km/h) | brake | accel")
        for lo, sp, br, ac in d["human_bins"]:
            lines.append(f"  {lo:4d}    | {sp:7.0f}     | {br:.2f}  | {ac:.2f}")
    if m and m["model_bins"]:
        lines.append("Trained model in the simulator (same zone):")
        lines.append("  dist(m) | speed(km/h) | brake")
        for lo, sp, br in m["model_bins"]:
            lines.append(f"  {lo:4d}    | {sp:7.0f}     | {br:.2f}")
        if m["first_off_track_dist"] is not None:
            lines.append(f"  -> the car then leaves the track around "
                         f"{m['first_off_track_dist']:.0f} m "
                         f"({m['off_track_frames']}/{m['total_frames']} frames off track).")
    lines.append("")
    lines.append("QUESTIONS:")
    lines.append("1. Given the data and the loss, what is the most likely root "
                 "cause that the model barely brakes at the corner?")
    lines.append("2. What concrete changes to the loss function or training "
                 "procedure would fix it? Be specific (formulas / pseudocode "
                 "welcome) and list them in priority order.")
    lines.append("3. Any risks or tuning advice for your top recommendation.")

    system = ("You are an experienced machine-learning engineer reviewing an "
              "imitation-learning (behavioral cloning) setup. Reason from the "
              "evidence; do not assume a fix in advance. Be specific and technical.")
    return system, "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
#  Backends
# ──────────────────────────────────────────────────────────────────────────
def _http_json(url, payload, headers, timeout=120):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def call_watsonx(system, user):
    api_key = os.environ.get("WATSONX_API_KEY")
    project_id = os.environ.get("WATSONX_PROJECT_ID")
    if not (api_key and project_id):
        return None
    base = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    model_id = os.environ.get("WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct")

    # 1) exchange the API key for an IAM bearer token
    token_req = urllib.request.Request(
        "https://iam.cloud.ibm.com/identity/token",
        data=urllib.parse.urlencode({
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": api_key,
        }).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
    with urllib.request.urlopen(token_req, timeout=60) as r:
        token = json.loads(r.read().decode("utf-8"))["access_token"]

    # 2) generate
    out = _http_json(
        f"{base}/ml/v1/text/generation?version=2023-05-29",
        {
            "model_id": model_id,
            "project_id": project_id,
            "input": f"{system}\n\n{user}",
            "parameters": {"decoding_method": "greedy", "max_new_tokens": 900},
        },
        {"Authorization": f"Bearer {token}"})
    return out["results"][0]["generated_text"], model_id


def call_hf(system, user):
    """Run a locally downloaded Granite model via Hugging Face transformers."""
    model_ref = os.environ.get("GRANITE_HF_MODEL") or os.environ.get("GRANITE_MODEL_PATH")
    if not model_ref:
        return None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  (HF backend needs: pip install transformers accelerate)")
        return None
    print(f"  Loading Granite from {model_ref} (first run can be slow)...")
    tok = AutoTokenizer.from_pretrained(model_ref)
    model = AutoModelForCausalLM.from_pretrained(
        model_ref, torch_dtype="auto", device_map="auto")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(inputs, max_new_tokens=900, do_sample=False)
    text = tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
    return text.strip(), str(model_ref)


def call_ollama(system, user):
    base = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "granite3.3")
    try:
        out = _http_json(
            f"{base}/api/generate",
            {"model": model, "system": system, "prompt": user, "stream": False},
            {}, timeout=300)
    except (urllib.error.URLError, OSError):
        return None
    return out.get("response", ""), model


# ──────────────────────────────────────────────────────────────────────────
def main():
    play_log = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GRANITE_PLAY_LOG")
    d = dataset_evidence()
    m = model_evidence(play_log)
    if not (d or m):
        print("No data found. Run from the project root (needs driving_data.json "
              "and/or play_logs/).")
        return
    system, user = build_prompt(d, m)

    backend = os.environ.get("GRANITE_BACKEND")  # 'hf'|'watsonx'|'ollama'|None=auto
    has_hf = bool(os.environ.get("GRANITE_HF_MODEL") or os.environ.get("GRANITE_MODEL_PATH"))
    response, model_id = None, None
    if backend == "hf" or (backend is None and has_hf):
        res = call_hf(system, user)
        if res and res[0].strip():
            response, model_id = res
    if response is None and backend in (None, "watsonx"):
        res = call_watsonx(system, user)
        if res:
            response, model_id = res
    if response is None and backend in (None, "ollama"):
        res = call_ollama(system, user)
        if res and res[0].strip():
            response, model_id = res

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_DIR, f"brake_analysis_{stamp}.md")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Granite braking analysis — {stamp}\n\n")
        if m:
            f.write(f"_Model telemetry from: `{m['log']}`_\n\n")
        f.write("## Prompt sent to Granite\n\n")
        f.write(f"**System:** {system}\n\n")
        f.write("```\n" + user + "\n```\n\n")
        if response:
            f.write(f"## Granite response (`{model_id}`)\n\n{response}\n")
        else:
            f.write("## Granite response\n\n"
                    "_No backend was reachable._ Paste the prompt above into the "
                    "watsonx Prompt Lab / SkillsBuild Granite chat, then append "
                    "the reply here.\n")

    print(f"Wrote {out_path}")
    if response:
        print(f"\n=== Granite ({model_id}) said ===\n")
        print(response)
    else:
        print("\nNo Granite backend reachable — saved the prompt for manual use.")
        print("For a Hugging Face model downloaded locally, set GRANITE_HF_MODEL "
              "to its folder/id and re-run (needs: pip install transformers accelerate).")


if __name__ == "__main__":
    main()
