# Step 1: environment, the official LongCat-Video code (pinned) and the model downloads
import os
import sys
import time
import json
import shutil
import subprocess
import threading
import psutil
from concurrent.futures import ThreadPoolExecutor


def stop_old_engine():
    # A previous run's engine (or one left behind by a kernel restart) still holds both GPUs and ~20 GB of RAM.
    stale = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            args, name = p.info["cmdline"] or [], (p.info["name"] or "").lower()
            engine = "python" in name and any(os.path.basename(a) == "run_longcat.py" for a in args[1:3])
            if p.pid != os.getpid() and (engine or name == "cloudflared" or name.startswith("frpc")):
                p.kill()
                stale.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    psutil.wait_procs(stale, timeout=20)
    if stale:
        print(f"🧹 Stopped {len(stale)} leftover engine process(es): GPU memory and RAM are free again")


stop_old_engine()

# ---- Settings (every source is locked to one version) ----
LONGCAT_COMMIT = "58eb336edc5c8c63389838727e40b8c18434e54e"   # meituan-longcat/LongCat-Video, 15 Nov 2026 (Avatar 1.6 code)
INT8_REPO, INT8_REVISION = "TeamAIQ/LongCat-Video-Avatar-1.6-INT8-ConvRot", "a0da39246116c23d828f26dc06bc841d43a882de"
AVATAR_REPO, AVATAR_REVISION = "meituan-longcat/LongCat-Video-Avatar-1.6", "acfc628244201fc53fb6fe8a8903c5bac9bdf68f"
BASE_REPO, BASE_REVISION = "meituan-longcat/LongCat-Video", "03b55529b1d1d4045f5fbe14d65c8c6e8116b278"
T5_REPO, T5_REVISION = "DeepBeepMeep/Wan2.1", "176bbae94db7b26f00e5b698d95d4f578debc51c"  # UMT5-XXL in BF16
MODELS_DIR = "/kaggle/tmp/longcat_models" if os.path.isdir("/kaggle") else os.path.abspath("longcat_models")

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"  # no second copy of the weights in the xet cache
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    gpus = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip().splitlines()
except Exception:
    gpus = []
if len(gpus) < 2:
    print("❌ This notebook needs 2 GPUs: Settings → Accelerator → GPU T4 x2")
print(f"✅ {len(gpus)} x {gpus[0] if gpus else 'no GPU'} | RAM {psutil.virtual_memory().total / 1024**3:.0f} GB")

# ---- 1. The official LongCat-Video code, fetched by commit (never a branch) ----
t0 = time.time()
if not os.path.isdir("LongCat-Video/.git"):
    subprocess.run(["git", "init", "-q", "LongCat-Video"], check=True)
    subprocess.run(["git", "-C", "LongCat-Video", "remote", "add", "origin", "https://github.com/meituan-longcat/LongCat-Video.git"], check=True)
subprocess.run(["git", "-C", "LongCat-Video", "fetch", "-q", "--depth", "1", "origin", LONGCAT_COMMIT], check=True)
subprocess.run(["git", "-C", "LongCat-Video", "reset", "-q", "--hard", "FETCH_HEAD"], check=True)
print(f"✅ LongCat-Video code @ {LONGCAT_COMMIT[:10]} ({time.time() - t0:.0f}s)")

# ---- 2. Dependencies: only what the avatar pipeline imports ----
t0 = time.time()
pip = [sys.executable, "-m", "pip", "install", "-q", "--no-warn-conflicts", "--timeout", "120", "--retries", "5"]
subprocess.run(pip + ["comfy-kitchen==0.2.35", "transformers==4.57.3", "diffusers==0.36.0", "huggingface_hub[hf_xet]==0.36.0",
                      "gradio==5.29.0", "ftfy==6.3.1", "loguru==0.7.3", "pyloudnorm==0.1.1", "onnxruntime==1.23.2",
                      "onnx2torch==1.5.15", "pydub", "six", "audioread"], check=True)
subprocess.run(pip + ["--no-deps", "audio-separator==0.30.2"], check=True)  # its numpy<2 pin would downgrade Kaggle's numpy
print(f"✅ Dependencies installed ({time.time() - t0:.0f}s)")

# ---- 3. Model weights (parallel; files from an attached Kaggle Dataset are reused) ----
INT8_DIR, BASE_DIR, T5_DIR = (os.path.join(MODELS_DIR, d) for d in ("int8", "base", "umt5"))
BIG = [(INT8_REPO, INT8_REVISION, "base_model/diffusion_pytorch_model_int8_convrot.safetensors", 16.87, INT8_DIR),
       (T5_REPO, T5_REVISION, "umt5-xxl/models_t5_umt5-xxl-enc-bf16.safetensors", 11.36, T5_DIR),
       (AVATAR_REPO, AVATAR_REVISION, "whisper-large-v3/model.safetensors", 3.09, INT8_DIR),
       (INT8_REPO, INT8_REVISION, "loraA/lora.safetensors", 2.52, INT8_DIR),
       (INT8_REPO, INT8_REVISION, "loraB/lora.safetensors", 2.52, INT8_DIR),
       (BASE_REPO, BASE_REVISION, "vae/diffusion_pytorch_model.safetensors", 0.51, BASE_DIR),
       (AVATAR_REPO, AVATAR_REVISION, "vocal_separator/Kim_Vocal_2.onnx", 0.07, INT8_DIR)]
SMALL = [(INT8_REPO, INT8_REVISION, f, 0.0, INT8_DIR) for f in
         ("base_model/config.json", "scheduler/scheduler_config.json", "whisper-large-v3/config.json",
          "whisper-large-v3/preprocessor_config.json", "vocal_separator/download_checks.json",
          "vocal_separator/mdx_model_data.json", "vocal_separator/vr_model_data.json")]
SMALL += [(BASE_REPO, BASE_REVISION, f, 0.0, BASE_DIR) for f in
          ("tokenizer/special_tokens_map.json", "tokenizer/spiece.model", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
           "vae/config.json", "text_encoder/config.json")]
os.makedirs(MODELS_DIR, exist_ok=True)

attached = {}
if os.path.isdir("/kaggle/input"):
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            attached.setdefault(os.path.join(os.path.basename(root), f), os.path.join(root, f))


def fetch(item):
    from huggingface_hub import hf_hub_download
    repo, revision, filename, size_gb, local = item
    dest = os.path.join(local, filename)
    if os.path.exists(dest):
        return
    source = attached.get(os.path.join(os.path.basename(os.path.dirname(filename)), os.path.basename(filename))) if size_gb else None
    if source and os.path.getsize(source) > 0:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.symlink(source, dest)
        return
    for attempt in range(1, 4):
        try:
            hf_hub_download(repo_id=repo, filename=filename, revision=revision, local_dir=local)
            return
        except Exception as e:
            if attempt == 3:
                raise
            say(f"⚠️ {filename}: {str(e).splitlines()[0][:120]} - retrying ({attempt}/3)")
            time.sleep(5 * attempt)


def folder_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            if not os.path.islink(p):
                try:
                    total += os.path.getsize(p)
                except OSError:
                    pass
    return total


todo = [item for item in BIG if not os.path.exists(os.path.join(item[4], item[2]))]
need_gb = sum(item[3] for item in todo)
free_gb = shutil.disk_usage(MODELS_DIR).free / 1024**3
if need_gb > free_gb - 2:
    raise SystemExit(f"❌ Not enough disk space: need {need_gb:.0f} GB, {free_gb:.0f} GB free.")
done = threading.Event()
print_lock = threading.Lock()
bar_shown = [False]


def fmt_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def say(message):
    with print_lock:
        if bar_shown[0]:
            sys.stdout.write("\r" + " " * 100 + "\r")
            bar_shown[0] = False
        print(message, flush=True)


def monitor():
    # One live line: GB done / total, %, smoothed speed, ETA
    base = folder_bytes(MODELS_DIR)
    total = max(need_gb * 1e9, 1.0)
    start, samples = time.time(), []
    while not done.wait(1.0):
        got = folder_bytes(MODELS_DIR) - base
        now = time.time()
        samples = [(t, b) for t, b in samples if now - t <= 10] + [(now, got)]
        speed = (samples[-1][1] - samples[0][1]) / max(samples[-1][0] - samples[0][0], 1e-6) if len(samples) > 1 else 0
        got = min(got, total)
        frac = got / total
        fill = int(frac * 28)
        eta = fmt_time(max(0.0, total - got) / speed) if speed > 1e6 else "--"
        line = (f"⬇️ [{'█' * fill}{'░' * (28 - fill)}] {got / 1e9:5.1f} / {total / 1e9:.1f} GB {frac * 100:3.0f}% | "
                f"{speed / 1e6:4.0f} MB/s | ETA {eta} | {fmt_time(now - start)}")
        with print_lock:
            sys.stdout.write("\r" + line.ljust(100))
            sys.stdout.flush()
            bar_shown[0] = True


if todo:
    threading.Thread(target=monitor, daemon=True).start()
t0 = time.time()
try:
    with ThreadPoolExecutor(max_workers=6) as pool:
        for job in [pool.submit(fetch, item) for item in BIG + SMALL]:
            job.result()
finally:
    done.set()
    time.sleep(1.1)
if need_gb:
    say(f"✅ {need_gb:.1f} GB downloaded in {fmt_time(time.time() - t0)} (avg {need_gb * 1e3 / max(time.time() - t0, 1):.0f} MB/s)")
for d in (INT8_DIR, BASE_DIR, T5_DIR):
    shutil.rmtree(os.path.join(d, ".cache"), ignore_errors=True)

# Settings for Step 2's engine (also saved to a file, so Step 2 can be re-run alone after a kernel restart)
LC_SETTINGS = {"LC_MODELS_DIR": MODELS_DIR, "LC_CODE_DIR": os.path.abspath("LongCat-Video")}
os.environ.update(LC_SETTINGS)
with open("longcat_settings.json", "w") as f:
    json.dump(LC_SETTINGS, f)
print(f"✅ All weights ready. Run Step 2 (disk free: {shutil.disk_usage(MODELS_DIR).free / 1024**3:.0f} GB)")
