# Step 2: Engine + branded Gradio UI
# Writes run_longcat.py, then runs it: model load → public links.
import codecs
import os
import re
import sys
import subprocess

RUN_SCRIPT_CODE = r'''"""LongCat-Video-Avatar 1.6 INT8 engine for Kaggle GPU T4 x2 - AIQUEST Academy.

Step 2 of the notebook writes this file as run_longcat.py and runs it. It drives the official LongCat-Video code
(meituan-longcat/LongCat-Video, pinned commit) with these changes for two 16 GB T4s:

* DiT: the INT8 ConvRot checkpoint. Every block linear runs on Comfy Kitchen's INT8 tensor-core GEMM (W8A8 with the
  ConvRot Hadamard rotation).
* Tensor-parallel DiT: every block is split in half across the two T4s (16 of the 32 heads of each attention, half of
  the FFN), so both GPUs compute every block at the same time and only exchange partial sums. Each GPU's half is queued
  from a CPU thread of its own.
* The two official distillation LoRAs (DMD for the first 4 steps, OPSD for the last 4) run live on the split weights in
  FP16, exactly as the official mix with its default settings.
* AdaLN modulation depends only on the timestep, so it is computed once per timestep in FP32 and cached: those weights
  never sit on a GPU.
* Attention: Comfy Kitchen's INT8 attention for the T4 by default, or exact PyTorch SDPA (FP16).
* UMT5-XXL runs layer by layer from RAM on the real prompt tokens only; prompts are cached.
* The VAE decodes on both GPUs (the official spatial split with blended seams).
"""
import collections
import contextlib
import copy
import faulthandler
import functools
import gc
import glob
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import types
import urllib.request
import warnings

warnings.filterwarnings("ignore")
faulthandler.enable(file=sys.__stderr__, all_threads=True)  # a hard crash (segfault, abort) prints where it happened
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

TEST_MODE = os.environ.get("LC_TEST") == "1"  # CPU tests with tiny random models (never set in the notebook)
CODE_DIR = os.path.abspath(os.environ.get("LC_CODE_DIR", "LongCat-Video"))
MODELS_DIR = os.environ.get("LC_MODELS_DIR", "/kaggle/tmp/longcat_models")
WORK_DIR = os.path.abspath(os.environ.get("LC_WORK_DIR", "."))
OUTPUT_DIR = os.path.join(WORK_DIR, "outputs")
EXAMPLES_DIR = os.path.join(MODELS_DIR, "examples")
CKPT_DIR = os.path.join(MODELS_DIR, "int8")  # the official checkpoint layout (INT8 DiT, LoRAs, whisper, vocal separator)
DIT_FILE = os.path.join(CKPT_DIR, "base_model", "diffusion_pytorch_model_int8_convrot.safetensors")
DIT_CONFIG = os.path.join(CKPT_DIR, "base_model", "config.json")
LORA_FILES = {"dmd": os.path.join(CKPT_DIR, "loraA", "lora.safetensors"), "opsd": os.path.join(CKPT_DIR, "loraB", "lora.safetensors")}
T5_FILE = os.path.join(MODELS_DIR, "umt5", "umt5-xxl", "models_t5_umt5-xxl-enc-bf16.safetensors")
TOKENIZER_DIR = os.path.join(MODELS_DIR, "base", "tokenizer")
VAE_DIR = os.path.join(MODELS_DIR, "base", "vae")
WHISPER_DIR = os.path.join(CKPT_DIR, "whisper-large-v3")
SEPARATOR_DIR = os.path.join(CKPT_DIR, "vocal_separator")
SERVER_PORT = int(os.environ.get("LC_PORT", "7860"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(OUTPUT_DIR, ".gradio_cache"))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

_t_load = time.time()


def _gb(n):
    return f"{n / 1024**3:.1f} GB"


def _ram_in_use():
    """total - available: unlike psutil's 'used', this counts pinned (page-locked) memory, which Linux files as shared."""
    vm = psutil.virtual_memory()
    return f"{_gb(vm.total - vm.available)} / {_gb(vm.total)}"


def _fmt_secs(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def _first_line(e):
    text = str(e).strip()
    return text.splitlines()[0][:160] if text else type(e).__name__


class ConsoleBar:
    """One live progress line per phase, redrawn every second with a carriage return (the notebook cell overwrites it in
    place). A finished phase just clears its line. Step-based phases interpolate inside the current step from the
    previous step's duration, so the bar keeps moving and the ETA counts down between callbacks."""

    WIDTH, LINE = 24, 104
    WATCHED = {"decode": 1800, "save": 600}  # phase -> seconds before a stack dump

    def __init__(self, stream):
        self.out = stream
        self.lock = threading.RLock()
        self.phase = None
        self.visible = False
        self.drawn = 0
        self.text_pending = False
        threading.Thread(target=self._loop, daemon=True).start()

    def start(self, key, label, total=None, unit=""):
        with self.lock:
            if self.phase is not None and self.phase["key"] == key:
                self.phase["label"] = label
                return
            self.finish()
            now = time.time()
            self.phase = dict(key=key, label=label, total=total, unit=unit, done=0, t0=now, t_done=now, dt=None)
            if key in self.WATCHED:
                faulthandler.dump_traceback_later(self.WATCHED[key], file=sys.__stderr__)
            self._draw()

    def update(self, done, total=None):
        with self.lock:
            p = self.phase
            if p is None:
                return
            now = time.time()
            if total:
                p["total"] = total
            if done > p["done"]:
                p["dt"] = (now - p["t_done"]) / (done - p["done"])
                p["t_done"], p["done"] = now, done
            self._draw()

    def finish(self):
        with self.lock:
            p, self.phase = self.phase, None
            if p is None:
                return
            faulthandler.cancel_dump_traceback_later()
            self._clear()
            self.out.flush()

    def _render(self):
        p, now = self.phase, time.time()
        elapsed = now - p["t0"]
        if not p["total"]:
            spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int(now * 4) % 10]
            return f"  {spinner} {p['label']} | {_fmt_secs(elapsed)}"
        progress = p["done"]
        if p["dt"] and p["done"] < p["total"]:
            progress += min(0.97, (now - p["t_done"]) / p["dt"])
        frac = min(1.0, progress / p["total"])
        fill = int(round(frac * self.WIDTH))
        line = (f"  ⏳ {p['label']:<16} [{'█' * fill}{'░' * (self.WIDTH - fill)}] {frac * 100:3.0f}% | "
                f"{p['done']}/{p['total']} {p['unit']}")
        if p["dt"]:
            eta = max(0.0, (p["total"] - p["done"]) * p["dt"] - (now - p["t_done"]))
            line += f" | {p['dt']:.1f}s each | ETA {_fmt_secs(eta)}"
        return line + f" | {_fmt_secs(elapsed)}"

    def _draw(self):
        if self.phase is None or self.text_pending:
            return
        text = self._render()
        self.drawn = max(self.drawn, len(text), self.LINE)
        self.out.write("\r" + text.ljust(self.drawn))
        self.out.flush()
        self.visible = True

    def _clear(self):
        if self.visible:
            self.out.write("\r" + " " * self.drawn + "\r")
            self.visible = False
            self.drawn = 0

    def _loop(self):
        while True:
            time.sleep(1.0)
            with self.lock:
                self._draw()


class _BarAwareStream:
    """stdout / stderr wrapper: ordinary log lines clear the live bar first and print above it."""

    def __init__(self, stream, bar):
        self._stream, self._bar = stream, bar

    def write(self, data):
        with self._bar.lock:
            if data:
                self._bar._clear()
                self._bar.text_pending = not data.endswith("\n")
            return self._stream.write(data)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


BAR = ConsoleBar(sys.stdout)
if not TEST_MODE:
    sys.stdout = _BarAwareStream(sys.stdout, BAR)
    sys.stderr = _BarAwareStream(sys.stderr, BAR)

# ==== Devices ====
# GPU 1 ("main") holds its half of every block plus everything outside the blocks (embedders, final layer, VAE, the
# sampler state); GPU 0 holds the other half of every block and a second VAE for the split decode.
if TEST_MODE:
    DEVICES = (torch.device("cpu"), torch.device("cpu"))
    COMPUTE = torch.float32
else:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("❌ This notebook needs Kaggle's 'GPU T4 x2' accelerator (Settings → Accelerator → GPU T4 x2).")
    DEVICES = (torch.device("cuda", 0), torch.device("cuda", 1))
    COMPUTE = torch.float16  # Turing has FP16 tensor cores, no BF16 ones
    torch.cuda.set_device(DEVICES[1])
    torch.backends.cuda.enable_flash_sdp(False)  # Turing (SM 7.5) has no flash kernel
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
MAIN_DEV = DEVICES[1]


def _on(dev):
    """torch.cuda.device(dev) for a CUDA device; nothing for the CPU."""
    return torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext()


def _empty_cache():
    # Wait for queued GPU work first: with expandable segments, empty_cache unmaps freed memory right away, and a kernel
    # still reading a just-freed tensor (Whisper weights after unload, for example) then hits unmapped pages.
    gc.collect()
    for dev in DEVICES:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
    for dev in DEVICES:
        if dev.type == "cuda":
            with torch.cuda.device(dev):
                torch.cuda.empty_cache()


if not TEST_MODE and torch.cuda.get_device_properties(0).major < 8:
    # Turing has no BF16 convolution kernels: convolutions that reach BF16 (the DiT's patch embedding) run in FP32.
    def _conv_fp32_on_turing(original):
        def forward(self, input, weight, bias):
            if input.is_cuda and input.dtype == torch.bfloat16:
                bias32 = None if bias is None else bias.float()
                return original(self, input.float(), weight.float(), bias32).to(torch.bfloat16)
            return original(self, input, weight, bias)
        return forward
    for _conv in (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d):
        _conv._conv_forward = _conv_fp32_on_turing(_conv._conv_forward)


# ==== The official LongCat-Video code ====
# Block-sparse attention (Triton) is never used here (the model config disables it), so its module is replaced by a
# stand-in before the official modules import it.
_bsa = types.ModuleType("longcat_video.block_sparse_attention.bsa_interface")


def _no_bsa(*_args, **_kwargs):
    raise RuntimeError("Block-sparse attention is not used in this notebook")


_bsa.flash_attn_bsa_3d = _no_bsa
sys.modules["longcat_video.block_sparse_attention.bsa_interface"] = _bsa

import loguru as _loguru

_loguru.logger.remove()  # the pipeline's resolution warnings are not shown
from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from longcat_video.modules.avatar.rope_3d import RotaryPositionalEmbedding, RotaryPositionalEmbedding1D
from longcat_video.modules.avatar import attention as lc_attention
from longcat_video.modules.blocks import modulate_fp32
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from safetensors import safe_open

# ==== INT8 ConvRot linears on Comfy Kitchen ====
# The checkpoint stores each block linear as INT8 in a Hadamard-rotated basis (groups of 256 input channels) with one
# FP32 scale per output channel. Kitchen's INT8 GEMM rotates and quantizes the activations inside its own kernel, so a
# layer is one call. Only Kitchen's CUDA backend is used: its Triton INT8 path cannot compile for the T4.
KITCHEN = {"ok": False, "reason": ""}
_ck = None
if not TEST_MODE:
    try:
        from comfy_kitchen.backends import cuda as _ck
        KITCHEN["ok"] = bool(getattr(_ck, "_EXT_AVAILABLE", False)) and all(
            torch.cuda.get_device_capability(d) >= (7, 5) for d in DEVICES)
        if not KITCHEN["ok"]:
            KITCHEN["reason"] = "Comfy Kitchen's CUDA kernels are not available on this machine"
    except Exception as e:
        KITCHEN["reason"] = f"Comfy Kitchen could not be loaded: {_first_line(e)}"
_HADAMARD = {}


def hadamard(size, device):
    """The regular Hadamard matrix of the ConvRot format (Kronecker powers of a symmetric 4x4), normalized."""
    key = (size, str(device))
    if key not in _HADAMARD:
        h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32, device=device)
        h = h4
        while h.shape[0] < size:
            h = torch.kron(h, h4)
        _HADAMARD[key] = h.mul(size ** -0.5)
    return _HADAMARD[key]


def dequantize(qweight, scale, group):
    """INT8 ConvRot weight back to the original basis, in FP32."""
    w = qweight.float() * scale.float().reshape(-1, 1)
    return torch.matmul(w.reshape(w.shape[0], -1, group), hadamard(group, w.device)).reshape(w.shape)


def quantize(weight, group):
    """Plain absmax ConvRot quantization (the startup self-test and the CPU tests)."""
    r = torch.matmul(weight.float().reshape(weight.shape[0], -1, group), hadamard(group, weight.device).T).reshape(weight.shape)
    scale = (r.abs().amax(dim=1, keepdim=True) / 127).clamp_(min=1e-30)
    return (r / scale).round_().clamp_(-127, 127).to(torch.int8), scale.reshape(-1)


# This model's activations reach values far beyond FP16's range (65504): the residual stream, the FFN and the output
# projections run into the millions. Every GEMM writes BF16 (FP32's range, the official model's own precision). Its
# input is FP16 (Kitchen's GEMM is up to 1.8x slower on BF16 input on the T4), scaled by a power of two chosen on the GPU
# so it always fits; Kitchen quantizes each input row to INT8 with its own scale, so that factor changes nothing, and it
# is undone on the output together with the bias. Attention and the LoRA matmuls also run in FP16 on scaled inputs.
ACT = torch.float32 if TEST_MODE else torch.bfloat16
KITCHEN_OUT = {"bf16": True}  # Kitchen writes BF16 straight from its GEMM (checked at startup; else FP32, then BF16)


def _to_fp16(t, target=1024.0):
    """t in the compute dtype, scaled by a power of two (exact) so that its largest value is at most `target`; returns
    (scaled tensor, 0-dim inverse scale). Decided on the GPU (no host sync); values are never scaled up."""
    amax = t.detach().abs().amax().float().clamp_min(1e-30)
    exp = torch.ceil(torch.log2(amax / target)).clamp_min(0)
    return (t * torch.exp2(-exp).to(t.dtype)).to(COMPUTE), torch.exp2(exp)


class Int8Linear(nn.Module):
    """A linear whose weight stays INT8 on its GPU and runs on Kitchen's INT8 tensor-core GEMM. Without Kitchen (CPU
    tests) the weight is expanded once to the compute dtype. forward_scaled takes the FP16 input from _to_fp16 and returns
    the output in ACT, still at the input's scale and without the bias (see run_linear)."""

    def __init__(self, qweight, scale, bias, group, device):
        super().__init__()
        self.out_features, self.in_features = qweight.shape
        self.group = group
        self.kitchen = KITCHEN["ok"] and device.type == "cuda" and group == 256 and self.out_features % 4 == 0
        if self.kitchen:
            self.register_buffer("qweight", qweight.contiguous().to(device, copy=True), persistent=False)
            self.register_buffer("qscale", scale.float().reshape(-1).contiguous().to(device, copy=True), persistent=False)
            self.weight = None
        else:
            self.register_buffer("weight", dequantize(qweight, scale, group).to(device, COMPUTE), persistent=False)
        self.register_buffer("bias", None if bias is None else bias.to(device, ACT).contiguous(), persistent=False)

    def forward_scaled(self, x16, inv=None):
        """FP16 input at scale 1/inv (inv None: full scale) -> output in ACT at the same scale, bias included (scaled to
        match), so the caller only multiplies by inv."""
        x2 = x16.reshape(-1, self.in_features)
        out = ACT if KITCHEN_OUT["bf16"] else torch.float32
        bias = None
        if self.bias is not None:
            bias = self.bias if inv is None else (self.bias.float() / inv).to(ACT)
            bias = bias if bias.dtype == out else bias.to(out)
        if self.kitchen:
            x2 = x2 if x2.is_contiguous() else x2.contiguous()
            y = _ck.int8_linear(x2, self.qweight, self.qscale, bias=bias, out_dtype=out, convrot=True, convrot_groupsize=self.group)
            y = y if y.dtype == ACT else y.to(ACT)
        else:
            y = F.linear(x2, self.weight, None if bias is None else bias.to(self.weight.dtype)).to(ACT)
        return y.reshape(*x16.shape[:-1], self.out_features)


def probe_kitchen():
    """Kitchen's INT8 GEMM against the exact result on random data at the model's width, on both GPUs, at a short and a
    long input (Kitchen routes them to different kernels): FP16 input with FP16 output must match, and BF16 output is used
    only if it matches too (else FP32 output, converted)."""
    if not KITCHEN["ok"]:
        return
    try:
        g = torch.Generator(device="cpu").manual_seed(0)
        w = torch.randn(1024, 4096, generator=g) * 0.02
        q, s = quantize(w, 256)
        for dev in DEVICES:
            with _on(dev):
                wd = dequantize(q, s, 256).to(dev)
                qd, sd = q.to(dev), s.to(dev)
                for rows in (512, 8192):
                    x = (torch.randn(rows, 4096, device=dev) * 3).to(COMPUTE)
                    ref = x.float() @ wd.T
                    for out in (COMPUTE, torch.bfloat16, torch.float32):
                        try:
                            y = _ck.int8_linear(x, qd, sd, out_dtype=out, convrot=True).float()
                            diff = ((y - ref).norm() / ref.norm()).item()
                        except Exception:
                            diff = float("inf")
                        ok = math.isfinite(diff) and diff < 0.05
                        if out == COMPUTE and not ok:
                            KITCHEN.update(ok=False, reason=f"self-test differed {diff:.1%}")
                            return
                        if out == torch.bfloat16 and not ok:
                            KITCHEN_OUT["bf16"] = False
                        if out == torch.float32 and not ok and not KITCHEN_OUT["bf16"]:
                            KITCHEN.update(ok=False, reason="no usable output type")
                            return
    except Exception as e:
        KITCHEN.update(ok=False, reason=f"self-test failed: {_first_line(e)}")


probe_kitchen()
if not TEST_MODE and not KITCHEN["ok"]:
    raise SystemExit(f"❌ Comfy Kitchen's INT8 kernels are required on the T4 ({KITCHEN['reason']}). Re-run Step 1.")
INT8_LABEL = "Comfy Kitchen INT8 tensor cores"


# ==== Fused norm / modulation / residual kernels ====
# Each block normalizes and modulates the FP32 residual stream four times and adds four gated results back to it. In
# plain PyTorch every one of those is several full-size passes with a per-frame broadcast, which runs on PyTorch's slow
# generic elementwise kernel (about 25 ms per pass at 39k tokens on a T4, a third of a step in total). These Triton
# kernels do each one in a single pass over the tokens: layer norm (+ affine or + AdaLN modulation) straight to the
# FP16 GEMM input, and gate * update added into the residual. A startup test compares them with the PyTorch path on
# both GPUs; if Triton fails there, the PyTorch path is used. LayerNorm outputs are bounded (|x| <= sqrt(4096) before
# the modulation), so the FP16 GEMM inputs made here need no scaling; they are clamped to FP16's range all the same.
FUSED = {"on": False, "reason": ""}
try:
    if TEST_MODE:
        raise ImportError("CPU tests use the PyTorch path")
    import triton
    import triton.language as tl

    @triton.jit
    def _ln_to_fp16_kernel(X, OUT, W, BIAS, SHIFT, SCALE, rows_per_frame, mod_stride, eps,
                           C: tl.constexpr, BLOCK: tl.constexpr, AFFINE: tl.constexpr, MOD: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < C
        x = tl.load(X + row.to(tl.int64) * C + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / C
        xc = tl.where(mask, x - mean, 0.0)
        y = xc * tl.rsqrt(tl.sum(xc * xc, axis=0) / C + eps)
        if AFFINE:
            y = y * tl.load(W + cols, mask=mask, other=1.0) + tl.load(BIAS + cols, mask=mask, other=0.0)
        if MOD:
            f = (row // rows_per_frame).to(tl.int64) * mod_stride
            y = y * (1.0 + tl.load(SCALE + f + cols, mask=mask, other=0.0)) + tl.load(SHIFT + f + cols, mask=mask, other=0.0)
        y = tl.minimum(tl.maximum(y, -65504.0), 65504.0)
        tl.store(OUT + row.to(tl.int64) * C + cols, y.to(tl.float16), mask=mask)

    @triton.jit
    def _gated_add_kernel(X, Y, G, rows_per_frame, g_stride, C: tl.constexpr, BLOCK: tl.constexpr, GATE: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < C
        off = row.to(tl.int64) * C + cols
        x = tl.load(X + off, mask=mask, other=0.0)
        y = tl.load(Y + off, mask=mask, other=0.0).to(tl.float32)
        if GATE:
            y = y * tl.load(G + (row // rows_per_frame).to(tl.int64) * g_stride + cols, mask=mask, other=0.0)
        tl.store(X + off, x + y, mask=mask)

    @triton.jit
    def _ln_mod_gated_add_kernel(X, Y, SHIFT, SCALE, G, rows_per_frame, mod_stride, eps, C: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < C
        off = row.to(tl.int64) * C + cols
        y = tl.load(Y + off, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(y, axis=0) / C
        yc = tl.where(mask, y - mean, 0.0)
        y = yc * tl.rsqrt(tl.sum(yc * yc, axis=0) / C + eps)
        f = (row // rows_per_frame).to(tl.int64) * mod_stride
        y = y * (1.0 + tl.load(SCALE + f + cols, mask=mask, other=0.0)) + tl.load(SHIFT + f + cols, mask=mask, other=0.0)
        y = y * tl.load(G + f + cols, mask=mask, other=0.0)
        tl.store(X + off, tl.load(X + off, mask=mask, other=0.0) + y, mask=mask)

    FUSED["on"] = True
except Exception as e:
    FUSED["reason"] = _first_line(e)


def _rows(t):
    return t.reshape(-1, t.shape[-1])


def ln_to_fp16(x, per, shift=None, scale=None, w=None, b=None, fused=None):
    """LayerNorm(x) (eps 1e-6, FP32 math), then the affine weights or the AdaLN modulation x * (1 + scale) + shift
    (per-frame rows: frame = token // per), as the FP16 GEMM input. x: FP32 [B, N, C] (contiguous rows)."""
    if (FUSED["on"] if fused is None else fused) and x.is_cuda and x.is_contiguous():
        c = x.shape[-1]
        out = torch.empty(x.shape, dtype=COMPUTE, device=x.device)
        dummy = x
        _ln_to_fp16_kernel[(x.numel() // c,)](x, out, w if w is not None else dummy, b if b is not None else dummy,
                                              shift if shift is not None else dummy, scale if scale is not None else dummy,
                                              per, shift.stride(0) if shift is not None else 0, 1e-6, C=c,
                                              BLOCK=triton.next_power_of_2(c), AFFINE=w is not None, MOD=shift is not None,
                                              num_warps=8)
        return out
    bsz, n, c = x.shape
    if shift is not None:
        y = modulate_fp32(_ln, x.view(bsz, n // per, per, c), shift.reshape(bsz, -1, 1, c), scale.reshape(bsz, -1, 1, c)).view(bsz, n, c)
    else:
        y = _ln(x, w, b)
    return y.clamp(-65504.0, 65504.0).to(COMPUTE) if COMPUTE == torch.float16 else y.to(COMPUTE)


def gated_add_(x, y, per, gate=None, fused=None):
    """x += gate * y (gate per frame, or none), in place. x: FP32 [B, N, C] (contiguous rows), y: the same shape."""
    if (FUSED["on"] if fused is None else fused) and x.is_cuda and x.is_contiguous():
        c = x.shape[-1]
        _gated_add_kernel[(x.numel() // c,)](x, y.contiguous(), gate if gate is not None else x, per,
                                             gate.stride(0) if gate is not None else 0, C=c, BLOCK=triton.next_power_of_2(c),
                                             GATE=gate is not None, num_warps=8)
        return x
    bsz, n, c = x.shape
    if gate is None:
        return x.add_(y)
    x.view(bsz, n // per, per, c).addcmul_(gate.reshape(bsz, -1, 1, c), y.reshape(bsz, n // per, per, c))
    return x


def ln_mod_gated_add_(x, y, per, shift, scale, gate, fused=None):
    """x += gate * (LayerNorm(y) * (1 + scale) + shift): the official audio branch output, in place."""
    if (FUSED["on"] if fused is None else fused) and x.is_cuda and x.is_contiguous():
        c = x.shape[-1]
        _ln_mod_gated_add_kernel[(x.numel() // c,)](x, y.contiguous(), shift, scale, gate, per, shift.stride(0), 1e-6, C=c,
                                                    BLOCK=triton.next_power_of_2(c), num_warps=8)
        return x
    bsz, n, c = x.shape
    out = modulate_fp32(_ln, y.reshape(bsz, n // per, per, c), shift.reshape(bsz, -1, 1, c), scale.reshape(bsz, -1, 1, c))
    x.view(bsz, n // per, per, c).addcmul_(gate.reshape(bsz, -1, 1, c), out)
    return x


def _ln(x, w=None, b=None):
    """LayerNorm_FP32 of the official code (eps 1e-6), output in x's dtype."""
    return F.layer_norm(x.float(), (x.shape[-1],), w, b, 1e-6).to(x.dtype)


def probe_fused():
    """The Triton kernels against the PyTorch path on random data, on both GPUs (off if they fail or differ)."""
    if not FUSED["on"]:
        return
    try:
        for dev in DEVICES:
            with _on(dev):
                g = torch.Generator(device=dev).manual_seed(0)
                per, frames, c = 96, 5, 4096
                x = torch.randn(1, per * frames, c, device=dev, generator=g) * 30
                mods = torch.randn(frames, 3 * c, device=dev, generator=g)
                sh, sc, gt = mods[:, :c], mods[:, c:2 * c], mods[:, 2 * c:]
                w, b = torch.randn(c, device=dev, generator=g), torch.randn(c, device=dev, generator=g)
                y = (torch.randn(1, per * frames, c, device=dev, generator=g) * 100).to(ACT)
                checks = [(ln_to_fp16(x, per, sh, sc, fused=True), ln_to_fp16(x, per, sh, sc, fused=False)),
                          (ln_to_fp16(x, per, w=w, b=b, fused=True), ln_to_fp16(x, per, w=w, b=b, fused=False)),
                          (gated_add_(x.clone(), y, per, gt, fused=True), gated_add_(x.clone(), y, per, gt, fused=False)),
                          (gated_add_(x.clone(), y, per, fused=True), gated_add_(x.clone(), y, per, fused=False)),
                          (ln_mod_gated_add_(x.clone(), y, per, sh, sc, gt, fused=True),
                           ln_mod_gated_add_(x.clone(), y, per, sh, sc, gt, fused=False))]
                for a, r in checks:
                    diff = ((a.float() - r.float()).norm() / r.float().norm()).item()
                    if not math.isfinite(diff) or diff > 2e-3:
                        FUSED.update(on=False, reason=f"self-test differed {diff:.2e}")
                        return
    except Exception as e:
        FUSED.update(on=False, reason=f"self-test failed: {_first_line(e)}")


probe_fused()

# ==== Attention: Comfy Kitchen INT8 (Fast) or exact FP16 SDPA ====
# Self-attention over ~37-39k video tokens is most of a step. Fast: Comfy Kitchen's INT8 attention, a compiled CUDA
# kernel with Turing code (Hadamard-rotated INT8 Q/K, INT8 V and softmax weights, FP32 softmax; the T4's INT8 tensor
# cores run at twice the FP16 rate). It is offered only when a startup test on both GPUs passes and shows it faster, and
# every run compares it with exact attention on two blocks of its first step, finishing with exact attention by itself
# if they differ too much. Exact: PyTorch SDPA in FP16 (memory-efficient kernel; the T4 has no flash kernel).
ATTENTION_FAST, ATTENTION_EXACT = "Fast (Kitchen INT8)", "Exact (FP16)"
ATTENTION_MAX_DIFF = 0.08
INT8_MIN_TOKENS = 2048
ATTN = {"int8": None, "dtype": None, "contiguous": False, "speedup": None, "reason": "", "active": False,
        "checks": set(), "fallback": None}
_ATTN_LOCK = threading.Lock()


def _sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)


def _int8_attention(q, k, v):
    """Kitchen INT8 attention on [batch, heads, tokens, dim] tensors, output in q's dtype."""
    dtype, out_dtype = ATTN["dtype"] or q.dtype, q.dtype
    args = [(t if t.dtype == dtype else t.to(dtype)).contiguous() for t in (q, k, v)]  # the kernel gets dense tensors
    return ATTN["int8"].int8_attention(*args).to(out_dtype)


def _relative_diff(out, ref):
    """||out - ref|| / ||ref|| in FP32, by slices of tokens (no full-size FP32 copies)."""
    num = torch.zeros((), dtype=torch.float32, device=out.device)
    den = torch.zeros((), dtype=torch.float32, device=out.device)
    for a in range(0, out.shape[2], 4096):
        o, r = out[:, :, a:a + 4096].float(), ref[:, :, a:a + 4096].float()
        num += (o - r).square().sum()
        den += r.square().sum()
    return (num / den.clamp_min(1e-30)).sqrt().item()


def attention(q, k, v, block=-1):
    """Self-attention of the DiT blocks: q / k / v [batch, heads, tokens, dim]."""
    if not (ATTN["active"] and q.is_cuda and q.shape[2] >= INT8_MIN_TOKENS and k.shape[2] >= INT8_MIN_TOKENS):
        return _sdpa(q, k, v)
    check = False
    if q.device == DEVICES[0] and ATTN["checks"]:
        with _ATTN_LOCK:
            check = block in ATTN["checks"]
            ATTN["checks"].discard(block)
    if not check:
        return _int8_attention(q, k, v)
    free = torch.cuda.mem_get_info(q.device)[0] + torch.cuda.memory_reserved(q.device) - torch.cuda.memory_allocated(q.device)
    if free < 3 * q.numel() * q.element_size() + (512 << 20):
        return _int8_attention(q, k, v)
    ref = _sdpa(q, k, v)
    out = _int8_attention(q, k, v)
    diff = _relative_diff(out, ref)
    if not math.isfinite(diff) or diff > ATTENTION_MAX_DIFF:
        ATTN.update(active=False, fallback=diff)
        return ref
    return out


def _probe_case(dev, fused):
    """INT8 vs SDPA on random data in the model's layouts: v as a view of the fused QKV output (the normal path) or
    separate contiguous tensors."""
    g = torch.Generator(device=dev).manual_seed(0)
    tokens, heads, dim = 3000, 4, 128
    if fused:
        qkv = torch.randn(1, tokens, 3, heads, dim, device=dev, generator=g).to(COMPUTE).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = q.contiguous(), k.contiguous()
    else:
        q, k, v = (torch.randn(1, heads, tokens, dim, device=dev, generator=g).to(COMPUTE) for _ in range(3))
    return _relative_diff(_int8_attention(q, k, v), _sdpa(q, k, v))


def probe_int8_attention():
    if TEST_MODE:
        ATTN["reason"] = "needs a GPU"
        return
    try:
        import comfy_kitchen
        if not all(comfy_kitchen.int8_attention_is_available(dev) for dev in DEVICES):
            ATTN["reason"] = "not available on this GPU"
            return
        ATTN["int8"] = comfy_kitchen
        for dtype, contiguous in ((torch.float16, True),):
            ATTN.update(dtype=dtype, contiguous=contiguous)
            try:
                worst = 0.0
                for dev in DEVICES:
                    with _on(dev), torch.inference_mode():
                        worst = max(worst, _probe_case(dev, True), _probe_case(dev, False))
                if math.isfinite(worst) and worst < 0.1:
                    break
                ATTN["reason"] = f"self-test differed {worst:.1%} from SDPA"
            except Exception as e:
                ATTN["reason"] = f"self-test failed: {_first_line(e)}"
            ATTN["dtype"] = None
        if ATTN["dtype"] is None:
            ATTN["int8"] = None
            return
        ATTN["reason"] = ""
        with _on(MAIN_DEV), torch.inference_mode():
            q, k, v = (torch.randn(1, 16, 8192, 128, device=MAIN_DEV).to(COMPUTE) for _ in range(3))
            timings = []
            for run in (lambda: _sdpa(q, k, v), lambda: _int8_attention(q, k, v)):
                for _ in range(2):
                    run()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(3):
                    run()
                end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end))
            ratio = timings[0] / max(timings[1], 1e-3)
            del q, k, v
        if ratio < 1.05:
            ATTN.update(int8=None, dtype=None, reason=f"not faster than exact attention here: {ratio:.2f}x")
        else:
            ATTN["speedup"] = ratio
    except Exception as e:
        ATTN.update(int8=None, dtype=None, reason=f"unavailable: {_first_line(e)}")
    finally:
        _empty_cache()


def start_attention(choice, num_blocks):
    """Arms INT8 attention for one run (or not); the checks run on blocks 1 and N/2 of the first forward pass."""
    on = choice == ATTENTION_FAST and ATTN["int8"] is not None
    ATTN.update(active=on, checks={1, num_blocks // 2} if on else set(), fallback=None)
    return on


BAR.start("attention", "Loading: testing Kitchen INT8 attention")
probe_int8_attention()
BAR.finish()
ATTENTION_CHOICES = ([ATTENTION_FAST] if ATTN["int8"] is not None else []) + [ATTENTION_EXACT]
ATTENTION_DEFAULT = ATTENTION_CHOICES[0]
ATTENTION_LABEL = "Kitchen INT8 attention" if ATTN["int8"] is not None else "SDPA attention"


# ==== The DiT: built from the INT8 ConvRot file, split in half across the two GPUs ====
with open(DIT_CONFIG) as _f:
    DIT_CFG = {k: v for k, v in json.load(_f).items() if not k.startswith("_") and k not in ("architectures", "model_max_length")}
DIT_CFG["cp_split_hw"] = [1, 1]
HIDDEN, NUM_HEADS, DEPTH = DIT_CFG["hidden_size"], DIT_CFG["num_heads"], DIT_CFG["depth"]
HEAD_DIM = HIDDEN // NUM_HEADS
HALF_HEADS = NUM_HEADS // 2
HALF_INNER = HALF_HEADS * HEAD_DIM
NONBLOCK_DTYPE = torch.float32 if TEST_MODE else torch.bfloat16  # embedders and final layer: the official dtype

with torch.device("meta"):
    dit = LongCatVideoAvatarTransformer3DModel(**DIT_CFG)
dit.blocks = nn.ModuleList()  # the blocks live in SHARDS below (one half per GPU)
dit.eval().requires_grad_(False)


def _set_tensor(model, name, tensor):
    *path, leaf = name.split(".")
    parent = model
    for part in path:
        parent = getattr(parent, part)
    if leaf in parent._parameters:
        parent._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
    else:
        parent._buffers[leaf] = tensor


def _comfy_group(f, base):
    try:
        layer = json.loads(bytes(f.get_tensor(base + ".comfy_quant").tolist()).decode("utf-8"))
        return int(layer.get("convrot_groupsize", 256))
    except Exception:
        return 256


class Shard(nn.Module):
    """One GPU's half of one block: 16 heads of each attention, part of the FFN, the norms it needs."""


def _ffn_split(ffn):
    """Where the FFN is cut: a multiple of 256 (ConvRot groups stay whole on each GPU), as close to half as possible."""
    cut = (ffn // 2) // 256 * 256
    return [(0, cut), (cut, ffn)]


ADALN = []  # per block: AdaLN weights (6C and 3C rows) kept on the CPU; see modulation()


def _build_block(f, b):
    p = f"blocks.{b}."
    get = lambda name: f.get_tensor(p + name)
    keys = set(f.keys())

    def quant(name):
        base = p + name
        return f.get_tensor(base + ".weight"), f.get_tensor(base + ".weight_scale").reshape(-1), _comfy_group(f, base), \
            (f.get_tensor(base + ".bias") if base + ".bias" in keys else None)

    ADALN.append(tuple(get(n).to(torch.bfloat16) for n in ("adaLN_modulation.1.weight", "adaLN_modulation.1.bias",
                                                          "audio_adaLN_modulation.1.weight", "audio_adaLN_modulation.1.bias")))
    tensors = {n: quant(n) for n in ("attn.qkv", "attn.proj", "cross_attn.q_linear", "cross_attn.kv_linear", "cross_attn.proj",
                                     "audio_cross_attn.q_linear", "audio_cross_attn.kv_linear", "audio_cross_attn.proj",
                                     "ffn.w1", "ffn.w2", "ffn.w3")}
    ffn = tensors["ffn.w1"][0].shape[0]
    splits = _ffn_split(ffn)
    shards = []
    for s, dev in enumerate(DEVICES):
        heads = torch.arange(s * HALF_INNER, (s + 1) * HALF_INNER)
        cols = slice(s * HALF_INNER, (s + 1) * HALF_INNER)
        f0, f1 = splits[s]
        sh = Shard()

        def rows(name, parts, bias=True):
            q, sc, g, bi = tensors[name]
            r = torch.cat([heads + i * HIDDEN for i in range(parts)])
            return Int8Linear(q.index_select(0, r), sc.index_select(0, r), bi.index_select(0, r) if bi is not None and bias else None, g, dev)

        def colsplit(name, c):
            q, sc, g, bi = tensors[name]
            return Int8Linear(q[:, c], sc, bi if s == 0 else None, g, dev)  # the bias is added once, on GPU 0's partial

        sh.qkv, sh.proj = rows("attn.qkv", 3), colsplit("attn.proj", cols)
        sh.c_q, sh.c_kv, sh.c_proj = rows("cross_attn.q_linear", 1), rows("cross_attn.kv_linear", 2), colsplit("cross_attn.proj", cols)
        sh.a_q, sh.a_kv, sh.a_proj = (rows("audio_cross_attn.q_linear", 1), rows("audio_cross_attn.kv_linear", 2),
                                      colsplit("audio_cross_attn.proj", cols))
        (q1, s1, g1, _), (q3, s3, g3, _) = tensors["ffn.w1"], tensors["ffn.w3"]
        sh.w13 = Int8Linear(torch.cat([q1[f0:f1], q3[f0:f1]]), torch.cat([s1[f0:f1], s3[f0:f1]]), None, g1, dev)  # [gate | up]
        sh.w2 = colsplit("ffn.w2", slice(f0, f1))
        sh.ffn_cut = (f0, f1)
        for name in ("attn.q_norm", "attn.k_norm", "cross_attn.q_norm", "cross_attn.k_norm", "audio_cross_attn.q_norm",
                     "audio_cross_attn.k_norm"):
            sh.register_buffer(name.replace(".", "_"), get(name + ".weight").to(dev, COMPUTE), persistent=False)
        for name in ("pre_crs_attn_norm", "pre_video_crs_attn_norm"):
            sh.register_buffer(name + "_w", get(name + ".weight").to(dev, torch.float32), persistent=False)
            sh.register_buffer(name + "_b", get(name + ".bias").to(dev, torch.float32), persistent=False)
        sh.lora = {}
        shards.append(sh)
    return shards


SHARDS = []
BAR.start("load", "Loading: DiT (INT8) onto both GPUs", DEPTH, "blocks")
with safe_open(DIT_FILE, framework="pt", device="cpu") as _f:
    for _key in _f.keys():
        if not _key.startswith("blocks."):
            _set_tensor(dit, _key, _f.get_tensor(_key).to(MAIN_DEV, NONBLOCK_DTYPE))
    for _b in range(DEPTH):
        SHARDS.append(_build_block(_f, _b))
        BAR.update(_b + 1)
BAR.finish()
_left = [n for n, t in list(dit.named_parameters()) + list(dit.named_buffers()) if t is not None and t.is_meta]
if _left:
    raise RuntimeError(f"{len(_left)} DiT weights missing from the checkpoint, e.g. {_left[:3]}")
gc.collect()

# ==== The two official distillation LoRAs, live on the split weights ====
# LongCat-Video-Avatar 1.6 mixes two LoRAs per step: with the official defaults (mix_steps=4, mix_alpha=1.0) the first
# four steps use the DMD LoRA and the last four the OPSD LoRA. Both stay on the GPUs in FP16 (folded with their
# alpha_scale) and are applied next to each INT8 GEMM: x -> down -> up, added to the layer's output. Merging them into
# the INT8 weights would round most of their small changes away. Column-split layers (Q/K/V, the FFN input) use the full
# down matrix and their own rows of the up matrix; row-split layers (the output projections, FFN output) use their own
# columns of the down matrix, and the partial results add up in the exchange like the layer's own partial sums.
MIX_STEPS, MIX_ALPHA = 4, 1.0
LORA_STATE = {"active": [], "step": 0}
LORA_TARGETS = {  # LoRA module -> (shard layer, kind, parts)
    "attn.qkv": ("qkv", "rows", 3), "attn.proj": ("proj", "cols", 1), "cross_attn.q_linear": ("c_q", "rows", 1),
    "cross_attn.kv_linear": ("c_kv", "rows", 2), "ffn.w2": ("w2", "cols", 1),
}


def _lora_name(b, module):
    return "lora___lorahyphen___" + "___lorahyphen___".join(["blocks", str(b)] + module.split("."))


def _load_lora(key, path):
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for b in range(DEPTH):
            def parts(module):
                name = _lora_name(b, module)
                down = f.get_tensor(name + ".lora_down.weight").float()
                scale = float(f.get_tensor(name + ".alpha_scale").float().item()) if name + ".alpha_scale" in keys else 1.0
                ups = sorted(k for k in keys if k.startswith(name + ".lora_up.blocks."))
                ups = [f.get_tensor(k).float() for k in sorted(ups, key=lambda k: int(k.split(".")[-2]))] if ups else \
                    [f.get_tensor(name + ".lora_up.weight").float()]
                return down, [u * scale for u in ups]

            loaded = {module: parts(module) for module in list(LORA_TARGETS) + ["ffn.w1", "ffn.w3"]}
            for s, dev in enumerate(DEVICES):
                sh = SHARDS[b][s]
                entry = {}
                for module, (layer, kind, n) in LORA_TARGETS.items():
                    down, ups = loaded[module]
                    if kind == "rows":
                        r = slice(s * HALF_INNER, (s + 1) * HALF_INNER)
                        entry[layer] = ("rows", down.to(dev, COMPUTE), torch.stack([u[r] for u in ups]).to(dev, COMPUTE))
                    else:
                        c = sh.ffn_cut if layer == "w2" else (s * HALF_INNER, (s + 1) * HALF_INNER)
                        entry[layer] = ("cols", down[:, c[0]:c[1]].contiguous().to(dev, COMPUTE), ups[0].to(dev, COMPUTE))
                (d1, u1), (d3, u3) = loaded["ffn.w1"], loaded["ffn.w3"]
                f0, f1 = sh.ffn_cut
                entry["w13"] = ("rows", torch.cat([d1, d3]).to(dev, COMPUTE), torch.stack([u1[0][f0:f1], u3[0][f0:f1]]).to(dev, COMPUTE))
                sh.lora[key] = entry
            BAR.update(b + 1)


for _key, _path in LORA_FILES.items():
    BAR.start("lora", f"Loading: {_key.upper()} LoRA", DEPTH, "blocks")
    _load_lora(_key, _path)
    BAR.finish()
gc.collect()


def _lora_delta(entry, x):
    kind, down, up = entry
    lx = F.linear(x, down)
    if kind == "cols":
        return F.linear(lx, up)
    n = up.shape[0]
    if n == 1:
        return F.linear(lx, up[0])
    lead = lx.shape[:-1]
    lx = lx.reshape(-1, n, lx.shape[-1] // n).transpose(0, 1)  # [n, rows, r]
    out = torch.bmm(lx, up.transpose(1, 2))                      # [n, rows, out_n]
    return out.transpose(0, 1).reshape(*lead, n * up.shape[1])


def run_linear(sh, name, x, scaled=None):
    """One shard linear with the active LoRAs: x (any dtype) -> output in ACT. The input is scaled into FP16 once; the
    INT8 GEMM and the LoRA matmuls (FP16) both work on it, and the scale and the bias are undone on the sum. scaled:
    an input already in that form, (FP16 tensor, inverse scale), such as an attention output."""
    lin = getattr(sh, name)
    x16, inv = scaled if scaled is not None else _to_fp16(x)  # inv None: the input is already at full scale
    y = lin.forward_scaled(x16, inv)
    for key, multiplier in LORA_STATE["active"]:
        entry = sh.lora.get(key, {}).get(name)
        if entry is not None:
            delta = _lora_delta(entry, x16)
            y.add_(delta if multiplier == 1.0 else delta.mul_(multiplier))
    del x16
    if inv is not None:
        y.mul_(inv)
    return y


def set_lora_step(i):
    """The official 1.6 schedule: the first MIX_STEPS steps use alpha * DMD + (1 - alpha) * OPSD, the rest OPSD only."""
    if i < MIX_STEPS:
        active = [(k, m) for k, m in (("dmd", MIX_ALPHA), ("opsd", 1.0 - MIX_ALPHA)) if m != 0.0]
    else:
        active = [("opsd", 1.0)]
    LORA_STATE.update(active=active, step=i)


set_lora_step(0)

# ==== AdaLN modulation: computed once per timestep, cached ====
# Every block's AdaLN (6 x 4096 values for the block, 3 x 4096 for its audio branch) depends only on the timestep
# embedding, which takes two values per step (0 for condition frames, t for the rest). The official FP32 math runs once
# per new timestep and the result is cached on both GPUs (7 MB per timestep), so the 1.8 GB of AdaLN weights stay in RAM.
MOD_CACHE = collections.OrderedDict()


def modulation(t):
    """t: [B, T, C_t] FP32 timestep embeddings -> per device ([U, DEPTH, 9C] table, [B*T] row index)."""
    flat = t.reshape(-1, t.shape[-1]).float()
    uniq, inv = torch.unique(flat, dim=0, return_inverse=True)
    keys = [row.detach().cpu().numpy().tobytes() for row in uniq]
    missing = [i for i, k in enumerate(keys) if k not in MOD_CACHE]
    if missing:
        rows = F.silu(uniq[missing].to(MAIN_DEV, torch.float32))
        per_block = []
        for wb in ADALN:
            w, bias, wa, ba = (v.to(MAIN_DEV, torch.float32) for v in wb)
            per_block.append(torch.cat([F.linear(rows, w, bias), F.linear(rows, wa, ba)], dim=-1))
        table = torch.stack(per_block, dim=1)  # [missing, DEPTH, 9C]
        for j, i in enumerate(missing):
            MOD_CACHE[keys[i]] = [table[j].to(dev, copy=True) for dev in DEVICES]
        while len(MOD_CACHE) > 64:
            MOD_CACHE.popitem(last=False)
    tables = [torch.stack([MOD_CACHE[k][s] for k in keys]) for s in range(len(DEVICES))]
    return tables, [inv.to(dev) for dev in DEVICES]


# ==== One GPU's half of a block ====
ROPE = [RotaryPositionalEmbedding(HEAD_DIM, cp_split_hw=[1, 1]) for _ in DEVICES]
ROPE1D = [RotaryPositionalEmbedding1D(HEAD_DIM) for _ in DEVICES]


def _rms(x, w):
    """RMSNorm_FP32 of the official code, output in the compute dtype (normalized values are small)."""
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).to(w.dtype) * w


# ==== Fused attention-input and FFN-activation kernels ====
# q/k RMS norm + 3D RoPE + V's scaled FP16 cast in one pass per token (the official path makes ~15 full-size FP32
# passes over Q and K per block), written straight into the [batch, heads, tokens, dim] layout attention reads; and the
# FFN's silu(gate) * up formed directly as the scaled FP16 GEMM input. Checked against the PyTorch path at startup.
FUSED2 = {"on": False, "reason": ""}
_ROPE_TABLES = {}
if FUSED["on"]:
    try:
        @triton.jit
        def _qkv_prep_kernel(QKV, QW, KW, COS, SIN, VS, QO, KO, VO, n_tok, eps,
                             HH: tl.constexpr, D: tl.constexpr):
            n = tl.program_id(0).to(tl.int64)
            h = tl.arange(0, HH)[:, None]
            d = tl.arange(0, D)[None, :]
            dp = d ^ 1
            row = QKV + n * (3 * HH * D)
            cos = tl.load(COS + n * D + d)
            sin = tl.load(SIN + n * D + d)
            even = (d % 2) == 0
            out_off = h * (n_tok * D) + n * D + d
            for part in tl.static_range(2):
                base = row + part * HH * D
                x = tl.load(base + h * D + d).to(tl.float32)
                xp = tl.load(base + h * D + dp).to(tl.float32)
                r = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)[:, None]
                if part == 0:
                    w = tl.load(QW + d).to(tl.float32)
                    wp = tl.load(QW + dp).to(tl.float32)
                else:
                    w = tl.load(KW + d).to(tl.float32)
                    wp = tl.load(KW + dp).to(tl.float32)
                y = x * r * w
                yp = xp * r * wp
                rot = tl.where(even, -yp, yp)
                o = (y * cos + rot * sin).to(tl.float16)
                if part == 0:
                    tl.store(QO + out_off, o)
                else:
                    tl.store(KO + out_off, o)
            v = tl.load(row + 2 * HH * D + h * D + d).to(tl.float32) * tl.load(VS)
            v = tl.minimum(tl.maximum(v, -65504.0), 65504.0)
            tl.store(VO + out_off, v.to(tl.float16))

        @triton.jit
        def _swiglu_amax_kernel(H, AMAX, F: tl.constexpr, BLOCK: tl.constexpr):
            row = tl.program_id(0).to(tl.int64)
            cols = tl.arange(0, BLOCK)
            m = cols < F
            g = tl.load(H + row * 2 * F + cols, mask=m, other=0.0).to(tl.float32)
            u = tl.load(H + row * 2 * F + F + cols, mask=m, other=0.0).to(tl.float32)
            a = g / (1.0 + tl.exp(-g)) * u
            tl.store(AMAX + row, tl.max(tl.abs(a), axis=0))

        @triton.jit
        def _swiglu_fp16_kernel(H, OUT, S, F: tl.constexpr, BLOCK: tl.constexpr):
            row = tl.program_id(0).to(tl.int64)
            cols = tl.arange(0, BLOCK)
            m = cols < F
            g = tl.load(H + row * 2 * F + cols, mask=m, other=0.0).to(tl.float32)
            u = tl.load(H + row * 2 * F + F + cols, mask=m, other=0.0).to(tl.float32)
            a = g / (1.0 + tl.exp(-g)) * u * tl.load(S)
            a = tl.minimum(tl.maximum(a, -65504.0), 65504.0)
            tl.store(OUT + row * F + cols, a.to(tl.float16), mask=m)

        FUSED2["on"] = True
    except Exception as e:
        FUSED2["reason"] = _first_line(e)


def _pow2_scale(amax, target=1024.0):
    """(scale, inverse) powers of two putting amax at or below target; never scales up (GPU, no sync)."""
    exp = torch.ceil(torch.log2(amax.float().clamp_min(1e-30) / target)).clamp_min(0)
    return torch.exp2(-exp), torch.exp2(exp)


def rope_tables(s, shape, frame_index, num_ref):
    """cos / sin [tokens, head_dim] (FP32) of the official 3D RoPE for this grid, built once per grid and GPU."""
    rope = ROPE[s]
    key = '.'.join(str(i) for i in shape) + f"-{frame_index}-{num_ref}"
    dev = DEVICES[s]
    if (key, str(dev)) not in _ROPE_TABLES:
        if key not in rope.freqs_dict:
            rope.register_grid_size(shape, key, frame_index, num_ref)
        f = rope.freqs_dict[key].float().to(dev)
        _ROPE_TABLES[(key, str(dev))] = (f.cos().contiguous(), f.sin().contiguous())
    return _ROPE_TABLES[(key, str(dev))]


def qkv_prep(s, sh, qkv, ctx, fused=None):
    """qkv [B, N, 3*HH*D] (ACT) -> q, k (RMS norm + RoPE) and v (scaled) in FP16 [B, HH, N, D], and v's inverse scale."""
    bsz, n, _ = qkv.shape
    v_view = qkv.view(bsz, n, 3, HALF_HEADS, HEAD_DIM)[:, :, 2]
    v_scale, v_inv = _pow2_scale(v_view.abs().amax())
    if (FUSED2["on"] if fused is None else fused) and qkv.is_cuda and qkv.is_contiguous() and bsz == 1:
        cos, sin = rope_tables(s, ctx.shape, ctx.ref_img_index, ctx.num_ref_latents)
        q = torch.empty(bsz, HALF_HEADS, n, HEAD_DIM, device=qkv.device, dtype=COMPUTE)
        k, v = torch.empty_like(q), torch.empty_like(q)
        _qkv_prep_kernel[(n,)](qkv, sh.attn_q_norm, sh.attn_k_norm, cos, sin, v_scale.to(qkv.device), q, k, v, n, 1e-6,
                               HH=HALF_HEADS, D=HEAD_DIM, num_warps=4)
        return q, k, v, v_inv
    q, k, v = qkv.view(bsz, n, 3, HALF_HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4).unbind(0)
    q, k = _rms(q, sh.attn_q_norm), _rms(k, sh.attn_k_norm)
    q, k = ROPE[s](q, k, ctx.shape, ctx.ref_img_index, ctx.num_ref_latents)
    v = (v * v_scale.to(v.dtype)).to(COMPUTE)
    return q, k, v, v_inv


def swiglu_fp16(h, fused=None):
    """silu(gate) * up of the FFN ([..., gate | up] in ACT) as the scaled FP16 GEMM input: (act, inverse scale)."""
    half = h.shape[-1] // 2
    if (FUSED2["on"] if fused is None else fused) and h.is_cuda and h.is_contiguous():
        rows = h.numel() // h.shape[-1]
        amax = torch.empty(rows, device=h.device, dtype=torch.float32)
        block = triton.next_power_of_2(half)
        _swiglu_amax_kernel[(rows,)](h, amax, F=half, BLOCK=block, num_warps=8)
        scale, inv = _pow2_scale(amax.max())
        out = torch.empty(*h.shape[:-1], half, device=h.device, dtype=COMPUTE)
        _swiglu_fp16_kernel[(rows,)](h, out, scale, F=half, BLOCK=block, num_warps=8)
        return out, inv
    act = F.silu(h[..., :half]) * h[..., half:]
    return _to_fp16(act)


def probe_fused2():
    if not FUSED2["on"]:
        return
    try:
        class _Sh:
            pass
        class _Ctx:
            pass
        for s, dev in enumerate(DEVICES):
            with _on(dev):
                g = torch.Generator(device=dev).manual_seed(0)
                n_t, hh, ww = 4, 8, 12
                n = n_t * hh * ww
                sh = _Sh()
                sh.attn_q_norm = (torch.rand(HEAD_DIM, device=dev, generator=g) + 0.5).to(COMPUTE)
                sh.attn_k_norm = (torch.rand(HEAD_DIM, device=dev, generator=g) + 0.5).to(COMPUTE)
                qkv = (torch.randn(1, n, 3 * HALF_INNER, device=dev, generator=g) * 300).to(ACT)
                ctx = _Ctx(); ctx.shape = (n_t, hh, ww); ctx.ref_img_index = None; ctx.num_ref_latents = None
                a = qkv_prep(s, sh, qkv, ctx, fused=True)
                b = qkv_prep(s, sh, qkv, ctx, fused=False)
                for x, y in zip(a[:3], b[:3]):
                    diff = ((x.float() - y.float()).norm() / y.float().norm()).item()
                    if not math.isfinite(diff) or diff > 5e-3:
                        FUSED2.update(on=False, reason=f"qkv self-test differed {diff:.2e}")
                        return
                h = (torch.randn(1, 300, 2 * 768, device=dev, generator=g) * 50).to(ACT)
                (x1, i1), (x2, i2) = swiglu_fp16(h, fused=True), swiglu_fp16(h, fused=False)
                diff = ((x1.float() * i1 - x2.float() * i2).norm() / (x2.float() * i2).norm()).item()
                if not math.isfinite(diff) or diff > 5e-3:
                    FUSED2.update(on=False, reason=f"swiglu self-test differed {diff:.2e}")
                    return
    except Exception as e:
        FUSED2.update(on=False, reason=f"self-test failed: {_first_line(e)}")


probe_fused2()


def _attention_modes(q, k, v, ctx, block):
    """Official Attention.forward's condition handling (image-to-video, continuation with reference frame and the
    masked range, text-to-video), with the attention kernel of this run."""
    n_t = ctx.shape[0]
    n = q.shape[2]
    per = n // n_t
    ncl = ctx.num_cond_latents
    if ncl is not None and ncl == 1:
        x_cond = attention(q[:, :, :per].contiguous(), k[:, :, :per].contiguous(), v[:, :, :per].contiguous(), block)
        x_noise = attention(q[:, :, per:].contiguous(), k, v, block)
        return torch.cat([x_cond, x_noise], dim=2)
    if ncl is not None and ncl > 1:
        nref = ctx.num_ref_latents
        c = ncl * per
        x_ref = attention(q[:, :, :per].contiguous(), k[:, :, :per].contiguous(), v[:, :, :per].contiguous(), block)
        x_cond = attention(q[:, :, per:c].contiguous(), k[:, :, per:c].contiguous(), v[:, :, per:c].contiguous(), block)
        if ncl == n_t:
            return torch.cat([x_ref, x_cond], dim=2)
        q_noise = q[:, :, c:].contiguous()
        start, end, noisy = 0, 0, n_t - ncl
        mfr = ctx.mask_frame_range
        if mfr is not None and mfr > 0:
            start = ctx.ref_img_index - mfr - ncl + nref
            end = ctx.ref_img_index + mfr - ncl + nref + 1
        if start >= 0 and end > start and end <= noisy:
            a, z = start * per, end * per
            k_non, v_non = k[:, :, per:].contiguous(), v[:, :, per:].contiguous()
            x_noise = torch.cat([attention(q_noise[:, :, :a].contiguous(), k, v, block),
                                 attention(q_noise[:, :, a:z].contiguous(), k_non, v_non, block),
                                 attention(q_noise[:, :, z:].contiguous(), k, v, block)], dim=2)
        else:
            x_noise = attention(q_noise, k, v, block)
        return torch.cat([x_ref, x_cond, x_noise], dim=2)
    return attention(q, k, v, block)


def _ref_attn_map_part(q_noise, ref_k, masks):
    """This GPU's head group of the official get_attn_map_with_target (split_num=2: one group per GPU), by slices of
    tokens: softmax(q k^T / sqrt(d)) over the reference frame, averaged inside each speaker mask, mean over heads.
    q_noise [B, H, Nq, D], ref_k [B, H, S, D], masks [classes, S] -> [classes, Nq] (FP32)."""
    scale = 1.0 / q_noise.shape[-1] ** 0.5
    m = masks.float()
    msum = m.sum(-1)
    out = torch.empty(m.shape[0], q_noise.shape[2], device=q_noise.device, dtype=torch.float32)
    kt = ref_k.float().transpose(-2, -1)
    for a in range(0, q_noise.shape[2], 4096):
        attn = torch.softmax((q_noise[:, :, a:a + 4096].float() * scale) @ kt, dim=-1)  # [B, H, c, S]
        out[:, a:a + 4096] = ((attn @ m.T) / msum).mean(dim=1)[0].T
    return out


def _audio_cross(sh, s, x, cond, frames, x_ref_map, human_num):
    """Official SingleStreamAttention._process_cross_attn on this GPU's heads; returns the output projection's partial."""
    xr = x.reshape(x.shape[0] * frames, -1, x.shape[-1])  # "B (N_t S) C -> (B N_t) S C"
    bf, sx, _ = xr.shape
    q = run_linear(sh, "a_q", None, (xr, None))
    q = _rms(q.view(bf, sx, HALF_HEADS, HEAD_DIM).permute(0, 2, 1, 3), sh.audio_cross_attn_q_norm)
    rope_h1 = (0, DIT_CFG["class_interval"])
    rope_h2 = (DIT_CFG["class_range"] - DIT_CFG["class_interval"], DIT_CFG["class_range"])
    rope_bak = int(DIT_CFG["class_range"] // 2)
    if x_ref_map is not None:
        max_values = x_ref_map.max(1).values[:, None, None]
        min_values = x_ref_map.min(1).values[:, None, None]
        mm = torch.cat([max_values, min_values], dim=2)
        h1_max, h1_min = mm[0, :, 0].max(), mm[0, :, 1].min()
        h2_max, h2_min = mm[1, :, 0].max(), mm[1, :, 1].min()
        human1 = lc_attention.normalize_and_scale(x_ref_map[0], (h1_min, h1_max), rope_h1)
        human2 = lc_attention.normalize_and_scale(x_ref_map[1], (h2_min, h2_max), rope_h2)
        background_pos = rope_bak if x_ref_map.shape[0] <= 3 else 100
        back = torch.full((x_ref_map.size(1),), background_pos, dtype=human1.dtype, device=human1.device)
        max_indices = x_ref_map.argmax(dim=0).clamp(max=2)
        normalized_map = torch.stack([human1, human2, back], dim=1)
        normalized_pos = normalized_map[range(x_ref_map.size(1)), max_indices]
        q = q.reshape(x.shape[0], frames, HALF_HEADS, sx, HEAD_DIM).permute(0, 2, 1, 3, 4).reshape(x.shape[0], HALF_HEADS, frames * sx, HEAD_DIM)
        q = ROPE1D[s](q, normalized_pos)
        q = q.reshape(x.shape[0], HALF_HEADS, frames, sx, HEAD_DIM).permute(0, 2, 1, 3, 4).reshape(bf, HALF_HEADS, sx, HEAD_DIM)
    na = cond.shape[1]
    kv = run_linear(sh, "a_kv", cond).view(bf, na, 2, HALF_HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)
    k, v = kv.unbind(0)
    k = _rms(k, sh.audio_cross_attn_k_norm)
    if x_ref_map is not None:
        per_frame = torch.zeros(na, dtype=k.dtype, device=k.device)
        h1_pos = (rope_h1[0] + rope_h1[1]) / 2
        h2_pos = (rope_h2[0] + rope_h2[1]) / 2
        if human_num is not None and human_num > 2:
            background_pos = rope_bak if x_ref_map.shape[0] <= 3 else 100
            tph = per_frame.size(0) // human_num
            per_frame[:tph] = h1_pos
            per_frame[tph:2 * tph] = h2_pos
            per_frame[2 * tph:] = background_pos
        else:
            per_frame[:per_frame.size(0) // 2] = h1_pos
            per_frame[per_frame.size(0) // 2:] = h2_pos
        encoder_pos = torch.cat([per_frame] * frames, dim=0)
        k = k.reshape(x.shape[0], frames, HALF_HEADS, na, HEAD_DIM).permute(0, 2, 1, 3, 4).reshape(x.shape[0], HALF_HEADS, frames * na, HEAD_DIM)
        k = ROPE1D[s](k, encoder_pos)
        k = k.reshape(x.shape[0], HALF_HEADS, frames, na, HEAD_DIM).permute(0, 2, 1, 3, 4).reshape(bf, HALF_HEADS, na, HEAD_DIM)
    v, v_inv = _to_fp16(v)
    out = _sdpa(q, k, v.contiguous())
    out = out.transpose(1, 2).reshape(bf, sx, HALF_INNER)
    return out.reshape(x.shape[0], frames * sx, HALF_INNER), v_inv  # the output projection runs in _rowpar


def _text_cross(sh, x, y, y_seqlens):
    """Official MultiHeadCrossAttention._process_cross_attn on this GPU's heads (varlen over the batch)."""
    b, n, _ = x.shape
    q = run_linear(sh, "c_q", None, (x, None)).view(1, -1, HALF_HEADS, HEAD_DIM)
    kv = run_linear(sh, "c_kv", y).view(1, -1, 2, HALF_HEADS, HEAD_DIM)
    k, v = kv.unbind(2)
    q, k = _rms(q, sh.cross_attn_q_norm), _rms(k, sh.cross_attn_k_norm)
    v, v_inv = _to_fp16(v)
    outs, k0 = [], 0
    for i in range(b):
        qi = q[:, i * n:(i + 1) * n].transpose(1, 2)
        ki = k[:, k0:k0 + y_seqlens[i]].transpose(1, 2)
        vi = v[:, k0:k0 + y_seqlens[i]].transpose(1, 2)
        k0 += y_seqlens[i]
        outs.append(_sdpa(qi, ki, vi).transpose(1, 2))
    out = torch.cat(outs, dim=1).reshape(b, n, HALF_INNER)
    return out, v_inv  # the output projection runs in _rowpar


def _rowpar(sh, name, x16, inv):
    """A row-split output projection (its partial sum needs the other GPU's half): computed in row chunks, each handed
    over as soon as it exists, so copying chunk j to the other GPU overlaps computing chunk j + 1. Returns the total."""
    rows = x16.shape[1]
    k = max(1, min(EXCHANGE_CHUNKS, rows // CHUNK_MIN_ROWS))
    edges = [rows * i // k for i in range(k + 1)]
    for a, z in zip(edges[:-1], edges[1:]):
        yield ("chunk", run_linear(sh, name, None, (x16[:, a:z], inv)))
    total = yield ("sum", None)
    return total


EXCHANGE_CHUNKS, CHUNK_MIN_ROWS = 4, 8192


def _block_gen(s, b, x, ctx):
    """GPU s's half of block b (official LongCatAvatarSingleStreamBlock.forward). A generator: it yields each partial sum
    that needs the other GPU's half and receives the total. The residual stream x stays in FP32 on both GPUs."""
    sh = SHARDS[b][s]
    bsz, n, c = x.shape
    n_t = ctx.shape[0]
    per = n // n_t
    mods = ctx.mods[s][:, b].index_select(0, ctx.inv[s])  # [B*T, 9C]: per-frame AdaLN rows
    mod = [mods[:, i * c:(i + 1) * c] for i in range(9)]  # shift/scale/gate (attention), (FFN), (audio)

    # self-attention
    xm = ln_to_fp16(x, per, mod[0], mod[1])
    qkv = run_linear(sh, "qkv", None, (xm, None))
    del xm
    q, k, v, v_inv = qkv_prep(s, sh, qkv, ctx)
    out = _attention_modes(q, k, v, ctx, b)
    x_ref_part = None
    if ctx.masks is not None:
        c0 = (ctx.num_cond_latents or 0) * per
        x_ref_part = _ref_attn_map_part(q[:, :, c0:], k[:, :, :per], ctx.masks[s])
    del q, k, v, qkv
    out = out.transpose(1, 2).reshape(bsz, n, HALF_INNER)
    total = yield from _rowpar(sh, "proj", out, v_inv)
    del out
    gated_add_(x, total, per, mod[2])
    del total
    x_ref_map = None
    if x_ref_part is not None:
        x_ref_map = (yield ("small", x_ref_part)) / 2  # the official mean over the two head groups

    if not ctx.skip_crs_attn:
        ncl = ctx.num_cond_latents or 0
        start = ncl * per
        # text cross-attention (condition frames get none)
        xn = ln_to_fp16(x[:, start:], per, w=sh.pre_crs_attn_norm_w, b=sh.pre_crs_attn_norm_b)
        out, v_inv = _text_cross(sh, xn, ctx.y[s], ctx.y_seqlens)
        del xn
        total = yield from _rowpar(sh, "c_proj", out, v_inv)
        gated_add_(x[:, start:], total, per)
        del total, out
        # audio cross-attention (condition frames get none)
        xa = ln_to_fp16(x[:, start:], per, w=sh.pre_video_crs_attn_norm_w, b=sh.pre_video_crs_attn_norm_b)
        cond = ctx.audio[s]
        if ncl > 0:
            cond = cond.reshape(bsz, n_t, *cond.shape[1:])[:, ncl:].reshape(-1, *cond.shape[1:])
        human = ctx.human_num
        out, v_inv = _audio_cross(sh, s, xa, cond, n_t - ncl, x_ref_map if human is not None and human >= 2 else None, human)
        del xa
        total = yield from _rowpar(sh, "a_proj", out, v_inv)
        ln_mod_gated_add_(x[:, start:], total, per, mod[6][ncl:], mod[7][ncl:], mod[8][ncl:])
        del total, out

    # FFN
    xm = ln_to_fp16(x, per, mod[3], mod[4])
    h = run_linear(sh, "w13", None, (xm, None))
    del xm
    act, a_inv = swiglu_fp16(h)
    del h
    total = yield from _rowpar(sh, "w2", act, a_inv)
    del act
    gated_add_(x, total, per, mod[5])
    return x


# ==== Running the blocks: one CPU thread per GPU ====
# With one Python thread queuing both GPUs' kernels, anything that holds that thread up on one GPU also holds back the
# other GPU. Each GPU gets a thread of its own that queues its half of every block, and the two threads meet only where
# the math needs both halves (four partial-sum exchanges per block). Same kernels in the same order on each GPU as the
# one-thread path, so the result is bit-identical. The first transformer pass of the session runs on one thread (it
# compiles every kernel on both GPUs); the threads take over from the second pass.
TP_THREADS = {"on": True, "warm": False}
_TP_BARRIER = threading.Barrier(len(DEVICES), timeout=1800)
_TP_POOLS = []


def _tp_pools():
    if not _TP_POOLS:
        from concurrent.futures import ThreadPoolExecutor
        for dev in DEVICES:
            init = (lambda d=dev: torch.cuda.set_device(d)) if dev.type == "cuda" else None
            _TP_POOLS.append(ThreadPoolExecutor(1, thread_name_prefix=f"tp{len(_TP_POOLS)}", initializer=init))
    return _TP_POOLS


def _tp_job(fn, s, grad, inference, args):
    try:
        with torch.inference_mode(inference), torch.set_grad_enabled(grad), _on(DEVICES[s]):
            return fn(s, *args)
    except BaseException:
        _TP_BARRIER.abort()  # the other GPU's thread stops waiting at its next exchange
        raise


def _run_on_both(fn, args_per_gpu):
    if _TP_BARRIER.broken:
        _TP_BARRIER.reset()
    grad, inference = torch.is_grad_enabled(), torch.is_inference_mode_enabled()
    futures = [pool.submit(_tp_job, fn, s, grad, inference, args) for s, (pool, args) in enumerate(zip(_tp_pools(), args_per_gpu))]
    results, errors = [], []
    for future in futures:
        try:
            results.append(future.result())
        except BaseException as e:
            errors.append(e)
    if errors:  # the real error, not the other thread's broken barrier
        raise next((e for e in errors if not isinstance(e, threading.BrokenBarrierError)), errors[0])
    return results


_SIDE = {}


def _side(dev):
    """A copy stream per GPU: chunk copies run there, beside the compute stream."""
    if dev.type != "cuda":
        return None
    if dev not in _SIDE:
        _SIDE[dev] = torch.cuda.Stream(device=dev)
    return _SIDE[dev]


def _push_chunk(s, part, inbox, seq):
    """Copy this GPU's chunk to the other GPU on the copy streams of both, and post it (with its ready event) to the
    other GPU's inbox for exchange number seq."""
    src, dst = DEVICES[s], DEVICES[1 - s]
    if src.type == "cuda":
        src_side, dst_side = _side(src), _side(dst)
        src_side.wait_stream(torch.cuda.current_stream(src))
        with torch.cuda.stream(src_side), torch.cuda.stream(dst_side):
            buf = part.to(dst, non_blocking=True, copy=True)
        part.record_stream(src_side)
        ready = torch.cuda.Event()
        ready.record(dst_side)
    else:
        buf, ready = part.to(dst, copy=True), None
    inbox[1 - s].setdefault(seq, []).append((buf, ready))


def _collect(s, own, inbox, seq):
    """partial_0 + partial_1 on GPU s, chunk by chunk, once both GPUs have posted all their chunks."""
    _TP_BARRIER.wait()
    incoming = inbox[s].pop(seq)
    sums = []
    for part, (buf, ready) in zip(own, incoming):
        if ready is not None:
            cur = torch.cuda.current_stream(DEVICES[s])
            cur.wait_event(ready)
            buf.record_stream(cur)
        sums.append(part + buf)
    return sums[0] if len(sums) == 1 else torch.cat(sums, dim=1)


def _exchange_small(s, partial, shared):
    """The same sum for a small tensor (the two-speaker attention map), without chunks."""
    shared[s] = partial
    _TP_BARRIER.wait()
    incoming = shared[1 - s].to(DEVICES[s], non_blocking=True, copy=True)
    _TP_BARRIER.wait()
    shared[s] = None
    return partial + incoming


def _thread_blocks(s, x, ctx, shared):
    inbox, small, seq, own = shared["inbox"], shared["small"], 0, []
    for b in range(DEPTH):
        if STOP["requested"]:
            raise GenerationStopped()
        gen = _block_gen(s, b, x, ctx)
        msg = next(gen)
        while True:
            kind, value = msg
            reply = None
            if kind == "chunk":
                own.append(value)
                _push_chunk(s, value, inbox, seq)
            elif kind == "sum":
                reply = _collect(s, own, inbox, seq)
                own, seq = [], seq + 1
            else:
                reply = _exchange_small(s, value, small)
            try:
                msg = gen.send(reply)
            except StopIteration as done:
                x = done.value
                break
    return x


def _single_thread_blocks(xs, ctx):
    """The same block sequence from one thread (the session's first pass and the CPU tests): identical sums."""
    for b in range(DEPTH):
        if STOP["requested"]:
            raise GenerationStopped()
        gens = [_block_gen(s, b, xs[s], ctx) for s in range(len(DEVICES))]
        msgs = []
        for s, gen in enumerate(gens):
            with _on(DEVICES[s]):
                msgs.append(next(gen))
        while True:
            replies = [None, None]
            if msgs[0][0] == "chunk":
                own = [[], []]
                for s, gen in enumerate(gens):  # each GPU's chunks up to its "sum"
                    with _on(DEVICES[s]):
                        while msgs[s][0] == "chunk":
                            own[s].append(msgs[s][1])
                            msgs[s] = gen.send(None)
                for s, dev in enumerate(DEVICES):
                    with _on(dev):
                        sums = [p + q.to(dev, copy=True) for p, q in zip(own[s], own[1 - s])]
                        replies[s] = sums[0] if len(sums) == 1 else torch.cat(sums, dim=1)
            else:  # "small"
                for s, dev in enumerate(DEVICES):
                    with _on(dev):
                        replies[s] = msgs[s][1] + msgs[1 - s][1].to(dev, copy=True)
            nxt, done = [], []
            for s, gen in enumerate(gens):
                with _on(DEVICES[s]):
                    try:
                        nxt.append(gen.send(replies[s]))
                    except StopIteration as stop:
                        done.append(stop.value)
            if done:
                if len(done) != len(gens):
                    raise RuntimeError("The two halves of a block went out of step")
                xs = done
                break
            msgs = nxt
    return xs


class GenerationStopped(Exception):
    pass


STOP = {"requested": False}


class _Ctx:
    pass


def tp_forward(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask=None, num_cond_latents=0,
               return_kv=False, kv_cache_dict={}, skip_crs_attn=False, offload_kv_cache=False, audio_embs=None,
               num_ref_latents=None, ref_img_index=None, mask_frame_range=None, ref_target_masks=None):
    """The official LongCatVideoAvatarTransformer3DModel.forward with the block loop on the two GPUs."""
    if return_kv or kv_cache_dict:
        raise RuntimeError("KV caching is not used with the 1.6 LoRA mix")
    if hidden_states.shape[0] != 1:
        raise RuntimeError("The tensor-parallel DiT runs one video at a time (no CFG batch)")
    from einops import rearrange
    B, _, T, H, W = hidden_states.shape
    N_t, N_h, N_w = T // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2]
    if len(timestep.shape) == 1:
        timestep = timestep.unsqueeze(1).expand(-1, N_t)
    dtype = self.x_embedder.proj.weight.dtype
    hidden_states = hidden_states.to(MAIN_DEV, dtype)
    timestep = timestep.to(MAIN_DEV, dtype)
    encoder_hidden_states = encoder_hidden_states.to(MAIN_DEV, dtype)
    hidden_states = self.x_embedder(hidden_states)
    with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=MAIN_DEV.type == "cuda"):
        t = self.t_embedder(timestep.float().flatten(), dtype=torch.float32).reshape(B, N_t, -1)
    encoder_hidden_states = self.y_embedder(encoder_hidden_states)

    audio_cond = audio_embs.to(device=hidden_states.device, dtype=hidden_states.dtype)
    first_frame_audio_emb_s = audio_cond[:, :1, ...]
    latter_frame_audio_emb = audio_cond[:, 1:, ...]
    latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=self.vae_scale)
    middle_index = self.audio_window // 2
    latter_first = rearrange(latter_frame_audio_emb[:, :, :1, :middle_index + 1, ...], "b n_t n w s c -> b n_t (n w) s c")
    latter_last = rearrange(latter_frame_audio_emb[:, :, -1:, middle_index:, ...], "b n_t n w s c -> b n_t (n w) s c")
    latter_middle = rearrange(latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index + 1, ...], "b n_t n w s c -> b n_t (n w) s c")
    latter_frame_audio_emb_s = torch.concat([latter_first, latter_middle, latter_last], dim=2)
    audio_hidden_states = self.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s)
    if num_ref_latents is not None and num_ref_latents > 0:
        audio_hidden_states = torch.cat([audio_hidden_states[:, [0], :, :], audio_hidden_states], dim=1).contiguous()
    audio_hidden_states = audio_hidden_states[:, -N_t:]

    human_num = None
    if ref_target_masks is not None:
        human_num = len(audio_hidden_states)
        audio_hidden_states = torch.concat(audio_hidden_states.split(1), dim=2).squeeze(0)
    else:
        audio_hidden_states = rearrange(audio_hidden_states, "b t n c -> (b t) n c")

    token_ref_target_masks = None
    if ref_target_masks is not None:
        rtm = ref_target_masks.to(MAIN_DEV).unsqueeze(0).to(torch.float32)
        rtm = nn.functional.interpolate(rtm, size=(N_h, N_w), mode="nearest").squeeze(0)
        token_ref_target_masks = (rtm > 0).view(rtm.shape[0], -1).to(dtype)

    if self.text_tokens_zero_pad and encoder_attention_mask is not None:
        encoder_attention_mask = encoder_attention_mask.to(MAIN_DEV)
        encoder_hidden_states = encoder_hidden_states * encoder_attention_mask[:, None, :, None]
        encoder_attention_mask = (encoder_attention_mask * 0 + 1).to(encoder_attention_mask.dtype)
    if encoder_attention_mask is not None:
        encoder_attention_mask = encoder_attention_mask.squeeze(1).squeeze(1)
        encoder_hidden_states = encoder_hidden_states.squeeze(1).masked_select(
            encoder_attention_mask.unsqueeze(-1) != 0).view(1, -1, hidden_states.shape[-1])
        y_seqlens = encoder_attention_mask.sum(dim=1).tolist()
    else:
        y_seqlens = [encoder_hidden_states.shape[2]] * encoder_hidden_states.shape[0]
        encoder_hidden_states = encoder_hidden_states.squeeze(1).view(1, -1, hidden_states.shape[-1])

    ctx = _Ctx()
    ctx.shape = (N_t, N_h, N_w)
    ctx.num_cond_latents, ctx.num_ref_latents = num_cond_latents, num_ref_latents
    ctx.ref_img_index, ctx.mask_frame_range = ref_img_index, mask_frame_range
    ctx.skip_crs_attn, ctx.human_num, ctx.y_seqlens = skip_crs_attn, human_num, [int(v) for v in y_seqlens]
    ctx.y = [encoder_hidden_states.to(dev, ACT) for dev in DEVICES]
    ctx.audio = [audio_hidden_states.to(dev, ACT) for dev in DEVICES]
    ctx.masks = None if token_ref_target_masks is None else [token_ref_target_masks.to(dev) for dev in DEVICES]
    ctx.mods, ctx.inv = modulation(t)
    x = hidden_states.float().contiguous()  # the patch embedding returns a transposed view; the fused kernels need rows
    xs = [x.to(DEVICES[0], copy=True), x if MAIN_DEV != DEVICES[0] else x.clone()]
    del hidden_states
    if TP_THREADS["on"] and TP_THREADS["warm"]:
        shared = {"inbox": [{}, {}], "small": [None, None]}
        xs = _run_on_both(_thread_blocks, [(xs[s], ctx, shared) for s in range(len(DEVICES))])
    else:
        xs = _single_thread_blocks(xs, ctx)
        TP_THREADS["warm"] = True
    hidden_states = xs[1]
    del xs, ctx
    hidden_states = self.final_layer(hidden_states, t, (N_t, N_h, N_w))
    hidden_states = self.unpatchify(hidden_states, N_t, N_h, N_w)
    if not torch.isfinite(hidden_states).all():
        raise RuntimeError("the transformer produced non-finite values (numerical overflow)")
    return hidden_states.to(torch.float32)


dit.forward = types.MethodType(tp_forward, dit)
dit.lora_mix_config = {"mix_steps": MIX_STEPS, "mix_alpha": MIX_ALPHA}
STEP_HOOK = {"fn": None}


SAMPLER = {"keep": list(range(8))}  # which of the 8 official distilled steps run (Fast: every other one)


def _set_lora_mix_step(step_index):
    """Called by the official pipelines at the start of every denoising step."""
    if STOP["requested"]:
        raise GenerationStopped()
    set_lora_step(SAMPLER["keep"][step_index])  # the LoRA phase of the official step this one stands for
    if STEP_HOOK["fn"] is not None:
        STEP_HOOK["fn"](step_index)


dit.set_lora_mix_step = _set_lora_mix_step


# ==== UMT5-XXL text encoder: layer by layer from RAM, real tokens only ====
# The official pipeline runs UMT5-XXL in BF16 on 512 tokens (the prompt plus padding). The DiT zeroes the padding
# positions anyway, and with the attention mask the real tokens never see the padding, so only the real tokens are
# encoded and the rest is filled with zeros (same result). The 11 GB of BF16 weights (bit-identical to the official ones
# in BF16) stay in RAM; each layer is copied to GPU 1 in FP32 for its turn. Prompts are cached.
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel
from torch.func import functional_call

T5_MAP = {"norm1": "layer.0.layer_norm", "attn.q": "layer.0.SelfAttention.q", "attn.k": "layer.0.SelfAttention.k",
          "attn.v": "layer.0.SelfAttention.v", "attn.o": "layer.0.SelfAttention.o",
          "pos_embedding.embedding": "layer.0.SelfAttention.relative_attention_bias", "norm2": "layer.1.layer_norm",
          "ffn.gate.0": "layer.1.DenseReluDense.wi_0", "ffn.fc1": "layer.1.DenseReluDense.wi_1", "ffn.fc2": "layer.1.DenseReluDense.wo"}


class StreamedUMT5:
    def __init__(self, path, config_path, device):
        self.device = device
        with open(config_path) as fh:
            self.config = UMT5Config(**{k: v for k, v in json.load(fh).items() if not k.startswith("_")})
        with torch.device("meta"):
            model = UMT5EncoderModel(self.config)
        model.eval()  # no dropout
        self.blocks = model.encoder.block
        self.layers = [{} for _ in self.blocks]
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key == "token_embedding.weight":
                    self.embedding = f.get_tensor(key).to(torch.bfloat16)
                elif key == "norm.weight":
                    self.final_norm = f.get_tensor(key).to(torch.bfloat16)
                else:
                    _, index, rest = key.split(".", 2)
                    name = rest[:-len(".weight")]
                    self.layers[int(index)][T5_MAP[name] + ".weight"] = f.get_tensor(key).to(torch.bfloat16)
        self.eps = self.config.layer_norm_epsilon

    @torch.inference_mode()
    def __call__(self, ids):
        """ids: [1, n] real tokens -> [1, n, d_model] FP32 on the device."""
        h = F.embedding(ids.cpu(), self.embedding).to(self.device, torch.float32)
        with _on(self.device):
            for block, params in zip(self.blocks, self.layers):
                gpu = {k: v.to(self.device, torch.float32, non_blocking=True) for k, v in params.items()}
                h = functional_call(block, gpu, (h,))[0]
                del gpu
            var = h.pow(2).mean(-1, keepdim=True)
            h = h * torch.rsqrt(var + self.eps) * self.final_norm.to(self.device, torch.float32)
        return h


BAR.start("t5", "Loading: UMT5-XXL text encoder into RAM")
text_encoder = StreamedUMT5(T5_FILE, os.path.join(MODELS_DIR, "base", "text_encoder", "config.json"), MAIN_DEV)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
BAR.finish()
PROMPT_CACHE = collections.OrderedDict()
from longcat_video.pipeline_longcat_video_avatar import prompt_clean


def encode_prompt_cached(prompt, max_sequence_length=512):
    """(prompt_embeds [1, 1, 512, C] in the DiT's dtype, mask [1, 512]) as the official _get_t5_prompt_embeds."""
    text = prompt_clean(prompt)
    if text in PROMPT_CACHE:
        PROMPT_CACHE.move_to_end(text)
        embeds, mask = PROMPT_CACHE[text]
        return embeds.to(MAIN_DEV), mask.to(MAIN_DEV)
    inputs = tokenizer([text], padding="max_length", max_length=max_sequence_length, truncation=True,
                       add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
    ids, mask = inputs.input_ids, inputs.attention_mask
    n = int(mask[0].sum())
    real = text_encoder(ids[:, :n])
    embeds = torch.zeros(1, max_sequence_length, real.shape[-1], dtype=torch.float32, device=MAIN_DEV)
    embeds[:, :n] = real
    embeds = embeds.to(dit.dtype).view(1, 1, max_sequence_length, -1)
    PROMPT_CACHE[text] = (embeds.cpu(), mask)
    while len(PROMPT_CACHE) > 16:
        PROMPT_CACHE.popitem(last=False)
    return embeds, mask.to(MAIN_DEV)


def _pipe_encode_prompt(prompt, negative_prompt=None, do_classifier_free_guidance=True, num_videos_per_prompt=1,
                        max_sequence_length=512, device=None, dtype=None):
    if do_classifier_free_guidance:
        raise RuntimeError("The distilled sampler runs without CFG")
    embeds, mask = encode_prompt_cached(prompt if isinstance(prompt, str) else prompt[0], max_sequence_length)
    return embeds.to(dtype or embeds.dtype), mask, None, None


# ==== Whisper-large-v3 encoder (audio features), FP32 as in the official code ====
from transformers import AutoFeatureExtractor, WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder


class WhisperAudio:
    """The part of WhisperModel the official pipeline uses (.encoder, .dtype). The weights stay in RAM and are put on
    GPU 1 only while a voice is being encoded."""

    def __init__(self, folder, device):
        config = WhisperConfig.from_pretrained(folder)
        with torch.device("meta"):
            self.encoder = WhisperEncoder(config)
        state = {}
        with safe_open(os.path.join(folder, "model.safetensors"), framework="pt", device="cpu") as f:
            for key in f.keys():
                for prefix in ("model.encoder.", "encoder."):
                    if key.startswith(prefix):
                        state[key[len(prefix):]] = f.get_tensor(key).float()
        missing = self.encoder.load_state_dict(state, strict=False, assign=True).missing_keys
        if missing:
            raise RuntimeError(f"Whisper encoder weights missing: {missing[:3]}")
        self.encoder.eval().requires_grad_(False)
        for name, buf in list(self.encoder.named_buffers()):  # small: kept on the GPU
            owner = self.encoder.get_submodule(name.rsplit(".", 1)[0]) if "." in name else self.encoder
            owner._buffers[name.rsplit(".", 1)[-1]] = buf.to(device)
        self.cpu = {n: p.data for n, p in self.encoder.named_parameters()}
        self.device = device
        self.dtype = torch.float32

    def load(self):
        for n, p in self.encoder.named_parameters():
            p.data = self.cpu[n].to(self.device, non_blocking=True)

    def unload(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)  # the encoder's kernels may still be reading these weights
        for n, p in self.encoder.named_parameters():
            p.data = self.cpu[n]


BAR.start("whisper", "Loading: Whisper audio encoder")
whisper = WhisperAudio(WHISPER_DIR, MAIN_DEV)
feature_extractor = AutoFeatureExtractor.from_pretrained(WHISPER_DIR)
BAR.finish()

# ==== VAE: one copy per GPU; the decoder splits each frame between them ====
VAE_DTYPE = {"float16": torch.float16, "float32": torch.float32}[os.environ.get("LC_VAE_DTYPE", "float32" if TEST_MODE else "float16")]
BAR.start("vae", "Loading: VAE")
VAES = [AutoencoderKLWan.from_pretrained(VAE_DIR, torch_dtype=VAE_DTYPE).to(dev).eval().requires_grad_(False) for dev in DEVICES]
BAR.finish()
for _vae in VAES:  # the pipeline hands the VAE images in the DiT's dtype
    _vae.encode = (lambda orig: (lambda x, *a, **k: orig(x.to(VAE_DTYPE), *a, **k)))(_vae.encode)
VAE_STATE = {"parallel": True}


def _tile_bounds(rank, split_w, height, width, overlap):
    """The official _cp_tile_bounds for a 1 x 2 split."""
    core_w = width // split_w
    return 0, height, max(0, rank * core_w - overlap), min(width, (rank + 1) * core_w + overlap)


def _decode_tile(s, latents, bounds):
    top, bottom, left, right = bounds
    with torch.inference_mode():
        return VAES[s].decode(latents[..., top:bottom, left:right].to(DEVICES[s]).contiguous(), return_dict=False)[0].contiguous()


def decode_latents(latents):
    """Official LongCatVideoAvatarPipeline._decode_video_latents with a 1 x 2 split over the two GPUs (the same tiles,
    overlap and linear blend as its two-rank context-parallel path), each GPU decoding its half from its own thread."""
    BAR.start("decode", "Decoding video")
    try:
        height, width = latents.shape[-2:]
        if not VAE_STATE["parallel"] or width % 2:
            return VAES[1].decode(latents.to(MAIN_DEV), return_dict=False)[0]
        vae = VAES[1]
        scale = vae.spatial_compression_ratio
        overlap = max(1, (vae.tile_sample_min_height - vae.tile_sample_stride_height) // scale)
        bounds = [_tile_bounds(r, 2, height, width, overlap) for r in range(2)]
        tiles = _run_on_both(lambda s, b: _decode_tile(s, latents, b), [(bounds[0],), (bounds[1],)])
        tiles = [t.to(MAIN_DEV) for t in tiles]
        out_shape = (*tiles[0].shape[:3], height * scale, width * scale)
        output_sum = torch.zeros(out_shape, dtype=torch.float32, device=MAIN_DEV)
        weight_sum = torch.zeros((1, 1, 1, height * scale, width * scale), dtype=torch.float32, device=MAIN_DEV)
        overlap_px = overlap * scale
        for rank, tile in enumerate(tiles):
            _, _, left, right = bounds[rank]
            tile_w = tile.shape[-1]
            weight_w = torch.ones(tile_w, dtype=torch.float32, device=MAIN_DEV)
            blend_w = min(2 * overlap_px, tile_w)
            if rank > 0:
                weight_w[:blend_w] = torch.linspace(0, 1, blend_w, device=MAIN_DEV)
            if rank + 1 < 2:
                weight_w[-blend_w:] *= torch.linspace(1, 0, blend_w, device=MAIN_DEV)
            weight = torch.ones(tile.shape[-2], 1, device=MAIN_DEV) * weight_w[None, :]
            output_sum[..., left * scale:right * scale] += tile.float() * weight
            weight_sum[..., left * scale:right * scale] += weight
        return (output_sum / weight_sum.clamp_min_(1e-6)).to(tiles[0].dtype)
    finally:
        BAR.finish()


# ==== Voice isolation (the official Kim_Vocal_2 MDX-Net separator, CPU ONNX) ====
_SEPARATOR = {"model": None}


def separate_voice(path, duration):
    """The official separate_vocals: the voice track of an audio file at 16 kHz, cut to duration seconds."""
    if _SEPARATOR["model"] is None:
        import logging
        from audio_separator.separator import Separator
        sep = Separator(output_dir=os.path.join(OUTPUT_DIR, ".vocals"), output_single_stem="vocals",
                        model_file_dir=SEPARATOR_DIR, log_level=logging.ERROR)
        sep.load_model("Kim_Vocal_2.onnx")
        if MAIN_DEV.type == "cuda":
            # The same network on the GPU (onnx2torch, as audio-separator itself does for other segment sizes) instead of
            # ONNX Runtime on the CPU: identical chunks and segment size, ~10x faster.
            try:
                import onnx2torch
                net = onnx2torch.convert(os.path.join(SEPARATOR_DIR, "Kim_Vocal_2.onnx")).to(MAIN_DEV).eval()

                def _run(spek, net=net):
                    with torch.inference_mode():
                        return net(spek.to(MAIN_DEV))
                sep.model_instance.model_run = _run
            except Exception:
                pass
        _SEPARATOR["model"] = sep
    from longcat_video.audio_process.vocal_separator import separate_vocals
    return separate_vocals(_SEPARATOR["model"], path, duration=duration)


# ==== The official pipeline, wired to the parts above ====
class _TextEncoderInfo:
    config = types.SimpleNamespace(d_model=4096)
    dtype = torch.bfloat16


scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(CKPT_DIR, "scheduler"))
pipe = LongCatVideoAvatarPipeline(tokenizer=tokenizer, text_encoder=_TextEncoderInfo(), vae=VAES[1], scheduler=scheduler,
                                  dit=dit, audio_encoder=whisper, audio_feature_extractor=feature_extractor,
                                  model_type="avatar-v1.5")
pipe.device = MAIN_DEV
pipe.encode_prompt = _pipe_encode_prompt
pipe._decode_video_latents = decode_latents
gc.collect()


# ==== Generation: the official single / multi-person demos, as one function ====
FPS, NUM_FRAMES, COND_FRAMES = 25, 93, 13  # avatar 1.5 / 1.6 defaults
NEW_FRAMES = NUM_FRAMES - COND_FRAMES      # each continuation segment adds 80 frames (3.2 s)
STEPS = 8                                  # the distilled sampler (DMD 4 + OPSD 4)
# Steps of the official distilled schedule (8 steps: DMD LoRA for 1-4, OPSD for 5-8). Fast runs steps 1, 4 and 7, each
# with the LoRA of the official step it stands for.
QUALITY = {"⚡ Fast (3 steps)": [0, 3, 6], "💎 Best (8 steps)": list(range(8))}
QUALITY_DEFAULT = next(iter(QUALITY))
REF_IMG_INDEX, MASK_FRAME_RANGE = 10, 0    # the README's continuation settings for 1.6
AT2V_SIZE = (480, 832)                     # no picture: the official 480p text-to-video size
NEGATIVE_PROMPT = ("Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
                   "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
                   "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
                   "messy background, three legs, many people in the background, walking backwards")
PROGRESS = {"active": False, "fraction": 0.0, "text": "", "started": 0.0, "finished": 0.0, "video": None, "message": "", "final_n": 0}
STEP_SECONDS = {"fast": None, "exact": None}  # measured on this machine (updated after every segment)


def report(fraction, text):
    PROGRESS.update(active=True, fraction=float(fraction), text=text)


def segments_for(seconds):
    frames = max(1, math.ceil(seconds * FPS))
    return 1 + max(0, math.ceil((frames - NUM_FRAMES) / NEW_FRAMES))


def window_cost(frames, continuation):
    """Cost of one denoising step for a window, relative to the full 93-frame first window (linear layers grow with the
    tokens, attention with their square)."""
    r = ((frames - 1) // 4 + 1 + (1 if continuation else 0)) / 24
    return 0.45 * r + 0.55 * r * r


def plan_windows(seconds):
    """[(frames, continuation)] of the generation windows for a video of this length."""
    total = max(1, math.ceil(seconds * FPS))
    first = window_frames(total)
    windows, done = [(first, False)], first
    while done < total:
        n = window_frames(total - done, COND_FRAMES)
        windows.append((n, True))
        done += n - COND_FRAMES
    return windows


def estimate_minutes(seconds, attention, quality=None):
    """Whole-run estimate from this machine's measured step time (a fixed first guess before the first video)."""
    key = "fast" if attention == ATTENTION_FAST and ATTN["int8"] is not None else "exact"
    step = STEP_SECONDS[key] or (55.0 if key == "fast" else 90.0)
    steps = len(QUALITY.get(quality, QUALITY[QUALITY_DEFAULT]))
    total = sum(steps * step * window_cost(n, c) + 0.22 * n for n, c in plan_windows(seconds)) + 125
    return total / 60


_official_sigmas = pipe.get_timesteps_sigmas


def _sampled_sigmas(sampling_steps, use_distill=False):
    """The official distilled sigmas, restricted to the steps this run uses (all 8, or Fast's 3)."""
    full = _official_sigmas(8, use_distill=use_distill)
    return full[SAMPLER["keep"]] if use_distill else full


pipe.get_timesteps_sigmas = _sampled_sigmas


def load_audio(path, seconds):
    """Mono 16 kHz float audio (ffmpeg reads every format the browser can record or upload)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-t", f"{seconds:.3f}", "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def audio_seconds(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return len(load_audio(path, 3600)) / 16000


def prepare_voice(path, seconds, clean):
    """The voice the model listens to: the official vocal separation (music and noise removed) or the raw track."""
    if clean:
        voice, _ = separate_voice(path, seconds)
    else:
        voice = load_audio(path, seconds)
    if voice.size == 0 or float(np.abs(voice).max()) < 1e-4:
        raise ValueError("no voice found in that audio")
    return voice


def audio_prepare_multi(left, right, total_seconds, left_raw, right_raw, audio_type, sr=16000):
    """The official audio_prepare_multi: two voices in parallel ('para') or one after the other ('add'), padded."""
    if audio_type == "add":
        left_ext = np.concatenate([left, np.zeros_like(right)])
        right_ext = np.concatenate([np.zeros_like(left), right])
        merged = np.concatenate([left_raw, np.zeros_like(right_raw)]) + np.concatenate([np.zeros_like(left_raw), right_raw])
    else:
        n = min(len(left), len(right), len(left_raw), len(right_raw))
        left_ext, right_ext, merged = left[:n], right[:n], left_raw[:n] + right_raw[:n]
    pad = math.ceil((total_seconds - len(left_ext) / sr) * sr)
    if pad > 0:
        left_ext, right_ext, merged = (np.append(a, np.zeros(pad, dtype=np.float32)) for a in (left_ext, right_ext, merged))
    return left_ext, right_ext, merged


def speaker_masks(width, height, boxes):
    """The official multi-person masks: person 1, person 2, background, from boxes [y_min, x_min, y_max, x_max] in image
    pixels, or the left / right halves of the picture."""
    m1, m2 = torch.zeros(height, width), torch.zeros(height, width)
    if boxes:
        (ly0, lx0, ly1, lx1), (ry0, rx0, ry1, rx1) = boxes
    else:
        face_scale, half = 0.1, width // 2
        ly0, ly1 = int(height * face_scale), int(height * (1 - face_scale))
        ry0, ry1 = ly0, ly1
        lx0, lx1 = int(half * face_scale), int(half * (1 - face_scale))
        rx0, rx1 = int(half * face_scale + half), int(half * (1 - face_scale) + half)
    m1[ly0:ly1, lx0:lx1] = 1
    m2[ry0:ry1, rx0:rx1] = 1
    background = torch.where((m1 + m2) > 0, torch.tensor(0.0), torch.tensor(1.0))
    return torch.stack([m1, m2, background], dim=0)


def parse_boxes(text):
    """'y1,x1,y2,x2; y1,x1,y2,x2' -> two boxes, or None."""
    text = (text or "").strip()
    if not text:
        return None
    parts = [p for p in re.split(r"[;\n]+", text) if p.strip()]
    boxes = [[int(float(v)) for v in re.split(r"[,\s]+", p.strip()) if v] for p in parts]
    if len(boxes) != 2 or any(len(b) != 4 for b in boxes):
        raise ValueError("speaker boxes: write two boxes as y1,x1,y2,x2; y1,x1,y2,x2 (image pixels)")
    return boxes


def save_mp4(frames, path, audio_path):
    """uint8 [T, H, W, 3] frames + the soundtrack, cut to the video's length."""
    t, h, w, _ = frames.shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-t", f"{t / FPS:.3f}", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for a in range(0, t, 16):
            proc.stdin.write(np.ascontiguousarray(frames[a:a + 16]).tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {proc.returncode}")


def window_frames(new_frames, cond=0):
    """Frames of one generation window: cond + new, rounded up to the model's 4k+1 lengths, at most the official 93.
    The last window of a video only covers what is left of the audio instead of always rendering all 93 frames."""
    n = cond + max(1, new_frames)
    return min(NUM_FRAMES, max(17, (n - 1 + 3) // 4 * 4 + 1))


def audio_window(full_emb, start, frames=NUM_FRAMES):
    """The official 5-frame audio window around each video frame, from frame index start."""
    idx = torch.arange(start, start + frames).unsqueeze(1) + (torch.arange(5) - 2).unsqueeze(0)
    idx = idx.clamp(0, full_emb.shape[0] - 1)
    return full_emb[idx][None].to(MAIN_DEV)


@torch.inference_mode()
def generate(image, audio1, audio2, audio_type, prompt, max_seconds, seed, attention_choice, clean_voice, boxes_text,
             quality=None, out_path=None):
    """One talking video. Returns (mp4 path, status text)."""
    t_start = time.time()
    STOP["requested"] = False
    SAMPLER["keep"] = QUALITY.get(quality, QUALITY[QUALITY_DEFAULT])
    steps = len(SAMPLER["keep"])
    two = audio2 is not None
    boxes = parse_boxes(boxes_text) if two else None
    if two and image is None:
        raise ValueError("two-person videos need a picture of both people")
    raw_seconds = audio_seconds(audio1) if not two else (
        audio_seconds(audio1) + audio_seconds(audio2) if audio_type == "add" else min(audio_seconds(audio1), audio_seconds(audio2)))
    seconds = min(raw_seconds, float(max_seconds))
    n_seg = segments_for(seconds)
    gen_seconds = NUM_FRAMES / FPS + (n_seg - 1) * NEW_FRAMES / FPS
    process_seconds = gen_seconds + 1.0
    out_frames = max(1, min(math.ceil(seconds * FPS), NUM_FRAMES + (n_seg - 1) * NEW_FRAMES))

    # 1. voices -> audio features (whisper, FP32 on GPU 1)
    report(0.01, "🎙️ Preparing the voice" + ("s" if two else ""))
    BAR.start("voice", "Preparing the voice" + (" tracks" if two else ""))
    try:
        if two:
            limit = process_seconds if audio_type != "add" else process_seconds
            left = prepare_voice(audio1, limit, clean_voice)
            right = prepare_voice(audio2, limit, clean_voice)
            left, right, merged = audio_prepare_multi(left, right, process_seconds, load_audio(audio1, limit), load_audio(audio2, limit),
                                                      audio_type)
            sound_path = os.path.join(OUTPUT_DIR, ".voices_mix.wav")
            import soundfile as sf
            sf.write(sound_path, merged, 16000)
            voices = [left, right]
        else:
            voice = prepare_voice(audio1, process_seconds, clean_voice)
            pad = math.ceil((process_seconds - len(voice) / 16000) * 16000)
            if pad > 0:
                voice = np.append(voice, np.zeros(pad, dtype=np.float32))
            voices, sound_path = [voice], audio1
        whisper.load()
        try:
            embs = [pipe.get_audio_embedding(v, fps=FPS, device=MAIN_DEV, sample_rate=16000, model_type="avatar-v1.5",
                                             use_loudness_norm=False) for v in voices]  # 1.6: no loudness normalization
        finally:
            whisper.unload()
            _empty_cache()
        if any(torch.isnan(e).any() for e in embs):
            raise ValueError("broken audio features (NaN)")
    finally:
        BAR.finish()

    # 2. the prompt (cached)
    report(0.03, "📝 Encoding the prompt")
    BAR.start("text", "Encoding the prompt")
    try:
        encode_prompt_cached(prompt)
    finally:
        BAR.finish()

    generator = torch.Generator(device=MAIN_DEV).manual_seed(int(seed))
    fast = start_attention(attention_choice, DEPTH)
    total_steps = steps * n_seg
    step_times = []
    state = {"seg": 0, "t": time.time(), "cost": 1.0}

    def on_step(i):
        now = time.time()
        if i > 0:
            step_times.append((now - state["t"]) / state["cost"])  # as a full-window step
        state["t"] = now
        done = state["seg"] * steps + i
        BAR.update(done)
        report(0.05 + 0.85 * done / total_steps, f"🎬 Segment {state['seg'] + 1}/{n_seg} · step {i + 1}/{steps}")

    STEP_HOOK["fn"] = on_step
    common = dict(prompt=prompt, negative_prompt=NEGATIVE_PROMPT, num_inference_steps=steps,
                  use_distill=True, text_guidance_scale=1.0, audio_guidance_scale=1.0, generator=generator, output_type="both")
    masks = None
    BAR.start("denoise", "Generating", total_steps, "steps")
    try:
        n_frames = window_frames(out_frames)
        state["cost"] = window_cost(n_frames, False)
        audio = torch.cat([audio_window(e, 0, n_frames) for e in embs])
        if image is not None:
            img = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
            if two:
                masks = speaker_masks(img.width, img.height, boxes)
            output, latent = pipe.generate_ai2v(image=img, resolution="480p", audio_emb=audio, ref_target_masks=masks,
                                                num_frames=n_frames, **common)
        else:
            output, latent = pipe.generate_at2v(height=AT2V_SIZE[0], width=AT2V_SIZE[1], audio_emb=audio, num_frames=n_frames,
                                                **common)
        state["t"] = time.time()
        video = (output[0] * 255).astype(np.uint8)
        frames = [video]
        ref_latent = latent[:, :, :1].clone()
        height, width = video.shape[1:3]
        current = [Image.fromarray(f) for f in video]
        start, done_frames = 0, len(video)
        for seg in range(1, n_seg):
            state["seg"] = seg
            start += n_frames - COND_FRAMES
            n_frames = window_frames(out_frames - done_frames, COND_FRAMES)
            state["cost"] = window_cost(n_frames, True)
            audio = torch.cat([audio_window(e, start, n_frames) for e in embs])
            output, latent = pipe.generate_avc(video=current, video_latent=latent, height=height, width=width, num_frames=n_frames,
                                               num_cond_frames=COND_FRAMES, use_kv_cache=False, enhance_hf=False, audio_emb=audio,
                                               ref_latent=ref_latent, ref_img_index=REF_IMG_INDEX, mask_frame_range=MASK_FRAME_RANGE,
                                               ref_target_masks=masks, **common)
            state["t"] = time.time()
            video = (output[0] * 255).astype(np.uint8)
            frames.append(video[COND_FRAMES:])
            done_frames += len(video) - COND_FRAMES
            current = [Image.fromarray(f) for f in video]
    finally:
        STEP_HOOK["fn"] = None
        BAR.finish()
    if step_times:
        STEP_SECONDS["fast" if fast else "exact"] = float(np.median(step_times))

    report(0.95, "💾 Saving the video")
    all_frames = np.concatenate(frames)[:out_frames]
    out_path = out_path or os.path.join(OUTPUT_DIR, f"longcat_avatar_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    BAR.start("save", "Saving the video")
    try:
        save_mp4(all_frames, out_path, sound_path)
    finally:
        BAR.finish()
    total = time.time() - t_start
    lines = [f"✅ {len(all_frames) / FPS:.1f} s video ({len(all_frames)} frames, {width}x{height}, {n_seg} segment"
             f"{'s' if n_seg > 1 else ''}) in {_fmt_secs(total)}"]
    if step_times:
        lines.append(f"⏱️ {steps} steps · {np.median(step_times):.0f} s per full-length step · {ATTENTION_FAST if fast else ATTENTION_EXACT}")
    if fast and ATTN["fallback"] is not None:
        lines.append("⚠️ Fast attention was not accurate enough on this video, so it was finished with exact attention")
    if raw_seconds > seconds + 0.05:
        lines.append(f"ℹ️ The audio is {raw_seconds:.1f} s; the video stops at the {seconds:.0f} s limit")
    return out_path, "\n".join(lines)


# ==== Background job: generation never runs inside a Gradio event ====
import gradio as gr

JOB = {"id": 0, "done": 0, "running": False, "video": None, "status": "Ready."}
TYPE_CHOICES = {"🗣️ Together (both at once, e.g. a duet)": "para", "🔁 One after the other (person 1, then person 2)": "add"}


def _path(value):
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("path") or value.get("name")
    return value if isinstance(value, str) and value else getattr(value, "name", None)


def _job_worker(job_id, args):
    video, status = None, ""
    try:
        with _on(MAIN_DEV):
            video, status = generate(*args)
    except GenerationStopped:
        status = "⏹️ Stopped."
    except ValueError as e:
        status = f"❌ {e}"
    except torch.cuda.OutOfMemoryError:
        status = "❌ Out of GPU memory. Try a shorter video or Exact attention."
    except BaseException as e:
        traceback.print_exc()
        status = f"❌ Error: {_first_line(e)}"
    finally:
        STOP["requested"] = False
        _empty_cache()
    print(status.splitlines()[0] if status else "")
    JOB.update(running=False, done=job_id, video=video, status=status)
    PROGRESS.update(active=False, finished=time.time(), video=os.path.basename(video) if video else None,
                    message=status.splitlines()[0] if status else "", final_n=job_id)


def start_generation(image, audio1, audio2, audio_type, prompt, max_seconds, seed, attention, clean_voice, boxes, quality=None):
    if JOB["running"]:
        return "⏳ A video is already being generated. It appears here as soon as it is ready."
    image, audio1, audio2 = _path(image), _path(audio1), _path(audio2)
    if not audio1:
        return "❌ Add the voice (audio 1) first."
    prompt = (prompt or "").strip()
    if not prompt:
        return "❌ Write a short prompt describing the scene and the person."
    seed = int(seed) if seed is not None and int(seed) >= 0 else random.randint(0, 2**31 - 1)
    args = (image, audio1, audio2, TYPE_CHOICES.get(audio_type, "para"), prompt, float(max_seconds or 10), seed, attention,
            bool(clean_voice), boxes, quality or QUALITY_DEFAULT)
    seconds = min(float(max_seconds or 10), audio_seconds(audio1) if not audio2 else 999)
    n_seg = segments_for(seconds)
    print(f"🎬 {'2 people' if audio2 else '1 person'} | {'image' if image else 'no image'} | up to {seconds:.1f} s "
          f"({n_seg} segment{'s' if n_seg > 1 else ''}) | seed {seed} | est. ~{estimate_minutes(seconds, attention, quality):.0f}m")
    JOB.update(id=JOB["id"] + 1, running=True)
    PROGRESS.update(active=True, fraction=0.0, text="Starting...", started=time.time(), finished=0.0, video=None, message="")
    threading.Thread(target=_job_worker, args=(JOB["id"], args), daemon=True).start()
    return (f"🚀 Started (seed {seed}). Progress shows at the top of the page and the video appears below when it is ready. "
            "Reloading the page or a dropped connection does not stop the job.")


def deliver_result():
    if JOB["running"]:
        return gr.skip(), f"⏳ {PROGRESS['text'] or 'Working...'}"
    if not JOB["done"]:
        return gr.skip(), JOB["status"]
    return JOB["video"], JOB["status"]


def stop_generation():
    if not JOB["running"]:
        return "Nothing is running."
    STOP["requested"] = True
    return "⏹️ Stopping..."


def load_latest_video():
    videos = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")), key=os.path.getmtime)
    if not videos:
        return None, f"No videos in {OUTPUT_DIR} yet."
    return videos[-1], f"📁 Loaded the latest video: {videos[-1]}"


def plan_info(audio1, audio2, audio_type, max_seconds, attention, quality=None):
    audio1, audio2 = _path(audio1), _path(audio2)
    if not audio1:
        return "ℹ️ Add a voice to see the video length and the time it takes."
    try:
        a = audio_seconds(audio1)
        b = audio_seconds(audio2) if audio2 else 0.0
    except Exception as e:
        return f"❌ Unreadable audio: {_first_line(e)}"
    raw = a if not audio2 else (a + b if TYPE_CHOICES.get(audio_type) == "add" else min(a, b))
    seconds = min(raw, float(max_seconds or 10))
    n_seg = segments_for(seconds)
    cut = f" (audio {raw:.1f} s, cut at the limit)" if raw > seconds + 0.05 else ""
    return (f"ℹ️ Video: {seconds:.1f} s{cut} → {n_seg} segment{'s' if n_seg > 1 else ''}. "
            f"Estimated time ≈ {estimate_minutes(seconds, attention, quality):.0f} min.")


# The official examples (LongCat-Video's own assets and prompts, from the pinned code commit).
AVATAR_ASSETS = os.path.join(CODE_DIR, "assets", "avatar")


def official_examples():
    rows, labels = [], []
    try:
        cases = [("single_example_1.json", 3.7, "🎤 Singer on stage: 1 person, first 3.7 s"),
                 ("multi_example_1.json", 6.9, "🎶 Studio duet: 2 people singing together, first 6.9 s"),
                 ("multi_example_2.json", 13.3, "☕ Café chat: 2 people talking in turn, first 13.3 s")]
        for name, seconds, label in cases:
            with open(os.path.join(AVATAR_ASSETS, name), encoding="utf-8") as fh:
                data = json.load(fh)
            fix = lambda p: os.path.join(CODE_DIR, p) if p else None
            audio = data["cond_audio"]
            boxes = data.get("bbox")
            box_text = "; ".join(",".join(str(v) for v in boxes[k]) for k in ("person1", "person2")) if boxes else ""
            kind = next(k for k, v in TYPE_CHOICES.items() if v == data.get("audio_type", "para"))
            row = [fix(data["cond_image"]), fix(audio["person1"]), fix(audio.get("person2")), kind, data["prompt"], seconds, box_text]
            if all(os.path.exists(p) for p in (row[0], row[1]) if p) and (row[2] is None or os.path.exists(row[2])):
                rows.append(row)
                labels.append(f"{label} (~{estimate_minutes(seconds, ATTENTION_DEFAULT):.0f} min)")
    except Exception as e:
        print(f"⚠️ The official examples could not be loaded ({_first_line(e)}); the UI shows none")
    return rows, labels


# ==== Gradio UI (AIQUEST Academy branding) ====
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.gradio-container { max-width: 1000px !important; margin: auto !important; }
.brand-header { text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 28px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 10px 25px rgba(102,126,234,0.3); }
.brand-title { color: white; font-size: 2em; font-weight: 700; margin: 0 0 6px 0; }
.brand-subtitle { color: rgba(255,255,255,0.88); font-size: 1em; margin-bottom: 16px; }
.btn-row { display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; }
.social-btn { display: inline-flex; align-items: center; justify-content: center; min-width: 150px; padding: 10px 18px; border-radius: 10px; font-weight: 700; font-size: 13px; text-decoration: none; color: white; white-space: nowrap; }
.yt-btn  { background: #FF0000; box-shadow: 0 4px 12px rgba(255,0,0,0.3); }
.x-btn   { background: #000000; box-shadow: 0 4px 12px rgba(0,0,0,0.25); }
.sup-btn { background: linear-gradient(135deg,#f6d365,#fda085); box-shadow: 0 4px 12px rgba(253,160,133,0.35); }
button.primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
#stop-btn { background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
#clear-btn { background: linear-gradient(135deg, #6b7280 0%, #374151 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
#aiq-deliver { display: none !important; }
.status-line { text-align: center; color: #6b7280; font-size: 13px; margin: -8px 0 12px 0; }
.brand-footer { text-align: center; color: #6b7280; font-size: 12px; margin-top: 24px; }
"""

BRAND_HTML = """
<div class="brand-header">
  <div class="brand-title">🎙️ LongCat Video Avatar 1.6: Talking Video</div>
  <div class="brand-subtitle">Created by <strong>AIQuest Academy</strong> &nbsp;|&nbsp; Kaggle GPU T4 x2 Edition</div>
  <div class="btn-row">
    <a href="https://www.youtube.com/@aiquestacademy?sub_confirmation=1" target="_blank" class="social-btn yt-btn">▶ Subscribe</a>
    <a href="https://x.com/aiquestacademy" target="_blank" class="social-btn x-btn">𝕏 Follow on X</a>
    <a href="https://aiquest.site" target="_blank" class="social-btn sup-btn">❤️ Support My Work</a>
  </div>
</div>
"""
STATUS_HTML = (f'<div class="status-line">⚡ {INT8_LABEL} &nbsp;|&nbsp; {ATTENTION_LABEL} &nbsp;|&nbsp; LongCat-Video-Avatar 1.6 '
               f'tensor-parallel on 2 x T4 &nbsp;|&nbsp; 3 or 8 steps &nbsp;|&nbsp; 480p, 25 fps</div>')
FOOTER_HTML = ('<div class="brand-footer">⚡ Made with ❤️ by <strong>AIQUEST Academy</strong> · aiquest.site · © All rights reserved<br>'
               'Powered by <a href="https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.6" target="_blank">LongCat-Video-Avatar 1.6</a> '
               '(Meituan LongCat) · use voices and faces you have the rights to, and label results as AI-generated.</div>')
HOW_TO = ("**How it works:** add a picture of the person (a clear face, upper body works best) and their voice or song, describe "
          "the scene in one or two sentences, then press Generate. The video follows the audio's length up to the limit you set "
          "(3.7 s for the first segment, then 3.2 s per extra segment). **Two people:** add a second voice and pick whether they "
          "speak together or in turn; the picture should show person 1 on the left and person 2 on the right. No picture: the "
          "person is created from the prompt.")

PROGRESS_HEAD = """
<script>
(function () {
  // Progress overlay fed by plain HTTP polling: it keeps working when a tunnel drops Gradio's event stream. When a new
  // result is ready it clicks the hidden deliver button once, which fetches the video (no timer re-rendering the page).
  var box = null, fill = null, label = null, hideAt = 0, lastFinal = null;
  function build() {
    box = document.createElement("div");
    box.style.cssText = "position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:99999;width:min(560px,92vw);" +
      "background:#1f2937;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,.25);padding:10px 14px;display:none;" +
      "font-family:Inter,sans-serif;color:#fff;font-size:13px";
    label = document.createElement("div");
    label.style.cssText = "margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis";
    var track = document.createElement("div");
    track.style.cssText = "height:8px;background:rgba(255,255,255,.18);border-radius:6px;overflow:hidden";
    fill = document.createElement("div");
    fill.style.cssText = "height:100%;width:0;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);transition:width .6s ease";
    track.appendChild(fill); box.appendChild(label); box.appendChild(track); document.body.appendChild(box);
  }
  async function tick() {
    try {
      var r = await fetch("aiq_progress?t=" + Date.now(), {cache: "no-store"});
      var p = await r.json();
      if (!box && document.body) build();
      if (lastFinal === null) { lastFinal = p.final_n; }
      var deliver = document.getElementById("aiq-deliver");
      if (deliver && p.final_n && p.final_n !== lastFinal && !p.active) { lastFinal = p.final_n; deliver.click(); }
      if (box) {
        if (p.active) {
          box.style.display = "block"; hideAt = 0;
          fill.style.width = Math.round(p.fraction * 100) + "%";
          label.textContent = "⏳ " + p.text + " · " + Math.round(p.elapsed) + "s";
        } else if (box.style.display === "block") {
          if (!hideAt) {
            fill.style.width = "100%";
            label.textContent = (p.video ? "✅ Done in " : "⚠️ Finished after ") + Math.round(p.elapsed) + "s ";
            if (p.video) {
              var link = document.createElement("a");
              link.href = "aiq_video?t=" + Date.now(); link.target = "_blank";
              link.style.cssText = "color:#c7d2fe;font-weight:600";
              link.textContent = "· ▶ open / download the video";
              label.appendChild(link);
            } else if (p.message) { label.textContent += "· " + p.message; }
            hideAt = Date.now() + 60000;
          } else if (Date.now() > hideAt) { box.style.display = "none"; }
        }
      }
    } catch (e) {}
    setTimeout(tick, 1000);
  }
  tick();
})();
</script>
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, head=PROGRESS_HEAD, title="LongCat Video Avatar 1.6 | AIQUEST Academy") as demo:
    gr.HTML(BRAND_HTML)
    gr.HTML(STATUS_HTML)
    gr.Markdown(HOW_TO)
    with gr.Row():
        with gr.Column(scale=1):
            ui_image = gr.Image(label="🖼️ Picture of the person (optional)", type="filepath", height=300)
            ui_prompt = gr.Textbox(label="📝 Prompt", lines=4,
                                placeholder="A woman in a red jacket stands on a stage under warm lights, singing into a microphone.")
        with gr.Column(scale=1):
            ui_audio1 = gr.Audio(label="🎙️ Voice (person 1)", type="filepath", sources=["upload", "microphone"])
            with gr.Accordion("👥 Two people", open=False):
                ui_audio2 = gr.Audio(label="🎙️ Voice (person 2, on the right of the picture)", type="filepath", sources=["upload", "microphone"])
                ui_audio_type = gr.Radio(list(TYPE_CHOICES), value=next(iter(TYPE_CHOICES)), label="How they speak")
                ui_boxes = gr.Textbox(label="Speaker boxes (optional)", placeholder="y1,x1,y2,x2; y1,x1,y2,x2 in picture pixels",
                                   info="Empty: person 1 is the left half of the picture, person 2 the right half")
            ui_max_seconds = gr.Slider(3.7, 60, value=10, step=0.1, label="⏱️ Longest video (seconds)",
                                    info="The video follows the audio up to this length")
            ui_plan = gr.Markdown("ℹ️ Add a voice to see the video length and the time it takes.")
            with gr.Row():
                ui_seed = gr.Number(value=-1, precision=0, label="🎲 Seed (-1 = random)")
                ui_clean_voice = gr.Checkbox(value=True, label="🎚️ Isolate the voice",
                                          info="Removes music and noise before lip sync (the video keeps the original sound)")
            ui_quality = gr.Radio(list(QUALITY), value=QUALITY_DEFAULT, label="🎚️ Quality",
                                  info="Fast: 3 steps, about 2.5x quicker. Best: the official 8 steps")
            ui_attention = gr.Radio(ATTENTION_CHOICES, value=ATTENTION_DEFAULT, label="⚡ Attention",
                                 info=("Fast: Comfy Kitchen INT8 attention for the T4, about twice as fast. Exact: full precision"
                                       if ATTN["int8"] is not None else "Exact: full precision (Fast is not available on this GPU)"))
    with gr.Row():
        ui_gen_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
        ui_stop_btn = gr.Button("🛑 Stop", size="lg", elem_id="stop-btn")
        ui_clear_btn = gr.Button("🗑️ Clear", size="lg", elem_id="clear-btn")
    ui_video_out = gr.Video(label="🎥 Result")
    ui_latest_btn = gr.Button("📂 Load latest video (use if the UI lost its connection)")
    ui_status_out = gr.Textbox(label="ℹ️ Status", lines=4, interactive=False)
    ui_deliver_btn = gr.Button("deliver", elem_id="aiq-deliver")

    _rows, _labels = official_examples()
    if _rows:
        gr.Markdown("### 📚 Examples\nLongCat's own pictures, voices and prompts. Click one to fill everything above, then press "
                    "**🎬 Generate Video**.")
        gr.Examples(_rows, inputs=[ui_image, ui_audio1, ui_audio2, ui_audio_type, ui_prompt, ui_max_seconds, ui_boxes], example_labels=_labels, label=None)
    gr.HTML(FOOTER_HTML)

    for _comp in (ui_audio1, ui_audio2, ui_audio_type, ui_max_seconds, ui_attention, ui_quality):
        _comp.change(plan_info, [ui_audio1, ui_audio2, ui_audio_type, ui_max_seconds, ui_attention, ui_quality], ui_plan, queue=False)
    ui_gen_btn.click(start_generation, inputs=[ui_image, ui_audio1, ui_audio2, ui_audio_type, ui_prompt, ui_max_seconds, ui_seed, ui_attention, ui_clean_voice, ui_boxes,
                             ui_quality],
                  outputs=ui_status_out, queue=False)
    ui_deliver_btn.click(deliver_result, outputs=[ui_video_out, ui_status_out], queue=False, show_progress="hidden")
    ui_stop_btn.click(stop_generation, outputs=ui_status_out, queue=False)
    ui_latest_btn.click(load_latest_video, outputs=[ui_video_out, ui_status_out], queue=False)
    ui_clear_btn.click(lambda: (None, None, None, "", "Ready.", -1, ""), outputs=[ui_image, ui_audio1, ui_audio2, ui_prompt, ui_status_out, ui_seed, ui_boxes],
                    queue=False)


# ==== Cloudflare quick tunnel: a second public link in case the Gradio share link fails ====
def _dns_published(host):
    try:
        req = urllib.request.Request(f"https://cloudflare-dns.com/dns-query?name={host}&type=A", headers={"accept": "application/dns-json"})
        return bool(json.loads(urllib.request.urlopen(req, timeout=5).read()).get("Answer"))
    except Exception:
        return None


def start_cloudflare_tunnel(port):
    import stat
    found = {"url": None, "registered": False, "ready": False}
    binary = "/tmp/cloudflared"
    try:
        if not os.path.exists(binary):
            urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", binary)
            os.chmod(binary, os.stat(binary).st_mode | stat.S_IEXEC)
        tunnel = subprocess.Popen([binary, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate", "--protocol", "http2"],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        import atexit
        atexit.register(tunnel.kill)
    except Exception as e:
        print(f"⚠️ Cloudflare tunnel unavailable: {_first_line(e)}")
        return found

    def _watch():
        for line in tunnel.stdout:
            match = re.search(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com", line)
            if match and not found["url"]:
                found["url"] = match.group(0)
            if "Registered tunnel connection" in line:
                found["registered"] = True

    def _wait_ready():
        deadline, registered_at = time.time() + 300, None
        while time.time() < deadline:
            if found["registered"] and registered_at is None:
                registered_at = time.time()
            published = _dns_published(found["url"].split("//", 1)[1]) if found["url"] and registered_at else False
            if published or (published is None and time.time() - registered_at > 15):
                found["ready"] = True
                if found.get("late"):
                    print(f"🌐 Cloudflare: {found['url']}")
                    sys.stdout.flush()
                return
            time.sleep(3)
        print("⚠️ The Cloudflare link did not come up within 5 minutes; use the Gradio share link.")

    threading.Thread(target=_watch, daemon=True).start()
    threading.Thread(target=_wait_ready, daemon=True).start()
    return found


def launch_ui(block=True):
    """Starts the server, the progress routes and the tunnel; prints the links."""
    cf = start_cloudflare_tunnel(SERVER_PORT) if not TEST_MODE else {"ready": False, "url": None}
    demo.queue(max_size=20)
    demo.launch(server_name="127.0.0.1", server_port=SERVER_PORT, share=not TEST_MODE, inline=False, show_error=True,
                prevent_thread_lock=True, quiet=True, ssr_mode=False, max_threads=8, allowed_paths=[OUTPUT_DIR, CODE_DIR])
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.routing import APIRoute

    def _progress_route():
        end = PROGRESS["finished"] or time.time()
        elapsed = end - PROGRESS["started"] if PROGRESS["started"] else 0.0
        return JSONResponse({**PROGRESS, "elapsed": elapsed}, headers={"Cache-Control": "no-store"})

    def _video_route():
        videos = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")), key=os.path.getmtime)
        if not videos:
            return JSONResponse({"error": "no video yet"}, status_code=404)
        name = os.path.basename(videos[-1])
        return FileResponse(videos[-1], media_type="video/mp4", headers={"Content-Disposition": f'inline; filename="{name}"',
                                                                          "Cache-Control": "no-store"})

    for _path_name, _route in (("/aiq_progress", _progress_route), ("/aiq_video", _video_route)):
        demo.app.router.routes.insert(0, APIRoute(_path_name, _route, methods=["GET"]))
    for _ in range(60):
        if cf["ready"]:
            break
        time.sleep(0.5)
    _frees = [torch.cuda.mem_get_info(d)[0] if d.type == "cuda" else 0 for d in DEVICES]
    print(f"✅ Ready in {time.time() - _t_load:.0f}s | GPU free {_gb(_frees[0])} / {_gb(_frees[1])} | RAM {_ram_in_use()}")
    if cf["ready"]:
        print(f"🌐 Cloudflare: {cf['url']}")
    else:
        cf["late"] = True
        print("🌐 Cloudflare: warming up, the link appears here in a minute (Gradio works now)")
    print(f"🔗 Gradio: {getattr(demo, 'share_url', None) or 'unavailable, use the Cloudflare link'}")
    print(f"💾 Videos: {OUTPUT_DIR} (latest also at <link>/aiq_video)")
    sys.stdout.flush()
    while block:  # keep the server process alive; generation logs stream to the notebook cell
        time.sleep(3600)


if os.environ.get("LC_NO_LAUNCH") != "1":
    launch_ui()
'''


with open("run_longcat.py", "w", encoding="utf-8") as f:
    f.write(RUN_SCRIPT_CODE)

import json
if os.path.exists("longcat_settings.json"):  # Step 1's choices, for a Step 2 re-run after a kernel restart
    with open("longcat_settings.json") as f:
        for key, value in json.load(f).items():
            os.environ.setdefault(key, value)

# Minimal logs: only the engine's own status lines (they start with these markers) and its live progress line reach
# this cell. Library chatter is dropped; Python tracebacks and crash reports are always shown in full.
MARKERS = ("✅", "❌", "⚠️", "🎬", "🌐", "🔗", "💾", "⏹️", "🛑", "🧹", "⚙️", "⏳", "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
CRASH_START = ("Traceback (most recent call last)", "Fatal Python error", "Thread 0x", "Current thread 0x")
_in_crash = [False]


def _keep(part):
    text = part.strip()
    if not text:  # carriage returns / spaces that clear the live line
        return True
    if text.startswith(CRASH_START):
        _in_crash[0] = True
        return True
    if _in_crash[0]:
        if part[:1] in (" ", "\t") or text.startswith(CRASH_START):
            return True
        _in_crash[0] = False  # the exception line closes the traceback
        return True
    return text.startswith(MARKERS)


env = dict(os.environ, PYTHONWARNINGS="ignore", TQDM_DISABLE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
           TF_CPP_MIN_LOG_LEVEL="3", USE_TF="0", USE_FLAX="0", GRADIO_ANALYTICS_ENABLED="False")

print("🚀 Starting LongCat Video Avatar 1.6...")
import psutil


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

# Step 1's downloads leave freed buffers in this kernel process: hand them back to the OS before the engine starts.
import ctypes
import gc
gc.collect()
try:
    ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception:
    pass
_kernel_gb = psutil.Process().memory_info().rss / 1024**3
if _kernel_gb > 2.0:
    print(f"⚠️ This notebook kernel holds {_kernel_gb:.1f} GB of RAM. If the engine runs out of memory: Run → Restart, then run Step 2 only.")


def _die_with_kernel():
    # Linux: the engine gets SIGKILL when this notebook kernel dies or restarts (no orphan holding the GPUs).
    try:
        import signal
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        pass


proc = subprocess.Popen([sys.executable, "-u", "run_longcat.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                        preexec_fn=_die_with_kernel)
import threading
import time
_peak = {"ram": 0.0, "engine": 0.0}


def _watch_ram():  # peak RAM while the engine runs, reported if it dies
    engine = psutil.Process(proc.pid)
    while proc.poll() is None:
        try:
            vm = psutil.virtual_memory()
            _peak["ram"] = max(_peak["ram"], (vm.total - vm.available) / 1024**3)
            _peak["engine"] = max(_peak["engine"], engine.memory_info().rss / 1024**3)
        except Exception:
            pass
        time.sleep(0.5)


threading.Thread(target=_watch_ram, daemon=True).start()
# Read raw bytes so carriage returns survive: the engine redraws its live progress line in place, and a text-mode
# pipe would turn every redraw into a new line.
decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
pending = ""
try:
    while True:
        chunk = os.read(proc.stdout.fileno(), 65536)
        if not chunk:
            break
        pending += decoder.decode(chunk)
        parts = re.split(r"(?<=[\r\n])", pending)
        pending = parts.pop()  # text after the last line break / carriage return
        out = "".join(part for part in parts if _keep(part))
        if pending and pending.strip().startswith(MARKERS):
            out, pending = out + pending, ""  # show the live progress line right away
        if out:
            sys.stdout.write(out)
            sys.stdout.flush()
    sys.stdout.write(pending)
    rc = proc.wait()
    if rc != 0:
        import signal
        how = f"killed by {signal.Signals(-rc).name}" if rc < 0 else f"exit code {rc}"
        hint = (" - the system killed it, which almost always means it ran out of RAM" if rc == -9 else
                " - a crash inside a native library (see the traceback above)" if rc in (-11, -6) else "")
        print(f"\n❌ The engine stopped ({how}){hint}.\n   Peak RAM while it ran: {_peak['ram']:.1f} GB in use of "
              f"{psutil.virtual_memory().total / 1024**3:.1f} GB (engine process {_peak['engine']:.1f} GB).")
except KeyboardInterrupt:
    proc.kill()  # frees the GPUs and RAM right away
    proc.wait()
    print("🛑 Engine stopped. Re-run this cell to start it again.")
