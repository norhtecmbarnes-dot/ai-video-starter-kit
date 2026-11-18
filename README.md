# AI Video Starter Kit — make AI videos on a budget

This is the download kit that goes with the video **"How to make AI videos on a budget"**.
Everything here is what was used to produce the Elizabeth spelling-song video on a modest
RTX 4080 — no cloud rentals, no paid subscriptions, and a free browser-only route for
people without a graphics card.

## Contents

```
workflows/
  video_minimax_h3_i2v.json   — image + text → video (the one used to animate each scene)
  video_minimax_h3_t2v.json   — text → video
  video_minimax_h3_r2v.json   — reference image + text → video (keeps a character consistent)
kaggle/
  step1_setup_download.py     — Kaggle notebook cell 1: installs LongCat-Video-Avatar and downloads the model
  step2_launch_app.py         — Kaggle notebook cell 2: launches the avatar web app (public link)
```

## Route 1 — Your own GPU (free, unlimited, most control)

**What you need:** any RTX GPU with ~16 GB VRAM (a modest RTX 4080 is plenty). Bigger cards
just render faster; nothing else changes.

1. Install [ComfyUI (Windows portable)](https://www.comfy.org/download) — unzip and run.
2. Download these model files into your ComfyUI `models` folders:

| File | Folder | Link |
|---|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models` | [Hugging Face](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders` | [Hugging Face](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) |
| `minimax_h3_video_vae_fp16.safetensors` | `models/vae` | [Hugging Face](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors) |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae` | [Hugging Face](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) |

3. In ComfyUI, open the workflow JSON you downloaded from this repo
   (Workflows menu → Open), drop in your character image, type your scene prompt, and hit Run.
4. Expect roughly **1 minute of GPU time per second of video** on a 4080.

Tips learned the hard way:
- If output video looks stretched, check your input stills first — squashed sources
  propagate into every frame. Fix the stills, regenerate.
- Generate one scene at a time, not a whole song at once.

## Route 2 — Free Kaggle notebook (no GPU needed)

Kaggle gives ~30 free GPU hours per week (two T4s). Zero cost, zero install.

1. Create a free Kaggle account (needs phone verification to enable Internet).
2. New Notebook → Settings → Accelerator: **GPU T4 ×2**, Internet: **On**.
3. Paste [`kaggle/step1_setup_download.py`](kaggle/step1_setup_download.py) into cell 1 and run
   (downloads ~37 GB of weights, 4–8 minutes).
4. Paste [`kaggle/step2_launch_app.py`](kaggle/step2_launch_app.py) into cell 2 and run — it prints a
   public web-app link.
5. Upload one photo + any audio in the app; it renders a lip-synced talking/singing video.
   Expect about **5 minutes per 3.7 seconds** of footage on the free T4s.

## Route 3 — Free browser tools (no install at all)

- **Meta.ai** — quick free AI clips straight in the browser. Shortest clips, least control.
- **Gemini** — free lyrics, scripts and prompts.
- **Voice-clone tools** (e.g. CloneVoice, Vois.ai) — free/paid narration in your own voice.

## Route 4 — Paid plans

Services like **Video Express** generate finished videos (and consistent characters) in the
browser on a monthly subscription — simplest option, least control.

> The honest trade: free costs you time and tinkering; paid costs you money and control.

## The pipeline used in the video

1. Write the song/lyrics (free AI or your own).
2. Make the vocals (clone your voice, or an AI voice).
3. Create a character sheet from your description (any image tool; pick a style — Ghibli, Pixar, Disney…).
4. Generate each scene from the character sheet (workflows here).
5. Animate: image → video with MiniMax H3 on your GPU or a Kaggle T4.
6. Assemble in a free editor, export 1080p, and you have your video.
