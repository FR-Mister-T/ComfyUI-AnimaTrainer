# Anima LoRA Training — Cheat Sheet

> **Who this is for:** ComfyUI users who know the interface but are new to LoRA training.
> The "why" column explains the constraint so you can judge edge cases yourself.

---

## Before You Start

### Required model files

| File | Folder in ComfyUI | Why |
|---|---|---|
| DiT checkpoint | `models/diffusion_models/` | The Anima model weights being trained. `animaOfficial_preview3Base.safetensors` (~4.2 GB) |
| Text encoder | `models/text_encoders/` | Qwen3 0.6B — converts your captions to embeddings. `qwen_3_06b_base.safetensors` |
| VAE | `models/vae/` | Encodes images to latent space before training. `qwen_image_vae.safetensors` |

### Dataset checklist

- [ ] One `.txt` caption file per image, same base name, same folder
- [ ] Minimum 20 images; 50–200 for reliable results — fewer images → higher overfitting risk
- [ ] Images at least 512×512; 1024×1024 or larger recommended for style or character LoRAs
- [ ] At least 5 GB free disk for latent cache, text encoder cache, and checkpoints

### Settings to verify before every run

| Setting | Required value | Why it matters |
|---|---|---|
| `learning_rate` | **1.0** | Prodigy ignores this field and adapts its own LR automatically. A small value (1e-4) causes d_hat overflow and instant NaN loss. |
| `cache_latents` | **disk** | Pre-encodes all images with the VAE once. Skips the VAE forward pass every step — saves both time and VRAM. |
| `cache_text_encoder_outputs` | **disk** | Pre-encodes all captions once. Keeps Qwen3 (~1.5 GB) unloaded from VRAM for the entire training run. |
| `attention_mode` | **torch** | torch = SDPA, works on all platforms. Other modes require extra libraries or are untested on new GPU architectures. |
| `gradient_checkpointing` | not **disabled** (unless very high VRAM) | Recomputes activations on the backward pass instead of storing them — cuts peak VRAM by ~50% at a ~20% speed cost. Required at 1024 resolution with batch > 1 on cards under 32 GB. |
| `save_state` on every AnimaTrainLoRASave | **True** | Saves optimizer state alongside the LoRA weights. Without it, you cannot resume a run — you start over from scratch if anything interrupts training. |

### AMD / ROCm users — additional checks

| Check | Why |
|---|---|
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0` in `start_comfy.sh` | AOTriton kernels are JIT-compiled lazily on first use. On new architectures (gfx1201 / RDNA 4) they trigger a GPU page fault that kills the display GPU and crashes the system around step 100–123. |
| `HSA_ENABLE_SDMA=0` in `start_comfy.sh` | Standard ROCm stability workaround — prevents certain DMA-related hangs. |
| `blocks_to_swap = 0` always | Block swapping uses a background thread that accesses GPU memory. On ROCm this causes page faults. The feature is also useless for Anima 2B on 32 GB — the model fits entirely in VRAM, so there is nothing to gain. |

---

## VRAM Tiers

> **12 GB cards (RTX 3060, RTX 4070, RX 7700 XT):** sit between Low and Medium.
> Start with the Low tier, then relax `blocks_to_swap` to 8 and try `batch_size = 2` if stable.

---

### Low — under 12 GB

| Setting | Value | Why |
|---|---|---|
| `blocks_to_swap` | 16–20 | Offloads that many DiT blocks to CPU RAM. Each swapped block reduces peak VRAM but adds CPU↔GPU transfer time per step. |
| `gradient_checkpointing` | `enabled_with_cpu_offloading` | Recomputed activations are stored in CPU RAM instead of GPU VRAM. More savings than standard GC, but ~2× slower per step. |
| `cache_latents` | `disk` | Essential — skips VAE on every step, saves ~1–2 GB. |
| `cache_text_encoder_outputs` | `disk` | Unloads Qwen3 from VRAM after first pass, freeing ~1.5 GB for training. |
| `highvram` | `False` | Keeps all memory-saving measures active. |
| `batch_size` | 1 | Only one image processed per step. Gradient quality is lower but VRAM is minimal. |
| `resolution` | 512–768 | Lower resolution = smaller activations = much less VRAM. 512 uses roughly 4× less activation memory than 1024. |
| `network_dim` | 8–16 | LoRA rank — capacity of the adapter. Small rank is sufficient for style; larger datasets can use 32 if VRAM allows. |
| `network_alpha` | dim / 2 | Sets the scale applied to the LoRA output: `effective_scale = alpha / dim`. **dim/2** (e.g. 24 with dim=48) → scale=0.5, the classic sd-scripts convention — mild regularization, LoRA can't overshoot base weights aggressively early on. **alpha=1.0** is equally valid with Prodigy: Prodigy adapts its LR per-parameter and detects the smaller gradient signal, ramping up its estimate to compensate within ~200 steps. With a fixed-LR optimizer (AdamW), alpha=1.0 would be dangerously low — the scale directly multiplies your LR with no compensation. |
| `gradient_dtype` | `fp32` | Most stable. bf16 gradients can accumulate errors at low batch sizes. |
| `save_dtype` | `bf16` | Halves checkpoint file size vs fp32 with negligible quality loss. |
| `max_train_steps` | 1000–1500 | At batch 1 each step sees one image. Scale up if your dataset is large. |
| `prodigy_steps` | ~20 % of `max_train_steps` | Prodigy adapts its LR for this many steps then freezes into ScheduleFree. Too short = LR never settles. Too long = no training time left after adaptation. |

---

### Medium — 16–24 GB

| Setting | 16 GB | 24 GB | Why |
|---|---|---|---|
| `blocks_to_swap` | 8 | 0 | 16 GB needs some block offloading at 1024. 24 GB fits the full 2B model with GC enabled. |
| `gradient_checkpointing` | `enabled` | `enabled` | Standard GC — recomputes activations, no CPU transfer, ~20% speed cost, ~50% VRAM saving. |
| `cache_latents` | `disk` | `disk` | Always recommended. |
| `cache_text_encoder_outputs` | `disk` | `disk` | Always recommended. |
| `highvram` | `False` | `False` | Keep memory-saving measures on. |
| `batch_size` | 2 | 4 | Higher batch = better gradient estimates per step = fewer total steps needed. |
| `resolution` | 1024 | 1024 | Standard training resolution for illustration and character work. |
| `network_dim` | 16–32 | 16–32 | |
| `network_alpha` | dim / 2 | dim / 2 | `effective_scale = alpha / dim`. dim/2 → scale=0.5, classic convention. alpha=1.0 also valid with Prodigy — Prodigy's per-parameter LR adapts to compensate for the smaller scale. With fixed-LR optimizers, alpha=1.0 would be too low. |
| `gradient_dtype` | `fp32` | `fp32` | |
| `save_dtype` | `bf16` | `bf16` | |
| `max_train_steps` | 2000 | 1000 | Scale inversely with batch (see Step Scaling below). Values assume ~100-image dataset. |
| `prodigy_steps` | ~20 % | ~20 % | |

---

### High — 32 GB+ local

Covers cards like the AMD R9700 (32 GB) or high-end NVIDIA cards. CUDA and ROCm both apply — AMD users must still follow the ROCm checklist above.

| Setting | Value | Why |
|---|---|---|
| `blocks_to_swap` | 0 | No swap needed — the 2B model fits in VRAM with room to spare. |
| `gradient_checkpointing` | `enabled_with_unsloth_offloading` | Unsloth offloading stores recomputed activations in CPU RAM using non-blocking transfers on the main thread — safe on ROCm, no background thread risk. Enables batch 10 at ~29 GB peak VRAM on 32 GB hardware. |
| `cache_latents` | `disk` | Still recommended — avoids re-running the VAE every step even with abundant VRAM. |
| `cache_text_encoder_outputs` | `disk` | Still recommended — no need to keep Qwen3 in VRAM during training. |
| `highvram` | `False` | VRAM swings ±6 GB during a run (optimizer init, Prodigy peak-LR phase). Keeping safety measures on prevents surprises. |
| `batch_size` | 6–10 | High batch = smoother loss curve, better Prodigy LR adaptation, fewer total steps needed. |
| `resolution` | 1024–1280 | 1280 with portrait-heavy datasets actually uses less VRAM than 1024 square (dominant bucket is 512×1280 — ~37% fewer pixels). |
| `network_dim` | 32–64 | More capacity for complex concepts or large datasets. |
| `network_alpha` | dim / 2 | `effective_scale = alpha / dim`. dim/2 → scale=0.5, classic convention. alpha=1.0 also valid with Prodigy — per-parameter LR adapts to compensate. With fixed-LR optimizers, alpha=1.0 would be too low. |
| `gradient_dtype` | `fp32` | |
| `save_dtype` | `bf16` | |
| `max_train_steps` | 800–1536 (scale by batch) | |
| `prodigy_steps` | ~20 % of `max_train_steps` | |

**VRAM behavior to expect at batch 10 / unsloth GC:**
- Step 0: ~26 GB (model loaded)
- Step ~50: ~29 GB peak (optimizer state initialized + cache build)
- Step ~400: ~22 GB (allocator settles, cache overhead freed)
- Step ~470: ~28 GB spike (Prodigy peak-LR phase — orthograd temp tensors)

This ±6 GB swing is normal. Do not panic at the step-50 peak.

---

### Cloud / API — 40–80 GB (Runpod A100, H100, RTX 6000 Ada, etc.)

| Setting | Value | Why |
|---|---|---|
| `blocks_to_swap` | 0 | No swap needed. |
| `gradient_checkpointing` | `disabled` | With this much VRAM, storing all activations is fine. Disabling GC is fastest — backward pass does not recompute anything. |
| `cache_latents` | `disk` | Still saves time even at high batch — repeated VAE passes at batch 16 add up. |
| `cache_text_encoder_outputs` | `disk` | Keeps Qwen3 off VRAM; no reason to leave it loaded. |
| `highvram` | `True` | Disables conservative memory management — ekes out a small speed boost when VRAM is not a constraint. |
| `batch_size` | 8–16 | At this scale, higher batch converges faster with fewer steps. |
| `resolution` | 1024–1280 | |
| `network_dim` | 32–64 | |
| `network_alpha` | dim / 2 | `effective_scale = alpha / dim`. dim/2 → scale=0.5, classic convention. alpha=1.0 also valid with Prodigy — per-parameter LR adapts to compensate. With fixed-LR optimizers, alpha=1.0 would be too low. |
| `gradient_dtype` | `bf16` | A100/H100 have native bfloat16 hardware paths. bf16 gradients halve memory with no meaningful quality loss at this batch size. |
| `save_dtype` | `bf16` | |
| `max_train_steps` | 400–800 (scale by batch) | At batch 16, 400 steps = 6400 image exposures. Often enough for a strong LoRA. |
| `prodigy_steps` | ~20 % of `max_train_steps` | |

---

## Step Count Scaling

Formula: `new_steps = (reference_steps × reference_batch) / your_batch`

**Why:** each step processes `batch_size` images. Doubling the batch halves the steps needed for the same total number of image exposures. Prodigy adaptation quality also improves with larger batches because gradient estimates are less noisy.

Keep `prodigy_steps` at **~20 %** of `max_train_steps`.
Hard lower bound: `max_train_steps` must be at least `prodigy_steps × 1.5` — the ScheduleFree phase needs runway after adaptation or loss never descends.

| `batch_size` | `max_train_steps` | `prodigy_steps` | Image exposures |
|---|---|---|---|
| 1 | 4000 | 800 | 4000 |
| 2 | 2000 | 400 | 4000 |
| 4 | 1000 | 200 | 4000 |
| 6 | 700 | 140 | 4200 |
| 8 | 500 | 100 | 4000 |
| 10 | 400 | 80 | 4000 |
| 16 | 250 | 50 | 4000 |

---

## Prodigy Tuning by Dataset Size

Prodigy's `d_coef` controls how aggressively it grows its learning rate estimate. Larger datasets absorb more aggressive values; smaller datasets need a gentler touch.

| Dataset size | Images | `d_coef` | `num_repeats` | Notes |
|---|---|---|---|---|
| Tiny | < 30 | 1.2–1.3 | 3–4 | Very sensitive — 1.5 causes visible oscillation; lower d_coef prevents this |
| Small | 30–80 | 1.3–1.5 | 3 | d_coef 1.5 produces minor loss bumps near loop boundaries but recovers |
| Medium | 80–200 | 1.5 | 2–3 | Stable at 1.5 with batch ≥ 6; clean monotonic loss curve |
| Large | 200+ | 1.5–2.0 | 2 | More gradient signal absorbs higher d_coef; 2.0 wastes ~300 steps on oscillation |

**d_coef 2.0 warning:** causes a ~300-step oscillation band before ScheduleFree locks in. For runs under 2000 total steps this wastes 15–20 % of your training budget. Use 1.5 unless total steps > 2000.

---

## Loop Boundary Tips

AnimaTrainLoop runs a fixed number of steps then hands the trainer state to the next loop node. Each handoff is a potential source of small loss bumps.

- **Fewer longer loops are better.** 4 loops × 512 steps beats 8 loops × 256 steps for the same total — fewer handoff events = smoother loss curve.
- If you see a loss bump at every boundary, increase loop step count or reduce total loop count.
- **Enable `save_state = True` on every AnimaTrainLoRASave.** No state file = no resume. If training is interrupted between loops you lose everything back to the last saved state.

---

## Quick Diagnostics

| Symptom | Likely cause | Fix |
|---|---|---|
| NaN loss from step 1 | `learning_rate` set to a small value (e.g. 1e-4) with Prodigy | Set `learning_rate = 1.0` |
| Crash / screen off at step ~100–123 (AMD) | AOTriton experimental kernels active | `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0` in env |
| GPU page fault / system hang (AMD) | `blocks_to_swap > 0` on ROCm | Set `blocks_to_swap = 0` |
| Loss flat for 200+ steps | `d_coef` too low, or `prodigy_steps` ends too early | Raise `d_coef` or extend `prodigy_steps` |
| Loss oscillates for 300+ steps | `d_coef` too high (2.0 with short run) | Reduce `d_coef` to 1.5 |
| OOM crash | VRAM budget exceeded | Enable GC, reduce `batch_size`, add `blocks_to_swap` |
| Resume starts loss at 0.12+ (fresh-start level) | Optimizer state not loaded | Verify `resume_args` is connected; confirm `-state` folder path is correct |
| Loss bumps at every loop boundary | Loop step count too small | Switch to 512-step loops; keep `save_state = True` on each save node |
| First run much slower than second | Cold latent / text encoder cache | Normal — warm cache cuts step time by ~30 %. First run on a new dataset is always slower. |
