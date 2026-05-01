# Anima LoRA Training — Parameter Reference

A concise guide covering every parameter exposed in **Init Anima LoRA Training** and **Optimizer Config ProdigyPlusScheduleFree**, with baseline recommendations.

---

## Quick-start Baseline

For a typical concept or style LoRA (10–30 images):

| Parameter | Baseline value |
|---|---|
| network_dim | 16 |
| network_alpha | 1.0 |
| learning_rate | 1.0 (let Prodigy adapt) |
| max_train_steps | 1000–1500 |
| cache_latents | disk |
| cache_text_encoder_outputs | disabled |
| timestep_sampling | sigmoid |
| weighting_scheme | logit_normal |
| gradient_dtype | fp32 |
| save_dtype | bf16 |
| gradient_checkpointing | enabled |
| min_snr_gamma | 5.0 |
| lr_scheduler | schedulefree |

---

## Init Anima LoRA Training

### Output

| Parameter | Values | Notes |
|---|---|---|
| **output_name** | string | Base name for saved files. Steps and dtype are appended automatically. |
| **output_dir** | string | Relative to ComfyUI root. Created automatically if missing. |

---

### LoRA Network

| Parameter | Values | Advice |
|---|---|---|
| **network_dim** | 1–512 | LoRA rank. **8–16** for style/character; **32–64** for complex concepts. Higher = larger file and more capacity, not always better quality. |
| **network_alpha** | 0.0–2048 | Scaling factor. Effective LR scale = alpha/dim. Keep at **1.0** to decouple LR from dim choice. Setting alpha = dim gives scale = 1.0. |

---

### Training Duration & Learning Rate

| Parameter | Values | Advice |
|---|---|---|
| **learning_rate** | float | With ProdigyPlusScheduleFree, set to **1.0** — Prodigy ignores this and adapts its own LR automatically. Only matters for non-Prodigy optimizers. |
| **max_train_steps** | int | **500–1000** for quick tests; **1000–1500** for standard training; **2000+** for fine detail. More images = more steps needed. Rule of thumb: ~100 steps per training image. |

---

### Caching

| Parameter | Values | Advice |
|---|---|---|
| **cache_latents** | disk / memory / disabled | **disk** recommended — pre-encodes images with the VAE, skipping VAE forward pass each step. Speeds up training and reduces VRAM. Requires images to be fixed (no random crop mid-training). |
| **cache_text_encoder_outputs** | disk / memory / disabled | **disk** saves significant VRAM by offloading Qwen3 after caching. **Incompatible with caption dropout**. Use **disabled** if you want caption dropout during training. |

---

### Memory Management

| Parameter | Values | Advice |
|---|---|---|
| **blocks_to_swap** | 0–26 | Offloads Anima DiT blocks to CPU RAM. Use when VRAM is tight. **0** = no swap (fastest). Start with **8–12** if you run out of VRAM. Higher values trade speed for memory. Max 26 (of 28 total blocks). |
| **gradient_checkpointing** | enabled / enabled_with_cpu_offloading / enabled_with_unsloth_offloading / disabled | **enabled** (default) recomputes activations during backward pass to save VRAM at ~20% speed cost. **cpu_offloading** offloads activations to RAM (more savings, slower). **disabled** fastest but highest VRAM. |
| **highvram** | bool | **False** (default) enables memory saving measures. Set **True** only if you have ample VRAM and want a small speed boost. |

---

### Timestep Sampling

Controls how noise levels are selected during training. This determines *which parts of the diffusion process* the model trains on most.

| Parameter | Values | Advice |
|---|---|---|
| **timestep_sampling** | sigmoid / uniform / shift / flux_shift / sigma | **sigmoid** (default) — logit-normal distribution, concentrates training on middle timesteps. **shift** — adds a flow-shift bias, better for high-resolution images. **flux_shift** — resolution-adaptive shift similar to FLUX. **uniform** — equal weight across all timesteps. **sigma** — enables the `weighting_scheme` parameter below. |
| **sigmoid_scale** | 0.1–5.0 | Spread of the sigmoid distribution. **1.0** (default) is standard. Higher values spread more uniformly across timesteps. Only applies to `sigmoid`, `shift`, `flux_shift`. |
| **discrete_flow_shift** | 0.0–10.0 | Shift value for `shift` mode. **1.0** (default, no shift). Values **1.5–3.0** bias toward denoising mid-to-low noise, useful for detail. |

> **Note:** `weighting_scheme`, `logit_mean`, `logit_std`, and `mode_scale` only take effect when `timestep_sampling = sigma`. With all other sampling modes, sigmas are not computed and these parameters are ignored.

---

### Loss Weighting (active only with `timestep_sampling = sigma`)

| Parameter | Values | Advice |
|---|---|---|
| **weighting_scheme** | logit_normal / sigma_sqrt / mode / cosmap / none | **logit_normal** — focuses loss on middle timesteps (recommended with sigma sampling). **sigma_sqrt** — emphasizes low-noise (detail) steps. **mode** — peaks at a specific noise level set by mode_scale. **cosmap** — cosine-based smooth weighting. **none** — uniform weighting. |
| **logit_mean** | -10.0–10.0 | Mean for logit_normal distribution. **0.0** centers on mid-noise timesteps. Negative values bias toward high noise (structure); positive toward low noise (detail). |
| **logit_std** | 0.0–10.0 | Spread of logit_normal. **1.0** (default) is standard. Lower = more concentrated; higher = more uniform. |
| **mode_scale** | 0.0–10.0 | Only for `mode` weighting. **1.29** (default from literature). Higher shifts peak toward cleaner timesteps. |

---

### Precision & Performance

| Parameter | Values | Advice |
|---|---|---|
| **gradient_dtype** | fp32 / fp16 / bf16 | **fp32** (default) — most stable, recommended for starting out. **bf16** — halves gradient memory, comparable quality on modern GPUs. **fp16** — less stable than bf16, avoid unless needed. |
| **save_dtype** | fp32 / fp16 / bf16 / fp8_e4m3fn / fp8_e5m2 | **bf16** (recommended) — best balance of file size and quality. **fp32** for archival. **fp8** halves file size again with minor quality loss; use fp8_e4m3fn (wider range). |
| **attention_mode** | torch / sdpa / flash / xformers | **torch** (default, same as sdpa) — works everywhere. **flash** — faster if `flash-attn` is installed. **xformers** — requires `split_attn = True`. |
| **split_attn** | bool | **True only when attention_mode = xformers**. Splits attention batch to reduce memory. |

---

### Validation

| Parameter | Values | Advice |
|---|---|---|
| **sample_prompts** | string | Prompts for generating validation images. Separate multiple prompts with `\|`. Keep prompts representative of your training data. Example: `portrait of a woman \| cityscape at night`. |

---

### Optional: Per-Layer Learning Rates

By default all layer groups use the base `learning_rate`. Set any of these to **0** to keep group at base LR. Setting a non-zero value overrides LR for that group specifically.

| Parameter | Default | Notes |
|---|---|---|
| **llm_adapter_lr** | 0 (= base LR) | LR for the LLM Adapter (bridges Qwen3 to DiT). Sensitive — use a lower value (e.g. 0.5× base) if overfitting to text. |
| **self_attn_lr** | 0 (= base LR) | Self-attention layers. Higher LR here emphasizes spatial composition. |
| **cross_attn_lr** | 0 (= base LR) | Cross-attention layers (text-image alignment). Key for concept fidelity. |
| **mlp_lr** | 0 (= base LR) | MLP layers. Lower LR here reduces risk of breaking base model quality. |

---

## Optimizer Config ProdigyPlusScheduleFree

ProdigyPlus is a self-adapting optimizer — it estimates the optimal learning rate automatically. Set `learning_rate = 1.0` in the training node and let Prodigy handle LR.

### Core Optimizer

| Parameter | Default | Values | Advice |
|---|---|---|---|
| **min_snr_gamma** | 5.0 | 0.0–20.0 | Reduces weight of high-loss (high-noise) timesteps. **5.0** is recommended by the paper. **0.0** disables it. Lower values have stronger effect. Helps training stability. |
| **d0** | 1e-6 | float | Initial LR seed. Rarely needs changing. Increase slightly (e.g. 1e-5) if Prodigy is very slow to start adapting. |
| **d_coef** | 1.0 | 0.5–2.0 | Coefficient for LR estimate. **1.0** is default; **0.5** more conservative, **2.0** more aggressive. |

---

### Schedule

| Parameter | Default | Values | Advice |
|---|---|---|---|
| **lr_scheduler** | schedulefree | schedulefree / cosine / linear / polynomial | **schedulefree** (default) — Prodigy manages its own implicit decay; best for most cases. **cosine/linear** — traditional decay curves without schedule-free. **polynomial** — configurable decay shape; set power and cycles below. |
| **lr_scheduler_power** | 1.0 | 0.1–2.0 | Polynomial scheduler only. **1.0** = linear. Higher = sharp initial drop, gentle later. |
| **lr_scheduler_num_cycles** | 1 | 1–10 | Polynomial scheduler only. Number of LR restarts. Use `power = 1.0` if cycles > 1. |
| **prodigy_steps** | 0 | int | Freeze Prodigy's LR adaptation after N steps. **0** = never freeze (recommended for most runs). Set to **15–25% of max_train_steps** for very long runs (3000+ steps) or when using `use_speed = True`. |

---

### Stability & Experimental Features

| Parameter | Default | Advice |
|---|---|---|
| **use_stableadamw** | True | Scales updates by gradient RMS (similar to Adafactor). Strongly recommended — improves stability. Disable only if the adaptive LR never moves from d0. |
| **stochastic_rounding** | True | Improves bf16 training quality by rounding stochastically instead of deterministically. Leave **True** when using bf16 gradient dtype. |
| **use_orthograd** | True | Projects gradients orthogonal to weights, reducing overfitting and improving generalization. Recommended **True**. |
| **split_groups** | False | Tracks separate adaptation values per parameter group. Enable for fine-grained LR control across layers. Slight overhead. |
| **use_bias_correction** | False | RAdam-style correction. Slows initial LR adaptation by ~10×. Pair with `use_speed = True` to compensate. |
| **use_speed** | False | Momentum-based alternative to Prodigy's LR estimation. **Faster convergence** but less stable with weight decay or very long runs. When enabled, set `prodigy_steps` to 20–25% of total steps to freeze adaptation before instability. |
| **use_grams** | False | Experimental: sign-aligned updates without first-moment estimates. Minor effect; leave **False** unless experimenting. |
| **use_adopt** | False | Experimental: updates second moment after parameter step (partial ADOPT). Leave **False** for normal runs. |
| **use_focus** | False | Experimental: modifies update for large step sizes. **Incompatible with factorisation, Muon, and Adam-atan2**. Leave **False**. |
| **extra_optimizer_args** | "" | Additional key=value pairs passed to the optimizer, separated by `\|`. For advanced use only. |

---

## VRAM Reference

Anima is significantly smaller than FLUX, making LoRA training accessible on mid-range GPUs.

| VRAM | Recommended settings |
|---|---|
| 8 GB | blocks_to_swap=16, cache_latents=disk, cache_text_encoder_outputs=disk, gradient_checkpointing=enabled |
| 12 GB | blocks_to_swap=8, cache_latents=disk, cache_text_encoder_outputs=disabled, gradient_checkpointing=enabled |
| 16 GB | blocks_to_swap=0, cache_latents=disk, gradient_checkpointing=enabled |
| 24 GB+ | blocks_to_swap=0, highvram=True, gradient_checkpointing=disabled |

---

## Common Configurations

### Standard concept LoRA
```
network_dim: 16, network_alpha: 1.0
timestep_sampling: sigmoid
weighting_scheme: logit_normal (ignored with sigmoid — use sigma if you want it active)
max_train_steps: 1200
gradient_dtype: fp32, save_dtype: bf16
min_snr_gamma: 5.0, lr_scheduler: schedulefree
```

### Style LoRA (broader generalization)
```
network_dim: 8, network_alpha: 1.0
timestep_sampling: shift, discrete_flow_shift: 1.5
max_train_steps: 800–1000
gradient_dtype: fp32, save_dtype: bf16
mlp_lr: 0 (keep same as base)
```

### Detailed character / high fidelity
```
network_dim: 32, network_alpha: 1.0
timestep_sampling: sigma, weighting_scheme: sigma_sqrt
max_train_steps: 1500–2000
gradient_dtype: fp32, save_dtype: bf16
cross_attn_lr: 0 (same as base, focus on fidelity)
```

---

## Notes

- **Prodigy LR**: The `learning_rate` field in Init Anima LoRA Training is overridden by Prodigy. Always set it to **1.0** when using ProdigyPlusScheduleFree.
- **cache_text_encoder_outputs + dropout**: These are mutually exclusive. Caching locks captions — disable caching if you want caption dropout active during training.
- **weighting_scheme is inactive** unless `timestep_sampling = sigma`. With sigmoid/shift/uniform/flux_shift, sigmas are not computed and loss weighting is uniform.
- **LoRA format**: Trained LoRAs use sd-scripts key format (`lora_unet_*`). To load in a native ComfyUI Anima node, use `networks/convert_anima_lora_to_comfy.py` to convert keys to ComfyUI format (`diffusion_model.*`).
