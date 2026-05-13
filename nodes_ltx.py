import os
import sys
import subprocess
import logging
import toml
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


def _nearest_valid_frames(n: int) -> int:
    """Snap to nearest (k*8)+1 frame count, minimum 1."""
    if n <= 1:
        return 1
    k = max(1, round((n - 1) / 8))
    return k * 8 + 1


def _check_musubi_tuner():
    try:
        import musubi_tuner  # noqa: F401
    except ImportError:
        raise ImportError(
            "musubi-tuner is not installed. LTX training nodes require it.\n"
            "Install in your ComfyUI venv:\n"
            "  pip install git+https://github.com/AkaneTendo25/musubi-tuner@ltx-2-dev"
        )


class LTXModelSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dit": (folder_paths.get_filename_list("unet"),),
                "gemma3_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Path to Gemma 3 12B directory (HuggingFace format) or a single FP8 .safetensors file.",
                    },
                ),
                "gemma_load_dtype": (
                    ["4bit_nf4", "8bit", "bf16", "fp16"],
                    {
                        "default": "4bit_nf4",
                        "tooltip": (
                            "Gemma 3 quantization used during preprocessing (text encoding). "
                            "4bit_nf4 keeps Gemma at ~7 GB RAM. bf16 requires ~24 GB."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("LTX_MODELS",)
    RETURN_NAMES = ("ltx_models",)
    FUNCTION = "select"
    CATEGORY = "FluxTrainer/LTX"

    def select(self, dit, gemma3_path, gemma_load_dtype):
        dit_path = folder_paths.get_full_path("unet", dit)
        gemma3_path = gemma3_path.strip()
        if not gemma3_path:
            raise ValueError("gemma3_path is required. Provide the path to your Gemma 3 12B directory or .safetensors file.")
        gemma3_path = os.path.abspath(gemma3_path)
        if not os.path.exists(gemma3_path):
            raise ValueError(f"Gemma 3 path not found: {gemma3_path}")

        return ({
            "dit": dit_path,
            "gemma3_path": gemma3_path,
            "gemma_load_dtype": gemma_load_dtype,
        },)


class LTXVideoDataset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_dir": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Directory containing video files and matching .txt caption files "
                            "(same filename stem, e.g. clip01.mp4 + clip01.txt)."
                        ),
                    },
                ),
                "target_frames": (
                    "INT",
                    {
                        "default": 97,
                        "min": 1,
                        "max": 1000,
                        "step": 8,
                        "tooltip": (
                            "Frame count for training. Must be (n×8)+1 — e.g. 25, 33, 49, 65, 97, 121. "
                            "Will snap to the nearest valid value if needed. "
                            "97 frames ≈ 3.9s at 25fps (good for 4–8s source clips)."
                        ),
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 4096,
                        "step": 32,
                        "tooltip": "Training width in pixels. Must be a multiple of 32.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 384,
                        "min": 64,
                        "max": 4096,
                        "step": 32,
                        "tooltip": "Training height in pixels. Must be a multiple of 32.",
                    },
                ),
                "frame_extraction": (
                    ["head", "center", "tail"],
                    {
                        "tooltip": "Which segment of each video to sample frames from: head=start, center=middle, tail=end.",
                    },
                ),
                "target_fps": (
                    "FLOAT",
                    {
                        "default": 25.0,
                        "min": 1.0,
                        "max": 60.0,
                        "step": 0.5,
                        "tooltip": "Training frame rate. Source videos are resampled to this FPS before extraction.",
                    },
                ),
                "caption_extension": (
                    "STRING",
                    {
                        "default": ".txt",
                        "multiline": False,
                        "tooltip": "Extension of caption sidecar files.",
                    },
                ),
                "num_repeats": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Number of times each video is repeated per epoch.",
                    },
                ),
                "caption_dropout_rate": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Probability of dropping the caption per training step to strengthen unconditional generation. 0.05 is a safe default.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LTX_DATASET",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "configure"
    CATEGORY = "FluxTrainer/LTX"

    def configure(
        self, video_dir, target_frames, width, height,
        frame_extraction, target_fps, caption_extension,
        num_repeats, caption_dropout_rate,
    ):
        video_dir = os.path.abspath(video_dir.strip())
        if not os.path.isdir(video_dir):
            raise ValueError(f"video_dir not found: {video_dir}")

        if width % 32 != 0:
            raise ValueError(f"width {width} must be a multiple of 32.")
        if height % 32 != 0:
            raise ValueError(f"height {height} must be a multiple of 32.")

        valid_frames = _nearest_valid_frames(target_frames)
        if valid_frames != target_frames:
            logger.warning(
                f"target_frames {target_frames} is not (n×8)+1 — snapped to {valid_frames}. "
                f"Valid values near this range: {valid_frames - 8}, {valid_frames}, {valid_frames + 8}."
            )

        video_files = [f for f in os.listdir(video_dir) if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS]
        if not video_files:
            raise ValueError(f"No video files found in {video_dir}. Supported: {', '.join(sorted(VIDEO_EXTENSIONS))}")

        missing = [
            os.path.splitext(f)[0] for f in video_files
            if not os.path.exists(os.path.join(video_dir, os.path.splitext(f)[0] + caption_extension))
        ]
        if missing:
            logger.warning(
                f"{len(missing)}/{len(video_files)} video(s) have no caption file: "
                + ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
            )

        logger.info(
            f"LTXVideoDataset: {len(video_files)} videos | {width}×{height} | "
            f"{valid_frames} frames ({frame_extraction}) | {target_fps} fps | ×{num_repeats} repeats"
        )

        return ({
            "video_dir": video_dir,
            "target_frames": valid_frames,
            "width": width,
            "height": height,
            "frame_extraction": frame_extraction,
            "target_fps": target_fps,
            "caption_extension": caption_extension,
            "num_repeats": num_repeats,
            "caption_dropout_rate": caption_dropout_rate,
            "num_videos": len(video_files),
        },)


class LTXPreprocessDataset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ltx_models": ("LTX_MODELS",),
                "dataset": ("LTX_DATASET",),
                "cache_dir": (
                    "STRING",
                    {
                        "default": "ltx_trainer_cache",
                        "multiline": False,
                        "tooltip": "Directory where encoded latents and text embeddings are written. Root is the ComfyUI folder.",
                    },
                ),
            },
            "optional": {
                "force_recompute": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Recompute all cache files even if they already exist.",
                    },
                ),
                "vae_spatial_tile_size": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 1024,
                        "step": 64,
                        "tooltip": "Spatial tile size for VAE encoding. Lower = less VRAM. 512 is safe on 32 GB.",
                    },
                ),
                "vae_temporal_tile_size": (
                    "INT",
                    {
                        "default": 64,
                        "min": 16,
                        "max": 256,
                        "step": 8,
                        "tooltip": "Temporal tile size for VAE encoding. Lower = less VRAM. 64 is safe on 32 GB.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("PRECOMPUTED_LTX_DATASET",)
    RETURN_NAMES = ("precomputed_dataset",)
    FUNCTION = "preprocess"
    CATEGORY = "FluxTrainer/LTX"

    def preprocess(
        self, ltx_models, dataset, cache_dir,
        force_recompute=False,
        vae_spatial_tile_size=512,
        vae_temporal_tile_size=64,
    ):
        _check_musubi_tuner()

        cache_dir = os.path.abspath(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

        toml_path = os.path.join(cache_dir, "ltx_dataset.toml")
        self._write_dataset_toml(dataset, cache_dir, toml_path)

        # Step 1 — video latent cache
        n = dataset["num_videos"]
        if force_recompute or not self._cache_complete(cache_dir, n, "_ltx2.safetensors"):
            logger.info("LTX preprocess — 1/2: encoding video latents (VAE)...")
            self._run_latent_cache(ltx_models, toml_path, vae_spatial_tile_size, vae_temporal_tile_size)
        else:
            logger.info("LTX preprocess — 1/2: video latent cache already complete, skipping.")

        # Step 2 — text encoder cache
        if force_recompute or not self._cache_complete(cache_dir, n, "_ltx2_te.safetensors"):
            logger.info("LTX preprocess — 2/2: encoding text embeddings (Gemma 3)...")
            self._run_text_cache(ltx_models, toml_path)
        else:
            logger.info("LTX preprocess — 2/2: text encoder cache already complete, skipping.")

        logger.info(f"LTX preprocessing complete → {cache_dir}")

        return ({
            "cache_dir": cache_dir,
            "toml_path": toml_path,
            "dataset": dataset,
        },)

    def _write_dataset_toml(self, dataset, cache_dir, toml_path):
        config = {
            "general": {
                "resolution": [dataset["width"], dataset["height"]],
                "caption_extension": dataset["caption_extension"],
                "batch_size": 1,
                "enable_bucket": False,
            },
            "datasets": [
                {
                    "video_directory": dataset["video_dir"],
                    "cache_directory": cache_dir,
                    "target_frames": [dataset["target_frames"]],
                    "frame_extraction": dataset["frame_extraction"],
                    "target_fps": dataset["target_fps"],
                    "num_repeats": dataset["num_repeats"],
                    "caption_dropout_rate": dataset["caption_dropout_rate"],
                }
            ],
        }
        with open(toml_path, "w") as f:
            toml.dump(config, f)
        logger.info(f"Dataset TOML → {toml_path}")

    def _cache_complete(self, cache_dir, num_videos, suffix):
        """True if at least num_videos files with the given suffix exist in cache_dir."""
        found = sum(1 for f in os.listdir(cache_dir) if f.endswith(suffix))
        return found >= num_videos

    def _run_subprocess(self, cmd, step_name):
        """Run a subprocess and stream its combined stdout/stderr to the logger."""
        logger.info(f"CMD: {' '.join(str(c) for c in cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            logger.info(f"[{step_name}] {line.rstrip()}")
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(
                f"{step_name} subprocess exited with code {process.returncode}. See log output above."
            )

    def _run_latent_cache(self, ltx_models, toml_path, spatial_tile, temporal_tile):
        cmd = [
            sys.executable, "-m", "musubi_tuner.ltx2_cache_latents",
            "--dataset_config", toml_path,
            "--ltx2_checkpoint", ltx_models["dit"],
            "--ltx2_mode", "video",
            "--vae_spatial_tile_size", str(spatial_tile),
            "--vae_temporal_tile_size", str(temporal_tile),
            "--device", "cuda",
        ]
        self._run_subprocess(cmd, "cache_latents")

    def _run_text_cache(self, ltx_models, toml_path):
        cmd = [
            sys.executable, "-m", "musubi_tuner.ltx2_cache_text_encoder_outputs",
            "--dataset_config", toml_path,
            "--ltx2_checkpoint", ltx_models["dit"],
            "--ltx2_mode", "video",
        ]

        gemma_path = ltx_models["gemma3_path"]
        if os.path.isdir(gemma_path):
            cmd += ["--gemma_root", gemma_path]
        else:
            cmd += ["--gemma_safetensors", gemma_path]

        dtype = ltx_models["gemma_load_dtype"]
        if dtype == "4bit_nf4":
            cmd += ["--gemma_load_in_4bit", "--gemma_bnb_4bit_quant_type", "nf4"]
        elif dtype == "8bit":
            cmd += ["--gemma_load_in_8bit"]

        self._run_subprocess(cmd, "cache_text")


NODE_CLASS_MAPPINGS = {
    "LTXModelSelect": LTXModelSelect,
    "LTXVideoDataset": LTXVideoDataset,
    "LTXPreprocessDataset": LTXPreprocessDataset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXModelSelect": "LTX Model Select",
    "LTXVideoDataset": "LTX Video Dataset",
    "LTXPreprocessDataset": "LTX Preprocess Dataset",
}
