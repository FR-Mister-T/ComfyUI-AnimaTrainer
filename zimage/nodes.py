import csv
import logging
import os
import shutil

import torch
import folder_paths
import comfy.model_management as mm
import comfy.utils

logger = logging.getLogger(__name__)

script_directory = os.path.dirname(os.path.abspath(__file__))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def _check_diffsynth():
    try:
        import diffsynth  # noqa: F401
    except ImportError:
        raise ImportError(
            "diffsynth is not installed. Z-Image training nodes require it.\n"
            "Install in your ComfyUI Python environment:\n"
            "  pip install diffsynth>=2.0.12 accelerate>=0.34.0 peft>=0.12.0 pandas>=2.0.0"
        )


def _convert_state_dict_dtype(state_dict, dtype_str):
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str)
    if dtype is None:
        return state_dict
    return {k: v.to(dtype) if v.is_floating_point() else v for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# ZImageModelSelect
# ---------------------------------------------------------------------------

class ZImageModelSelect:
    """
    Select Z-Image model weights for training.

    The transformer comes from the base model (Tongyi-MAI/Z-Image).
    The text encoder, VAE, and tokenizer are shared with Z-Image-Turbo and
    loaded from the shared_model field.

    Both fields accept a HuggingFace model ID (downloaded automatically on
    first use) or an absolute local directory path in HF Diffusers layout.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transformer_model": (
                    "STRING",
                    {
                        "default": "Tongyi-MAI/Z-Image",
                        "multiline": False,
                        "tooltip": (
                            "HuggingFace model ID or local path for the Z-Image transformer. "
                            "Use the undistilled base model (Tongyi-MAI/Z-Image) for LoRA training "
                            "— it has a full training signal. Z-Image-Turbo is distilled and "
                            "should not be used as the base for training."
                        ),
                    },
                ),
                "shared_model": (
                    "STRING",
                    {
                        "default": "Tongyi-MAI/Z-Image-Turbo",
                        "multiline": False,
                        "tooltip": (
                            "HuggingFace model ID or local path for the text encoder and VAE. "
                            "Both Z-Image variants share the same text encoder and VAE. "
                            "Z-Image-Turbo is the canonical source."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("ZIMAGE_MODELS",)
    RETURN_NAMES = ("zimage_models",)
    FUNCTION = "select"
    CATEGORY = "FluxTrainer/ZImage"

    def select(self, transformer_model, shared_model):
        transformer_model = transformer_model.strip()
        shared_model = shared_model.strip()
        if not transformer_model:
            raise ValueError("transformer_model must not be empty.")
        if not shared_model:
            raise ValueError("shared_model must not be empty.")

        for label, path in [("transformer_model", transformer_model), ("shared_model", shared_model)]:
            if os.path.sep in path or path.startswith("/") or path.startswith("."):
                if not os.path.isdir(path):
                    raise ValueError(f"{label} local path not found: {path}")

        model_id_with_origin_paths = (
            f"{transformer_model}:transformer/*.safetensors,"
            f"{shared_model}:text_encoder/*.safetensors,"
            f"{shared_model}:vae/diffusion_pytorch_model.safetensors"
        )
        logger.info(f"ZImageModelSelect: transformer={transformer_model} | text_enc+VAE={shared_model}")
        return ({"transformer_model": transformer_model, "shared_model": shared_model,
                 "model_id_with_origin_paths": model_id_with_origin_paths},)


# ---------------------------------------------------------------------------
# ZImageDataset
# ---------------------------------------------------------------------------

class ZImageDataset:
    """
    Scan a folder of images with sidecar caption files and generate a CSV
    metadata file for DiffSynth's UnifiedDataset.

    Each image must have a matching caption sidecar with the same stem
    (e.g. photo01.jpg + photo01.txt). Images without a caption are skipped
    with a warning. The generated zimage_metadata.csv is written into
    image_dir and consumed by ZImageInitTraining.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_dir": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Directory containing training images and matching caption sidecar files. "
                            "Supported formats: jpg, jpeg, png, webp, bmp, tiff."
                        ),
                    },
                ),
                "caption_extension": (
                    "STRING",
                    {
                        "default": ".txt",
                        "multiline": False,
                        "tooltip": "Extension of caption sidecar files (e.g. .txt or .caption).",
                    },
                ),
                "max_pixels": (
                    "INT",
                    {
                        "default": 1048576,
                        "min": 65536,
                        "max": 4194304,
                        "step": 65536,
                        "tooltip": (
                            "Maximum pixels per training image (width × height). Images are "
                            "dynamically resized to fit within this budget while preserving "
                            "aspect ratio. Both dimensions are rounded to multiples of 16. "
                            "Default 1048576 = 1024×1024."
                        ),
                    },
                ),
                "repeat": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 500,
                        "step": 1,
                        "tooltip": (
                            "Times each image is repeated per epoch. Increase for small datasets. "
                            "With 10 images and repeat=50, one epoch = 500 steps."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("ZIMAGE_DATASET",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "configure"
    CATEGORY = "FluxTrainer/ZImage"

    def configure(self, image_dir, caption_extension, max_pixels, repeat):
        image_dir = os.path.abspath(image_dir.strip())
        if not os.path.isdir(image_dir):
            raise ValueError(f"image_dir not found: {image_dir}")

        caption_extension = caption_extension.strip()
        if not caption_extension.startswith("."):
            caption_extension = "." + caption_extension

        rows = []
        missing_captions = []
        for fname in sorted(os.listdir(image_dir)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in IMAGE_EXTENSIONS:
                continue
            cap_path = os.path.join(image_dir, stem + caption_extension)
            if not os.path.exists(cap_path):
                missing_captions.append(fname)
                continue
            with open(cap_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            if not prompt:
                logger.warning(f"ZImageDataset: empty caption for {fname} — skipping.")
                continue
            rows.append({"image": fname, "prompt": prompt})

        if missing_captions:
            logger.warning(
                f"ZImageDataset: {len(missing_captions)} image(s) skipped — "
                f"no {caption_extension} sidecar: "
                + ", ".join(missing_captions[:5])
                + ("..." if len(missing_captions) > 5 else "")
            )
        if not rows:
            raise ValueError(
                f"No valid image+caption pairs found in {image_dir}. "
                f"Expected images with matching {caption_extension} caption sidecars."
            )

        metadata_path = os.path.join(image_dir, "zimage_metadata.csv")
        with open(metadata_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "prompt"])
            writer.writeheader()
            writer.writerows(rows)

        logger.info(
            f"ZImageDataset: {len(rows)} images | max_pixels={max_pixels} | "
            f"×{repeat} repeat | metadata → {metadata_path}"
        )
        return ({"image_dir": image_dir, "metadata_path": metadata_path,
                 "max_pixels": max_pixels, "repeat": repeat, "num_images": len(rows)},)


# ---------------------------------------------------------------------------
# ZImageInitTraining
# ---------------------------------------------------------------------------

class ZImageInitTraining:
    """
    Initialize Z-Image LoRA training in-process.

    Loads models, injects LoRA via PEFT, sets up the optimizer and dataloader,
    and returns a ZIMAGE_TRAINER handle. Training does not start here —
    connect ZImageTrainLoop to run steps.

    LoRA targets the S3-DiT transformer only (lora_base_model=dit). The
    default target modules cover all attention projections and SwiGLU MLP
    gates as recommended by DiffSynth.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zimage_models": ("ZIMAGE_MODELS",),
                "dataset": ("ZIMAGE_DATASET",),
                "output_name": (
                    "STRING",
                    {"default": "zimage_lora", "multiline": False},
                ),
                "output_dir": (
                    "STRING",
                    {
                        "default": "zimage_trainer_output",
                        "multiline": False,
                        "tooltip": "Output folder for LoRA checkpoints. Relative to the ComfyUI root.",
                    },
                ),
                "rank": (
                    "INT",
                    {
                        "default": 32,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                        "tooltip": (
                            "LoRA rank. 32 is the DiffSynth default for Z-Image and a solid "
                            "starting point. Higher rank captures more detail at the cost of "
                            "VRAM and file size."
                        ),
                    },
                ),
                "learning_rate": (
                    "FLOAT",
                    {
                        "default": 1e-4,
                        "min": 1e-7,
                        "max": 1e-2,
                        "step": 1e-6,
                        "tooltip": "Learning rate. DiffSynth default for Z-Image LoRA is 1e-4.",
                    },
                ),
                "num_epochs": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                        "tooltip": (
                            "Total training epochs. One epoch = all images × repeat. "
                            "With 10 images, repeat=50: 1 epoch = 500 steps."
                        ),
                    },
                ),
                "save_dtype": (
                    ["bf16", "fp16", "fp32"],
                    {
                        "default": "bf16",
                        "tooltip": "Dtype to save LoRA checkpoints as. bf16 is smallest and sufficient.",
                    },
                ),
                "sample_prompts": (
                    "STRING",
                    {
                        "default": "a portrait of a person | a landscape scene",
                        "multiline": True,
                        "tooltip": (
                            "Validation prompts for ZImageTrainValidate. "
                            "Separate multiple prompts with |."
                        ),
                    },
                ),
            },
            "optional": {
                "target_modules": (
                    "STRING",
                    {
                        "default": "to_q,to_k,to_v,to_out.0,w1,w2,w3",
                        "multiline": False,
                        "tooltip": (
                            "Comma-separated S3-DiT module names to inject LoRA into. "
                            "Default covers all attention projections and SwiGLU MLP gates. "
                            "Leave unchanged unless you have a specific reason to modify."
                        ),
                    },
                ),
                "weight_decay": (
                    "FLOAT",
                    {
                        "default": 0.01,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "AdamW weight decay. DiffSynth default is 0.01.",
                    },
                ),
                "gradient_accumulation": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 32,
                        "step": 1,
                        "tooltip": (
                            "Accumulate gradients over N steps before updating weights. "
                            "Effective batch size = gradient_accumulation × 1."
                        ),
                    },
                ),
                "use_gradient_checkpointing": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Recompute activations during backward pass to reduce VRAM. "
                            "Slightly slower but essential for Z-Image (6B) on 32 GB."
                        ),
                    },
                ),
                "num_workers": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 16,
                        "step": 1,
                        "tooltip": "DataLoader worker processes. 0 = main process only (for debugging).",
                    },
                ),
                "lora_checkpoint": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Path to an existing LoRA checkpoint to resume from. Leave empty to start fresh.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("ZIMAGE_TRAINER", "INT")
    RETURN_NAMES = ("network_trainer", "epochs_count")
    FUNCTION = "init_training"
    CATEGORY = "FluxTrainer/ZImage"

    def init_training(
        self,
        zimage_models,
        dataset,
        output_name,
        output_dir,
        rank,
        learning_rate,
        num_epochs,
        save_dtype,
        sample_prompts,
        target_modules="to_q,to_k,to_v,to_out.0,w1,w2,w3",
        weight_decay=0.01,
        gradient_accumulation=1,
        use_gradient_checkpointing=True,
        num_workers=4,
        lora_checkpoint="",
    ):
        _check_diffsynth()

        import importlib.util, accelerate
        from diffsynth.core import UnifiedDataset
        from diffsynth.diffusion.logger import ModelLogger

        # Load bundled train.py without relying on package install path
        _spec = importlib.util.spec_from_file_location(
            "zimage_train", os.path.join(script_directory, "train.py")
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        ZImageTrainingModule = _mod.ZImageTrainingModule

        mm.soft_empty_cache()

        output_dir = os.path.abspath(output_dir.strip())
        os.makedirs(output_dir, exist_ok=True)

        lora_checkpoint = lora_checkpoint.strip() or None
        if lora_checkpoint and not os.path.exists(lora_checkpoint):
            raise ValueError(f"lora_checkpoint not found: {lora_checkpoint}")

        target_modules = target_modules.strip()
        if not target_modules:
            raise ValueError("target_modules must not be empty.")

        accelerator = accelerate.Accelerator(
            mixed_precision="bf16",
            gradient_accumulation_steps=gradient_accumulation,
        )

        ds = UnifiedDataset(
            base_path=dataset["image_dir"],
            metadata_path=dataset["metadata_path"],
            repeat=dataset["repeat"],
            data_file_keys=["image"],
            main_data_operator=UnifiedDataset.default_image_operator(
                base_path=dataset["image_dir"],
                max_pixels=dataset["max_pixels"],
                height_division_factor=16,
                width_division_factor=16,
            ),
        )

        logger.info(
            f"ZImageInitTraining: loading models — transformer={zimage_models['transformer_model']} | "
            f"rank={rank} | lr={learning_rate} | epochs={num_epochs}"
        )

        with torch.inference_mode(False):
            model = ZImageTrainingModule(
                model_id_with_origin_paths=zimage_models["model_id_with_origin_paths"],
                lora_base_model="dit",
                lora_target_modules=target_modules,
                lora_rank=rank,
                lora_checkpoint=lora_checkpoint,
                use_gradient_checkpointing=use_gradient_checkpointing,
                device="cpu",
            )

        optimizer = torch.optim.AdamW(
            model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        dataloader = torch.utils.data.DataLoader(
            ds, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers
        )

        model.to(device=accelerator.device)
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )

        model_logger = ModelLogger(output_dir, remove_prefix_in_ckpt="pipe.dit.")

        prompts = [p.strip() for p in sample_prompts.split("|") if p.strip()]

        trainer = {
            "model": model,
            "accelerator": accelerator,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "dataloader": dataloader,
            "dataset": ds,
            "model_logger": model_logger,
            "global_step": 0,
            "current_epoch": 0,
            "num_epochs": num_epochs,
            "output_dir": output_dir,
            "output_name": output_name,
            "save_dtype": save_dtype,
            "sample_prompts": prompts,
            "data_iter": iter(dataloader),
        }

        logger.info(
            f"ZImageInitTraining: ready — {dataset['num_images']} images × {dataset['repeat']} "
            f"repeat | steps per epoch ≈ {dataset['num_images'] * dataset['repeat']}"
        )
        return (trainer, num_epochs)


# ---------------------------------------------------------------------------
# ZImageTrainLoop
# ---------------------------------------------------------------------------

class ZImageTrainLoop:
    """
    Run N training steps on the Z-Image LoRA.

    Connect the output network_trainer back into the input to create a
    training loop in the ComfyUI graph. Insert ZImageTrainLoRASave or
    ZImageTrainValidate between loop iterations to checkpoint or preview
    quality at any step count you choose.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "network_trainer": ("ZIMAGE_TRAINER",),
                "steps": (
                    "INT",
                    {
                        "default": 100,
                        "min": 1,
                        "max": 100000,
                        "step": 1,
                        "tooltip": "Number of training steps to run before returning.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("ZIMAGE_TRAINER", "INT")
    RETURN_NAMES = ("network_trainer", "steps")
    FUNCTION = "train"
    CATEGORY = "FluxTrainer/ZImage"

    def train(self, network_trainer, steps):
        model = network_trainer["model"]
        accelerator = network_trainer["accelerator"]
        optimizer = network_trainer["optimizer"]
        scheduler = network_trainer["scheduler"]
        dataset = network_trainer["dataset"]
        model_logger = network_trainer["model_logger"]
        data_iter = network_trainer["data_iter"]
        global_step = network_trainer["global_step"]

        comfy_pbar = comfy.utils.ProgressBar(steps)

        with torch.inference_mode(False):
            model.train()
            for _ in range(steps):
                try:
                    data = next(data_iter)
                except StopIteration:
                    network_trainer["current_epoch"] += 1
                    data_iter = iter(network_trainer["dataloader"])
                    network_trainer["data_iter"] = data_iter
                    data = next(data_iter)

                with accelerator.accumulate(model):
                    if dataset.load_from_cache:
                        loss = model({}, inputs=data)
                    else:
                        loss = model(data)
                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                global_step += 1
                comfy_pbar.update(1)

        network_trainer["global_step"] = global_step
        network_trainer["data_iter"] = data_iter
        logger.info(f"ZImageTrainLoop: step {global_step} | loss {loss.item():.5f}")
        return (network_trainer, global_step)


# ---------------------------------------------------------------------------
# ZImageTrainLoRASave
# ---------------------------------------------------------------------------

class ZImageTrainLoRASave:
    """
    Save the current LoRA weights as a safetensors checkpoint.

    Checkpoint is named step-{N}.safetensors. The pipe.dit. prefix is
    stripped so the file contains bare transformer key names compatible
    with DiffSynth's LoRA loader. Optionally copies to the ComfyUI
    loras/zimage_trainer/ folder for quick inference testing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "network_trainer": ("ZIMAGE_TRAINER",),
                "copy_to_comfy_lora_folder": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Copy the saved checkpoint to ComfyUI/models/loras/zimage_trainer/.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("ZIMAGE_TRAINER", "STRING", "INT")
    RETURN_NAMES = ("network_trainer", "lora_path", "steps")
    FUNCTION = "save"
    CATEGORY = "FluxTrainer/ZImage"

    def save(self, network_trainer, copy_to_comfy_lora_folder):
        model = network_trainer["model"]
        accelerator = network_trainer["accelerator"]
        model_logger = network_trainer["model_logger"]
        global_step = network_trainer["global_step"]
        save_dtype = network_trainer["save_dtype"]
        output_dir = network_trainer["output_dir"]

        file_name = f"step-{global_step}.safetensors"

        with torch.inference_mode(False):
            accelerator.wait_for_everyone()
            state_dict = accelerator.get_state_dict(model)
            if accelerator.is_main_process:
                state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                    state_dict, remove_prefix=model_logger.remove_prefix_in_ckpt
                )
                state_dict = _convert_state_dict_dtype(state_dict, save_dtype)
                os.makedirs(output_dir, exist_ok=True)
                lora_path = os.path.join(output_dir, file_name)
                accelerator.save(state_dict, lora_path, safe_serialization=True)

        if copy_to_comfy_lora_folder:
            destination_dir = os.path.join(folder_paths.models_dir, "loras", "zimage_trainer")
            os.makedirs(destination_dir, exist_ok=True)
            shutil.copy(lora_path, os.path.join(destination_dir, file_name))
            logger.info(f"ZImageTrainLoRASave: copied → {destination_dir}")

        logger.info(f"ZImageTrainLoRASave: saved {lora_path}")
        return (network_trainer, lora_path, global_step)


# ---------------------------------------------------------------------------
# ZImageTrainValidationSettings
# ---------------------------------------------------------------------------

class ZImageTrainValidationSettings:
    """Inference parameters used by ZImageTrainValidate."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 16}),
                "steps": (
                    "INT",
                    {
                        "default": 28,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Denoising steps for validation inference. 28 is the Z-Image base model default.",
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 1.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "CFG guidance scale. Z-Image base model range is 3–5.",
                    },
                ),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("ZIMAGE_VALSETTINGS",)
    RETURN_NAMES = ("validation_settings",)
    FUNCTION = "set"
    CATEGORY = "FluxTrainer/ZImage"

    def set(self, **kwargs):
        return (kwargs,)


# ---------------------------------------------------------------------------
# ZImageTrainValidate
# ---------------------------------------------------------------------------

class ZImageTrainValidate:
    """
    Generate validation images using the current LoRA state.

    Switches the model to eval mode, runs ZImagePipeline inference using
    the sample_prompts configured in ZImageInitTraining (or overridden via
    validation_settings), then switches back to training mode. Returns
    images as a ComfyUI IMAGE batch for preview.

    If inference fails (e.g. VRAM too tight mid-training), a blank image
    is returned with a warning rather than crashing the graph.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "network_trainer": ("ZIMAGE_TRAINER",),
            },
            "optional": {
                "validation_settings": ("ZIMAGE_VALSETTINGS",),
            },
        }

    RETURN_TYPES = ("ZIMAGE_TRAINER", "IMAGE")
    RETURN_NAMES = ("network_trainer", "validation_images")
    FUNCTION = "validate"
    CATEGORY = "FluxTrainer/ZImage"

    def validate(self, network_trainer, validation_settings=None):
        model = network_trainer["model"]
        accelerator = network_trainer["accelerator"]
        sample_prompts = network_trainer["sample_prompts"]

        vs = validation_settings or {}
        width = vs.get("width", 512)
        height = vs.get("height", 512)
        num_steps = vs.get("steps", 28)
        guidance_scale = vs.get("guidance_scale", 4.0)
        seed = vs.get("seed", 42)

        model_obj = accelerator.unwrap_model(model)
        pipe = model_obj.pipe

        images = []
        try:
            model_obj.eval()
            with torch.no_grad():
                for prompt in sample_prompts:
                    result = pipe(
                        prompt=prompt,
                        negative_prompt="",
                        height=height,
                        width=width,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        cfg_normalization=False,
                        generator=torch.Generator(device=accelerator.device).manual_seed(seed),
                    )
                    pil_img = result.images[0]
                    import numpy as np
                    img_tensor = torch.from_numpy(np.array(pil_img)).float() / 255.0
                    images.append(img_tensor)
        except Exception as e:
            logger.warning(f"ZImageTrainValidate: inference failed — {e}. Returning blank image.")
            images = [torch.zeros(height, width, 3)]
        finally:
            model_obj.train()

        image_batch = torch.stack(images)  # [N, H, W, 3]
        return (network_trainer, image_batch)


# ---------------------------------------------------------------------------
# ZImageTrainEnd
# ---------------------------------------------------------------------------

class ZImageTrainEnd:
    """
    Finalize Z-Image LoRA training, save the final checkpoint, and free
    GPU memory. Always call this node at the end of your training graph
    to ensure the model is properly released.

    The final checkpoint is named step-{N}-final.safetensors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "network_trainer": ("ZIMAGE_TRAINER",),
                "copy_to_comfy_lora_folder": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Copy the final LoRA to ComfyUI/models/loras/zimage_trainer/.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("lora_name", "lora_path", "total_steps")
    FUNCTION = "endtrain"
    CATEGORY = "FluxTrainer/ZImage"
    OUTPUT_NODE = True

    def endtrain(self, network_trainer, copy_to_comfy_lora_folder):
        model = network_trainer["model"]
        accelerator = network_trainer["accelerator"]
        model_logger = network_trainer["model_logger"]
        global_step = network_trainer["global_step"]
        save_dtype = network_trainer["save_dtype"]
        output_dir = network_trainer["output_dir"]
        output_name = network_trainer["output_name"]

        file_name = f"{output_name}_step-{global_step}-final.safetensors"

        with torch.inference_mode(False):
            accelerator.wait_for_everyone()
            state_dict = accelerator.get_state_dict(model)
            if accelerator.is_main_process:
                state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                    state_dict, remove_prefix=model_logger.remove_prefix_in_ckpt
                )
                state_dict = _convert_state_dict_dtype(state_dict, save_dtype)
                os.makedirs(output_dir, exist_ok=True)
                lora_path = os.path.join(output_dir, file_name)
                accelerator.save(state_dict, lora_path, safe_serialization=True)
                logger.info(f"ZImageTrainEnd: final checkpoint → {lora_path}")

            accelerator.end_training()

        if copy_to_comfy_lora_folder:
            destination_dir = os.path.join(folder_paths.models_dir, "loras", "zimage_trainer")
            os.makedirs(destination_dir, exist_ok=True)
            shutil.copy(lora_path, os.path.join(destination_dir, file_name))
            logger.info(f"ZImageTrainEnd: copied → {destination_dir}")

        mm.soft_empty_cache()

        return (output_name, lora_path, global_step)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "ZImageModelSelect": ZImageModelSelect,
    "ZImageDataset": ZImageDataset,
    "ZImageInitTraining": ZImageInitTraining,
    "ZImageTrainLoop": ZImageTrainLoop,
    "ZImageTrainLoRASave": ZImageTrainLoRASave,
    "ZImageTrainValidationSettings": ZImageTrainValidationSettings,
    "ZImageTrainValidate": ZImageTrainValidate,
    "ZImageTrainEnd": ZImageTrainEnd,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageModelSelect": "Z-Image Model Select",
    "ZImageDataset": "Z-Image Dataset",
    "ZImageInitTraining": "Z-Image Init Training",
    "ZImageTrainLoop": "Z-Image Train Loop",
    "ZImageTrainLoRASave": "Z-Image Train LoRA Save",
    "ZImageTrainValidationSettings": "Z-Image Train Validation Settings",
    "ZImageTrainValidate": "Z-Image Train Validate",
    "ZImageTrainEnd": "Z-Image Train End",
}
