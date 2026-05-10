import os
import torch
import folder_paths
import comfy.model_management as mm
import comfy.utils
import toml
import json
import time
import shutil
import shlex

script_directory = os.path.dirname(os.path.abspath(__file__))

from .anima_train_network_comfy import AnimaNetworkTrainer
from .anima_lllite_train_comfy import AnimaLLLiteTrainer
from .library import anima_train_utils
from .library.device_utils import init_ipex

init_ipex()

from .library import train_util
from .train_network import setup_parser as train_network_setup_parser

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AnimaModelSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dit": (folder_paths.get_filename_list("unet"),),
                "qwen3": (folder_paths.get_filename_list("clip"),),
                "vae": (folder_paths.get_filename_list("vae"),),
            },
            "optional": {
                "t5_tokenizer_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Path to T5 tokenizer directory. Leave empty to use bundled configs/t5_old/",
                    },
                ),
                "llm_adapter_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Path to separate LLM adapter weights. Leave empty to load from DiT checkpoint.",
                    },
                ),
                "lora_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "pre-trained LoRA path to load (network_weights)",
                    },
                ),
            },
        }

    RETURN_TYPES = ("TRAIN_ANIMA_MODELS",)
    RETURN_NAMES = ("anima_models",)
    FUNCTION = "loadmodel"
    CATEGORY = "FluxTrainer/Anima"

    def loadmodel(self, dit, qwen3, vae, t5_tokenizer_path="", llm_adapter_path="", lora_path=""):
        dit_path = folder_paths.get_full_path("unet", dit)
        qwen3_path = folder_paths.get_full_path("clip", qwen3)
        vae_path = folder_paths.get_full_path("vae", vae)

        anima_models = {
            "dit": dit_path,
            "qwen3": qwen3_path,
            "vae": vae_path,
            "t5_tokenizer_path": t5_tokenizer_path if t5_tokenizer_path else None,
            "llm_adapter_path": llm_adapter_path if llm_adapter_path else None,
            "lora_path": lora_path,
        }

        return (anima_models,)


class InitAnimaLoRATraining:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "anima_models": ("TRAIN_ANIMA_MODELS",),
                "dataset": ("JSON",),
                "optimizer_settings": ("ARGS",),
                "output_name": ("STRING", {"default": "anima_lora", "multiline": False}),
                "output_dir": (
                    "STRING",
                    {
                        "default": "anima_trainer_output",
                        "multiline": False,
                        "tooltip": "path to output folder, root is the 'ComfyUI' folder",
                    },
                ),
                "network_dim": ("INT", {"default": 8, "min": 1, "max": 100000, "step": 1, "tooltip": "LoRA rank (network dim)"}),
                "network_alpha": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2048.0, "step": 0.01, "tooltip": "LoRA alpha (scaling factor)"},
                ),
                "learning_rate": (
                    "FLOAT",
                    {"default": 1e-4, "min": 0.0, "max": 10.0, "step": 0.000001, "tooltip": "learning rate"},
                ),
                "max_train_steps": (
                    "INT",
                    {"default": 1500, "min": 1, "max": 100000, "step": 1, "tooltip": "max number of training steps"},
                ),
                "cache_latents": (["disk", "memory", "disabled"], {"tooltip": "cache VAE latents to speed up training"}),
                "cache_text_encoder_outputs": (
                    ["disk", "memory", "disabled"],
                    {"tooltip": "cache Qwen3 text encoder outputs"},
                ),
                "blocks_to_swap": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 26,
                        "step": 1,
                        "tooltip": "number of DiT blocks to swap to CPU to reduce VRAM usage (max 26 for Anima-28B)",
                    },
                ),
                "weighting_scheme": (["logit_normal", "sigma_sqrt", "mode", "cosmap", "none"],),
                "logit_mean": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "mean for logit_normal weighting scheme",
                    },
                ),
                "logit_std": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "std for logit_normal weighting scheme",
                    },
                ),
                "mode_scale": (
                    "FLOAT",
                    {"default": 1.29, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "scale for mode weighting scheme"},
                ),
                "timestep_sampling": (
                    ["sigmoid", "uniform", "sigma", "shift", "flux_shift"],
                    {"tooltip": "method to sample timesteps during training"},
                ),
                "sigmoid_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "scale for sigmoid timestep sampling"},
                ),
                "discrete_flow_shift": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001, "tooltip": "flow shift for rectified flow scheduler"},
                ),
                "highvram": ("BOOLEAN", {"default": False, "tooltip": "disable memory saving measures (faster if enough VRAM)"}),
                "gradient_dtype": (
                    ["fp32", "fp16", "bf16"],
                    {"default": "fp32", "tooltip": "dtype for gradient computation"},
                ),
                "save_dtype": (
                    ["fp32", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2"],
                    {"default": "bf16", "tooltip": "dtype to save LoRA checkpoints as"},
                ),
                "attention_mode": (
                    ["torch", "xformers", "flash", "sdpa"],
                    {"default": "torch", "tooltip": "attention implementation (torch=sdpa, flash requires flash-attn)"},
                ),
                "split_attn": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "split attention computation (required for xformers attention mode)"},
                ),
                "sample_prompts": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "illustration of a kitten | photograph of a turtle",
                        "tooltip": "validation sample prompts, separate multiple with `|`",
                    },
                ),
            },
            "optional": {
                "additional_args": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "additional CLI args passed to the training command"},
                ),
                "resume_args": ("ARGS", {"default": "", "tooltip": "resume training args"}),
                "gradient_checkpointing": (
                    ["enabled", "enabled_with_cpu_offloading", "enabled_with_unsloth_offloading", "disabled"],
                    {"default": "enabled", "tooltip": "gradient checkpointing mode"},
                ),
                "vae_chunk_size": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1024,
                        "step": 2,
                        "tooltip": "Spatial chunk size for VAE encoding to reduce VRAM usage. Must be even. 0 = disabled. Recommended: 256 for tight VRAM.",
                    },
                ),
                "loss_args": ("ARGS", {"default": "", "tooltip": "loss function args"}),
                "llm_adapter_lr": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.000001,
                        "tooltip": "learning rate for LLM adapter (0=same as base LR)",
                    },
                ),
                "self_attn_lr": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.000001,
                        "tooltip": "learning rate for self-attention layers (0=same as base LR)",
                    },
                ),
                "cross_attn_lr": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.000001,
                        "tooltip": "learning rate for cross-attention layers (0=same as base LR)",
                    },
                ),
                "mlp_lr": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.000001,
                        "tooltip": "learning rate for MLP layers (0=same as base LR)",
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (
        "NETWORKTRAINER",
        "INT",
        "KOHYA_ARGS",
    )
    RETURN_NAMES = (
        "network_trainer",
        "epochs_count",
        "args",
    )
    FUNCTION = "init_training"
    CATEGORY = "FluxTrainer/Anima"

    def init_training(
        self,
        anima_models,
        dataset,
        optimizer_settings,
        sample_prompts,
        output_name,
        attention_mode,
        gradient_dtype,
        save_dtype,
        additional_args=None,
        resume_args=None,
        gradient_checkpointing="enabled",
        vae_chunk_size=0,
        prompt=None,
        extra_pnginfo=None,
        loss_args=None,
        llm_adapter_lr=0.0,
        self_attn_lr=0.0,
        cross_attn_lr=0.0,
        mlp_lr=0.0,
        **kwargs,
    ):
        mm.soft_empty_cache()

        output_dir = os.path.abspath(kwargs.get("output_dir"))
        os.makedirs(output_dir, exist_ok=True)

        total, used, free = shutil.disk_usage(output_dir)
        required_free_space = 2 * (2**30)
        if free <= required_free_space:
            raise ValueError(f"Insufficient disk space. Required: {required_free_space/2**30:.1f}GB. Available: {free/2**30:.1f}GB")

        dataset_config = dataset["datasets"]
        dataset_toml = toml.dumps(json.loads(dataset_config))

        parser = train_network_setup_parser()
        anima_train_utils.add_anima_training_arguments(parser)
        parser.add_argument("--unsloth_offload_checkpointing", action="store_true")

        if additional_args is not None:
            print(f"additional_args: {additional_args}")
            args, _ = parser.parse_known_args(args=shlex.split(additional_args))
        else:
            args, _ = parser.parse_known_args()

        if kwargs.get("cache_latents") == "memory":
            kwargs["cache_latents"] = True
            kwargs["cache_latents_to_disk"] = False
        elif kwargs.get("cache_latents") == "disk":
            kwargs["cache_latents"] = True
            kwargs["cache_latents_to_disk"] = True
            kwargs["caption_dropout_rate"] = 0.0
            kwargs["shuffle_caption"] = False
            kwargs["token_warmup_step"] = 0.0
            kwargs["caption_tag_dropout_rate"] = 0.0
        else:
            kwargs["cache_latents"] = False
            kwargs["cache_latents_to_disk"] = False

        if kwargs.get("cache_text_encoder_outputs") == "memory":
            kwargs["cache_text_encoder_outputs"] = True
            kwargs["cache_text_encoder_outputs_to_disk"] = False
        elif kwargs.get("cache_text_encoder_outputs") == "disk":
            kwargs["cache_text_encoder_outputs"] = True
            kwargs["cache_text_encoder_outputs_to_disk"] = True
        else:
            kwargs["cache_text_encoder_outputs"] = False
            kwargs["cache_text_encoder_outputs_to_disk"] = False

        if "|" in sample_prompts:
            prompts = sample_prompts.split("|")
        else:
            prompts = [sample_prompts]

        config_dict = {
            "sample_prompts": prompts,
            "save_precision": save_dtype,
            "mixed_precision": "bf16",
            "num_cpu_threads_per_process": 1,
            "pretrained_model_name_or_path": anima_models["dit"],
            "qwen3": anima_models["qwen3"],
            "vae": anima_models["vae"],
            "t5_tokenizer_path": anima_models["t5_tokenizer_path"],
            "llm_adapter_path": anima_models["llm_adapter_path"],
            "save_model_as": "safetensors",
            "persistent_data_loader_workers": False,
            "max_data_loader_n_workers": 0,
            "seed": 42,
            "network_module": ".networks.lora_anima",
            "dataset_config": dataset_toml,
            "output_name": f"{output_name}_rank{kwargs.get('network_dim')}_{save_dtype}",
            "loss_type": "l2",
            "alpha_mask": dataset["alpha_mask"],
            "network_train_unet_only": True,
            "disable_mmap_load_safetensors": False,
            "attn_mode": attention_mode,
            "split_attn": kwargs.pop("split_attn", False),
        }

        gradient_dtype_settings = {
            "fp16": {"full_fp16": True, "full_bf16": False, "mixed_precision": "fp16"},
            "bf16": {"full_bf16": True, "full_fp16": False, "mixed_precision": "bf16"},
        }
        config_dict.update(gradient_dtype_settings.get(gradient_dtype, {}))

        if gradient_checkpointing == "disabled":
            config_dict["gradient_checkpointing"] = False
        elif gradient_checkpointing == "enabled_with_cpu_offloading":
            config_dict["gradient_checkpointing"] = True
            config_dict["cpu_offload_checkpointing"] = True
        elif gradient_checkpointing == "enabled_with_unsloth_offloading":
            config_dict["gradient_checkpointing"] = True
            config_dict["unsloth_offload_checkpointing"] = True
        else:
            config_dict["gradient_checkpointing"] = True

        if anima_models["lora_path"]:
            config_dict["network_weights"] = anima_models["lora_path"]

        if vae_chunk_size and vae_chunk_size > 0:
            config_dict["vae_chunk_size"] = vae_chunk_size

        if llm_adapter_lr and llm_adapter_lr > 0:
            config_dict["llm_adapter_lr"] = llm_adapter_lr
        if self_attn_lr and self_attn_lr > 0:
            config_dict["self_attn_lr"] = self_attn_lr
        if cross_attn_lr and cross_attn_lr > 0:
            config_dict["cross_attn_lr"] = cross_attn_lr
        if mlp_lr and mlp_lr > 0:
            config_dict["mlp_lr"] = mlp_lr

        config_dict.update(kwargs)
        config_dict.update(optimizer_settings)

        if loss_args:
            config_dict.update(loss_args)

        if resume_args:
            config_dict.update(resume_args)

        for key, value in config_dict.items():
            setattr(args, key, value)

        saved_args_file_path = os.path.join(output_dir, f"{output_name}_args.json")
        with open(saved_args_file_path, "w") as f:
            json.dump(vars(args), f, indent=4)

        metadata = {}
        if extra_pnginfo is not None:
            metadata.update(extra_pnginfo["workflow"])

        saved_workflow_file_path = os.path.join(output_dir, f"{output_name}_workflow.json")
        with open(saved_workflow_file_path, "w") as f:
            json.dump(metadata, f, indent=4)

        with torch.inference_mode(False):
            network_trainer = AnimaNetworkTrainer()
            training_loop = network_trainer.init_train(args)

        epochs_count = network_trainer.num_train_epochs

        trainer = {
            "network_trainer": network_trainer,
            "training_loop": training_loop,
        }
        return (trainer, epochs_count, args)


class AnimaTrainLoop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "network_trainer": ("NETWORKTRAINER",),
                "steps": ("INT", {"default": 1, "min": 1, "max": 10000, "step": 1, "tooltip": "number of training steps to run"}),
            },
        }

    RETURN_TYPES = (
        "NETWORKTRAINER",
        "INT",
    )
    RETURN_NAMES = (
        "network_trainer",
        "steps",
    )
    FUNCTION = "train"
    CATEGORY = "FluxTrainer/Anima"

    def train(self, network_trainer, steps):
        with torch.inference_mode(False):
            training_loop = network_trainer["training_loop"]
            network_trainer = network_trainer["network_trainer"]

            target_global_step = network_trainer.global_step + steps
            comfy_pbar = comfy.utils.ProgressBar(steps)
            network_trainer.comfy_pbar = comfy_pbar

            network_trainer.optimizer_train_fn()

            while network_trainer.global_step < target_global_step:
                steps_done = training_loop(
                    break_at_steps=target_global_step,
                    epoch=network_trainer.current_epoch.value,
                )

                if network_trainer.global_step >= network_trainer.args.max_train_steps:
                    break

            trainer = {
                "network_trainer": network_trainer,
                "training_loop": training_loop,
            }
        return (trainer, network_trainer.global_step)


class AnimaTrainLoRASave:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "network_trainer": ("NETWORKTRAINER",),
                "save_state": ("BOOLEAN", {"default": False, "tooltip": "also save the full training state (optimizer, scheduler, etc.)"}),
                "copy_to_comfy_lora_folder": ("BOOLEAN", {"default": False, "tooltip": "copy the saved LoRA to ComfyUI loras/anima_trainer/ folder"}),
            },
        }

    RETURN_TYPES = (
        "NETWORKTRAINER",
        "STRING",
        "INT",
    )
    RETURN_NAMES = (
        "network_trainer",
        "lora_path",
        "steps",
    )
    FUNCTION = "save"
    CATEGORY = "FluxTrainer/Anima"

    def save(self, network_trainer, save_state, copy_to_comfy_lora_folder):
        with torch.inference_mode(False):
            trainer = network_trainer["network_trainer"]
            global_step = trainer.global_step

            ckpt_name = train_util.get_step_ckpt_name(trainer.args, "." + trainer.args.save_model_as, global_step)
            trainer.save_model(ckpt_name, trainer.accelerator.unwrap_model(trainer.network), global_step, trainer.current_epoch.value + 1)

            remove_step_no = train_util.get_remove_step_no(trainer.args, global_step)
            if remove_step_no is not None:
                remove_ckpt_name = train_util.get_step_ckpt_name(trainer.args, "." + trainer.args.save_model_as, remove_step_no)
                trainer.remove_model(remove_ckpt_name)

            if save_state:
                train_util.save_and_remove_state_stepwise(trainer.args, trainer.accelerator, global_step)

            lora_path = os.path.join(trainer.args.output_dir, ckpt_name)
            if copy_to_comfy_lora_folder:
                destination_dir = os.path.join(folder_paths.models_dir, "loras", "anima_trainer")
                os.makedirs(destination_dir, exist_ok=True)
                shutil.copy(lora_path, os.path.join(destination_dir, ckpt_name))

        return (network_trainer, lora_path, global_step)


class AnimaTrainEnd:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "network_trainer": ("NETWORKTRAINER",),
                "save_state": ("BOOLEAN", {"default": True, "tooltip": "save full training state on finish"}),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "lora_name",
        "metadata",
        "lora_path",
    )
    FUNCTION = "endtrain"
    CATEGORY = "FluxTrainer/Anima"
    OUTPUT_NODE = True

    def endtrain(self, network_trainer, save_state):
        with torch.inference_mode(False):
            training_loop = network_trainer["training_loop"]
            network_trainer = network_trainer["network_trainer"]

            network_trainer.metadata["ss_epoch"] = str(network_trainer.num_train_epochs)
            network_trainer.metadata["ss_training_finished_at"] = str(time.time())

            network = network_trainer.accelerator.unwrap_model(network_trainer.network)

            network_trainer.accelerator.end_training()
            network_trainer.optimizer_eval_fn()

            if save_state:
                train_util.save_state_on_train_end(network_trainer.args, network_trainer.accelerator)

            ckpt_name = train_util.get_last_ckpt_name(network_trainer.args, "." + network_trainer.args.save_model_as)
            network_trainer.save_model(ckpt_name, network, network_trainer.global_step, network_trainer.num_train_epochs, force_sync_upload=True)
            logger.info("model saved.")

            final_lora_name = str(network_trainer.args.output_name)
            final_lora_path = os.path.join(network_trainer.args.output_dir, ckpt_name)

            metadata = json.dumps(network_trainer.metadata, indent=2)

            training_loop = None
            network_trainer = None
            mm.soft_empty_cache()

        return (final_lora_name, metadata, final_lora_path)


class AnimaTrainValidationSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 1, "max": 256, "step": 1, "tooltip": "number of denoising steps"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16, "tooltip": "image width"}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16, "tooltip": "image height"}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 32.0, "step": 0.05, "tooltip": "CFG guidance scale"}),
                "flow_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "flow shift for sampling schedule"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
            },
        }

    RETURN_TYPES = ("VALSETTINGS",)
    RETURN_NAMES = ("validation_settings",)
    FUNCTION = "set"
    CATEGORY = "FluxTrainer/Anima"

    def set(self, **kwargs):
        return (kwargs,)


class AnimaTrainValidate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "network_trainer": ("NETWORKTRAINER",),
            },
            "optional": {
                "validation_settings": ("VALSETTINGS",),
            },
        }

    RETURN_TYPES = (
        "NETWORKTRAINER",
        "IMAGE",
    )
    RETURN_NAMES = (
        "network_trainer",
        "validation_images",
    )
    FUNCTION = "validate"
    CATEGORY = "FluxTrainer/Anima"

    def validate(self, network_trainer, validation_settings=None):
        training_loop = network_trainer["training_loop"]
        network_trainer_obj = network_trainer["network_trainer"]

        params = (
            network_trainer_obj.current_epoch.value,
            network_trainer_obj.global_step,
            validation_settings,
        )
        network_trainer_obj.optimizer_eval_fn()
        with torch.inference_mode(False):
            image_tensors = network_trainer_obj.sample_images(*params)

        trainer = {
            "network_trainer": network_trainer_obj,
            "training_loop": training_loop,
        }

        if image_tensors is None:
            blank = torch.zeros(1, 512, 512, 3)
            return (trainer, blank)

        return (trainer, image_tensors)


class InitAnimaLLLiteTraining:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "anima_models": ("TRAIN_ANIMA_MODELS",),
                "train_data_dir": ("STRING", {"default": "", "multiline": False, "tooltip": "Directory with training images"}),
                "conditioning_data_dir": ("STRING", {"default": "", "multiline": False, "tooltip": "Directory with conditioning/control images (paired with train_data_dir)"}),
                "optimizer_settings": ("ARGS",),
                "output_name": ("STRING", {"default": "anima_lllite", "multiline": False}),
                "output_dir": ("STRING", {"default": "anima_lllite_output", "multiline": False, "tooltip": "Output folder path (root is the ComfyUI folder)"}),
                "max_train_steps": ("INT", {"default": 1500, "min": 1, "max": 100000, "step": 1}),
                "cache_latents": (["disk", "memory", "disabled"],),
                "cache_text_encoder_outputs": (["disk", "memory", "disabled"],),
                "weighting_scheme": (["logit_normal", "sigma_sqrt", "mode", "cosmap", "none"],),
                "timestep_sampling": (["sigmoid", "uniform", "sigma", "shift", "flux_shift"],),
                "discrete_flow_shift": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "attention_mode": (["torch", "xformers", "flash", "sdpa"], {"default": "torch"}),
                "save_dtype": (["fp32", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2"], {"default": "bf16"}),
                "gradient_dtype": (["fp32", "fp16", "bf16"], {"default": "fp32"}),
                "cond_emb_dim": ("INT", {"default": 32, "min": 8, "max": 256, "step": 8, "tooltip": "Conditioning embedding dimension"}),
                "lllite_mlp_dim": ("INT", {"default": 64, "min": 16, "max": 512, "step": 8, "tooltip": "LLLite MLP hidden dimension"}),
                "lllite_target_layers": ("STRING", {"default": "self_attn_q", "multiline": False, "tooltip": "Target layers: preset name or comma-separated atomics (self_attn_q, self_attn_qkv, self_attn_qkv_cross_q)"}),
                "lllite_cond_dim": ("INT", {"default": 64, "min": 16, "max": 256, "step": 8, "tooltip": "Conditioning trunk channel width"}),
                "lllite_cond_resblocks": ("INT", {"default": 1, "min": 0, "max": 8, "step": 1, "tooltip": "Number of ResBlocks in conditioning trunk"}),
                "lllite_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "tooltip": "Multiplier applied to LLLite output"}),
                "lllite_use_aspp": ("BOOLEAN", {"default": False, "tooltip": "Enable ASPP (Atrous Spatial Pyramid Pooling) in conditioning trunk"}),
                "sample_prompts": ("STRING", {"multiline": True, "default": "illustration of a kitten", "tooltip": "Validation prompts, separate with |"}),
            },
            "optional": {
                "network_weights": ("STRING", {"default": "", "multiline": False, "tooltip": "Path to existing LLLite weights to resume from"}),
                "lllite_dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9, "step": 0.05, "tooltip": "Dropout rate for LLLite mid output (0 = disabled)"}),
                "gradient_checkpointing": (["enabled", "disabled"], {"default": "enabled"}),
                "vae_chunk_size": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 2, "tooltip": "VAE spatial chunk size for VRAM reduction (0=disabled)"}),
                "additional_args": ("STRING", {"multiline": True, "default": "", "tooltip": "Additional CLI args passed to the training command"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "INT", "KOHYA_ARGS")
    RETURN_NAMES = ("network_trainer", "epochs_count", "args")
    FUNCTION = "init_training"
    CATEGORY = "FluxTrainer/Anima"

    def init_training(
        self,
        anima_models,
        train_data_dir,
        conditioning_data_dir,
        optimizer_settings,
        output_name,
        output_dir,
        max_train_steps,
        cache_latents,
        cache_text_encoder_outputs,
        weighting_scheme,
        timestep_sampling,
        discrete_flow_shift,
        attention_mode,
        save_dtype,
        gradient_dtype,
        cond_emb_dim,
        lllite_mlp_dim,
        lllite_target_layers,
        lllite_cond_dim,
        lllite_cond_resblocks,
        lllite_multiplier,
        lllite_use_aspp,
        sample_prompts,
        network_weights="",
        lllite_dropout=0.0,
        gradient_checkpointing="enabled",
        vae_chunk_size=0,
        additional_args=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        mm.soft_empty_cache()

        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        total, used, free = shutil.disk_usage(output_dir)
        if free <= 2 * (2**30):
            raise ValueError(f"Insufficient disk space. Available: {free/2**30:.1f}GB")

        from .anima_lllite_train_comfy import AnimaLLLiteTrainer
        from .train_network import setup_parser as train_network_setup_parser

        parser = train_network_setup_parser()
        anima_train_utils.add_anima_training_arguments(parser)
        parser.add_argument("--unsloth_offload_checkpointing", action="store_true")
        parser.add_argument("--cond_emb_dim", type=int, default=32)
        parser.add_argument("--lllite_mlp_dim", type=int, default=64)
        parser.add_argument("--lllite_target_layers", type=str, default="self_attn_q")
        parser.add_argument("--lllite_cond_dim", type=int, default=64)
        parser.add_argument("--lllite_cond_resblocks", type=int, default=1)
        parser.add_argument("--lllite_multiplier", type=float, default=1.0)
        parser.add_argument("--lllite_use_aspp", action="store_true")
        parser.add_argument("--lllite_dropout", type=float, default=None)
        parser.add_argument("--conditioning_data_dir", type=str, default=None)
        parser.add_argument("--skip_latents_validity_check", action="store_true")

        if additional_args:
            args, _ = parser.parse_known_args(args=shlex.split(additional_args))
        else:
            args, _ = parser.parse_known_args()

        # Cache latents mapping
        if cache_latents == "memory":
            args.cache_latents = True
            args.cache_latents_to_disk = False
        elif cache_latents == "disk":
            args.cache_latents = True
            args.cache_latents_to_disk = True
            args.caption_dropout_rate = 0.0
            args.shuffle_caption = False
            args.token_warmup_step = 0.0
            args.caption_tag_dropout_rate = 0.0
        else:
            args.cache_latents = False
            args.cache_latents_to_disk = False

        # Cache text encoder outputs mapping
        if cache_text_encoder_outputs == "memory":
            args.cache_text_encoder_outputs = True
            args.cache_text_encoder_outputs_to_disk = False
        elif cache_text_encoder_outputs == "disk":
            args.cache_text_encoder_outputs = True
            args.cache_text_encoder_outputs_to_disk = True
        else:
            args.cache_text_encoder_outputs = False
            args.cache_text_encoder_outputs_to_disk = False

        prompts = sample_prompts.split("|") if "|" in sample_prompts else [sample_prompts]

        gradient_dtype_settings = {
            "fp16": {"full_fp16": True, "full_bf16": False, "mixed_precision": "fp16"},
            "bf16": {"full_bf16": True, "full_fp16": False, "mixed_precision": "bf16"},
        }

        config = {
            "sample_prompts": prompts,
            "save_precision": save_dtype,
            "mixed_precision": "bf16",
            "num_cpu_threads_per_process": 1,
            "pretrained_model_name_or_path": anima_models["dit"],
            "qwen3": anima_models["qwen3"],
            "vae": anima_models["vae"],
            "t5_tokenizer_path": anima_models.get("t5_tokenizer_path"),
            "llm_adapter_path": anima_models.get("llm_adapter_path"),
            "save_model_as": "safetensors",
            "persistent_data_loader_workers": False,
            "max_data_loader_n_workers": 0,
            "seed": 42,
            "output_dir": output_dir,
            "output_name": f"{output_name}_{save_dtype}",
            "loss_type": "l2",
            "network_train_unet_only": True,
            "disable_mmap_load_safetensors": False,
            "attn_mode": attention_mode,
            "max_train_steps": max_train_steps,
            "train_data_dir": train_data_dir,
            "conditioning_data_dir": conditioning_data_dir,
            "weighting_scheme": weighting_scheme,
            "timestep_sampling": timestep_sampling,
            "discrete_flow_shift": discrete_flow_shift,
            "gradient_checkpointing": gradient_checkpointing == "enabled",
            "cond_emb_dim": cond_emb_dim,
            "lllite_mlp_dim": lllite_mlp_dim,
            "lllite_target_layers": lllite_target_layers,
            "lllite_cond_dim": lllite_cond_dim,
            "lllite_cond_resblocks": lllite_cond_resblocks,
            "lllite_multiplier": lllite_multiplier,
            "lllite_use_aspp": lllite_use_aspp,
            "lllite_dropout": lllite_dropout if lllite_dropout > 0.0 else None,
            "skip_cache_check": False,
            "skip_latents_validity_check": False,
        }
        config.update(gradient_dtype_settings.get(gradient_dtype, {}))

        if network_weights:
            config["network_weights"] = network_weights
        if vae_chunk_size and vae_chunk_size > 0:
            config["vae_chunk_size"] = vae_chunk_size

        config.update(optimizer_settings)

        for key, value in config.items():
            setattr(args, key, value)

        saved_args_file_path = os.path.join(output_dir, f"{output_name}_lllite_args.json")
        with open(saved_args_file_path, "w") as f:
            json.dump(vars(args), f, indent=4)

        metadata = {}
        if extra_pnginfo is not None:
            metadata.update(extra_pnginfo.get("workflow", {}))

        saved_workflow_file_path = os.path.join(output_dir, f"{output_name}_lllite_workflow.json")
        with open(saved_workflow_file_path, "w") as f:
            json.dump(metadata, f, indent=4)

        with torch.inference_mode(False):
            network_trainer = AnimaLLLiteTrainer()
            training_loop = network_trainer.init_train(args)

        epochs_count = network_trainer.num_train_epochs

        trainer = {
            "network_trainer": network_trainer,
            "training_loop": training_loop,
        }
        return (trainer, epochs_count, args)


class AnimaLLLiteTrainSave:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "network_trainer": ("NETWORKTRAINER",),
                "save_state": ("BOOLEAN", {"default": False, "tooltip": "Also save the full training state (optimizer, scheduler)"}),
                "copy_to_comfy_lora_folder": ("BOOLEAN", {"default": False, "tooltip": "Copy saved checkpoint to ComfyUI loras/anima_trainer/ folder"}),
            },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "STRING", "INT")
    RETURN_NAMES = ("network_trainer", "lllite_path", "steps")
    FUNCTION = "save"
    CATEGORY = "FluxTrainer/Anima"

    def save(self, network_trainer, save_state, copy_to_comfy_lora_folder):
        with torch.inference_mode(False):
            trainer = network_trainer["network_trainer"]
            global_step = trainer.global_step

            ckpt_name = train_util.get_step_ckpt_name(trainer.args, "." + trainer.args.save_model_as, global_step)
            trainer.save_model(ckpt_name, global_step, trainer.current_epoch.value + 1)

            remove_step_no = train_util.get_remove_step_no(trainer.args, global_step)
            if remove_step_no is not None:
                remove_ckpt_name = train_util.get_step_ckpt_name(trainer.args, "." + trainer.args.save_model_as, remove_step_no)
                old_path = os.path.join(trainer.args.output_dir, remove_ckpt_name)
                if os.path.exists(old_path):
                    os.remove(old_path)

            if save_state:
                train_util.save_and_remove_state_stepwise(trainer.args, trainer.accelerator, global_step)

            lllite_path = os.path.join(trainer.args.output_dir, ckpt_name)
            if copy_to_comfy_lora_folder:
                destination_dir = os.path.join(folder_paths.models_dir, "loras", "anima_trainer")
                os.makedirs(destination_dir, exist_ok=True)
                shutil.copy(lllite_path, os.path.join(destination_dir, ckpt_name))

        return (network_trainer, lllite_path, global_step)


class AnimaLLLiteTrainEnd:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "network_trainer": ("NETWORKTRAINER",),
                "save_state": ("BOOLEAN", {"default": True, "tooltip": "Save full training state on finish"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("lllite_name", "metadata", "lllite_path")
    FUNCTION = "endtrain"
    CATEGORY = "FluxTrainer/Anima"
    OUTPUT_NODE = True

    def endtrain(self, network_trainer, save_state):
        with torch.inference_mode(False):
            training_loop = network_trainer["training_loop"]
            network_trainer = network_trainer["network_trainer"]

            network_trainer.accelerator.end_training()
            network_trainer.optimizer_eval_fn()

            if save_state:
                train_util.save_state_on_train_end(network_trainer.args, network_trainer.accelerator)

            ckpt_name = train_util.get_last_ckpt_name(network_trainer.args, "." + network_trainer.args.save_model_as)
            network_trainer.save_model(ckpt_name, network_trainer.global_step, network_trainer.num_train_epochs)

            final_name = str(network_trainer.args.output_name)
            final_path = os.path.join(network_trainer.args.output_dir, ckpt_name)
            metadata = json.dumps(network_trainer.metadata, indent=2)

            training_loop = None
            network_trainer = None
            mm.soft_empty_cache()

        return (final_name, metadata, final_path)


NODE_CLASS_MAPPINGS = {
    "AnimaModelSelect": AnimaModelSelect,
    "InitAnimaLoRATraining": InitAnimaLoRATraining,
    "AnimaTrainLoop": AnimaTrainLoop,
    "AnimaTrainLoRASave": AnimaTrainLoRASave,
    "AnimaTrainEnd": AnimaTrainEnd,
    "AnimaTrainValidationSettings": AnimaTrainValidationSettings,
    "AnimaTrainValidate": AnimaTrainValidate,
    "InitAnimaLLLiteTraining": InitAnimaLLLiteTraining,
    "AnimaLLLiteTrainSave": AnimaLLLiteTrainSave,
    "AnimaLLLiteTrainEnd": AnimaLLLiteTrainEnd,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaModelSelect": "Anima Model Select",
    "InitAnimaLoRATraining": "Init Anima LoRA Training",
    "AnimaTrainLoop": "Anima Train Loop",
    "AnimaTrainLoRASave": "Anima Train LoRA Save",
    "AnimaTrainEnd": "Anima Train End",
    "AnimaTrainValidationSettings": "Anima Train Validation Settings",
    "AnimaTrainValidate": "Anima Train Validate",
    "InitAnimaLLLiteTraining": "Init Anima LLLite Training",
    "AnimaLLLiteTrainSave": "Anima LLLite Train Save",
    "AnimaLLLiteTrainEnd": "Anima LLLite Train End",
}
