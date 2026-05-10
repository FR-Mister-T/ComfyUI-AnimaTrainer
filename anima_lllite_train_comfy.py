import copy
import gc
import math
import os
from multiprocessing import Value
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from .library import (
    anima_train_utils,
    anima_utils,
    config_util,
    deepspeed_utils,
    flux_train_utils,
    qwen_image_autoencoder_kl,
    sd3_train_utils,
    strategy_anima,
    strategy_base,
    train_util,
)
from .library.config_util import BlueprintGenerator, ConfigSanitizer
from .library.utils import setup_logging
from .networks.control_net_lllite_anima import (
    LLLITE_ARCH_VERSION,
    AnimaControlNetLLLiteWrapper,
    ControlNetLLLiteDiT,
    load_lllite_weights,
    save_lllite_model,
)

setup_logging()
import logging

logger = logging.getLogger(__name__)


class AnimaLLLiteTrainer:
    def __init__(self):
        self.global_step = 0
        self.current_epoch = Value("i", 0)
        self.num_train_epochs = 0
        self.args = None
        self.accelerator = None
        self.wrapper = None
        self.lllite = None
        self.dit = None
        self.vae = None
        self.qwen3_text_encoder = None
        self.optimizer = None
        self.lr_scheduler = None
        self.optimizer_train_fn = None
        self.optimizer_eval_fn = None
        self.comfy_pbar = None
        self.metadata = {}
        self._train_dataloader = None
        self._noise_scheduler_copy = None
        self._tokenize_strategy = None
        self._text_encoding_strategy = None
        self._sample_prompts_te_outputs = None
        self._dit_weight_dtype = None
        self._weight_dtype = None
        self._save_dtype = None
        self._training_gen = None

    def init_train(self, args):
        self.args = args

        train_util.verify_training_args(args)
        train_util.prepare_dataset_args(args, True)
        deepspeed_utils.prepare_deepspeed_args(args)
        setup_logging(args, reset=True)

        if not args.skip_cache_check:
            args.skip_cache_check = args.skip_latents_validity_check

        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning("cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled")
            args.cache_text_encoder_outputs = True

        # Dataset — ControlNet sanitizer (controlnet=True)
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(False, False, True, True))
        if args.dataset_config is not None:
            user_config = config_util.load_user_config(args.dataset_config)
        else:
            user_config = {
                "datasets": [
                    {
                        "subsets": config_util.generate_controlnet_subsets_config_by_subdirs(
                            args.train_data_dir,
                            args.conditioning_data_dir,
                            args.caption_extension,
                        )
                    }
                ]
            }

        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, _ = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)

        current_epoch = self.current_epoch
        current_step = Value("i", 0)
        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)
        train_dataset_group.verify_bucket_reso_steps(16)

        # Accelerator
        accelerator = train_util.prepare_accelerator(args)
        self.accelerator = accelerator
        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        self._weight_dtype = weight_dtype
        self._save_dtype = save_dtype

        # Tokenizers
        logger.info("Loading tokenizers...")
        qwen3_text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=weight_dtype, device="cpu")
        t5_tokenizer = anima_utils.load_t5_tokenizer(args.t5_tokenizer_path)

        tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
            qwen3_tokenizer=qwen3_tokenizer,
            t5_tokenizer=t5_tokenizer,
            qwen3_max_length=args.qwen3_max_token_length,
            t5_max_length=args.t5_max_token_length,
        )
        strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
        self._tokenize_strategy = tokenize_strategy

        text_encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
        strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)
        self._text_encoding_strategy = text_encoding_strategy

        qwen3_text_encoder.to(weight_dtype)
        qwen3_text_encoder.requires_grad_(False)
        self.qwen3_text_encoder = qwen3_text_encoder

        # Latents caching strategy
        if args.cache_latents:
            latents_caching_strategy = strategy_anima.AnimaLatentsCachingStrategy(
                args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
            )
            strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

        # Text encoder output caching
        sample_prompts_te_outputs = None
        if args.cache_text_encoder_outputs:
            qwen3_text_encoder.to(accelerator.device)
            qwen3_text_encoder.eval()

            text_encoder_caching_strategy = strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check, False
            )
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_caching_strategy)

            with accelerator.autocast():
                train_dataset_group.new_cache_text_encoder_outputs([qwen3_text_encoder], accelerator)

            if args.sample_prompts is not None:
                prompts = args.sample_prompts if isinstance(args.sample_prompts, list) else [args.sample_prompts]
                sample_prompts_te_outputs = {}
                with accelerator.autocast(), torch.no_grad():
                    for p in prompts:
                        if p.strip() and p not in sample_prompts_te_outputs:
                            tokens_and_masks = tokenize_strategy.tokenize(p)
                            sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                tokenize_strategy, [qwen3_text_encoder], tokens_and_masks
                            )

            accelerator.wait_for_everyone()
            qwen3_text_encoder = None
            gc.collect()
            clean_memory_on_device(accelerator.device)

        self._sample_prompts_te_outputs = sample_prompts_te_outputs

        # VAE
        logger.info("Loading Anima VAE...")
        vae = qwen_image_autoencoder_kl.load_vae(
            args.vae,
            device="cpu",
            disable_mmap=True,
            spatial_chunk_size=getattr(args, "vae_chunk_size", 0),
            disable_cache=getattr(args, "vae_disable_cache", False),
        )
        self.vae = vae

        if args.cache_latents:
            vae.to(accelerator.device, dtype=weight_dtype)
            vae.requires_grad_(False)
            vae.eval()
            train_dataset_group.new_cache_latents(vae, accelerator)
            vae.to("cpu")
            clean_memory_on_device(accelerator.device)
            accelerator.wait_for_everyone()

        # DiT (frozen during LLLite training)
        logger.info("Loading Anima DiT (frozen)...")
        attn_mode = getattr(args, "attn_mode", "torch") or "torch"
        split_attn = getattr(args, "split_attn", False)
        dit = anima_utils.load_anima_model("cpu", args.pretrained_model_name_or_path, attn_mode, split_attn, "cpu", dit_weight_dtype=None)

        if args.gradient_checkpointing:
            dit.enable_gradient_checkpointing(cpu_offload=False, unsloth_offload=False)

        dit.requires_grad_(False)
        self.dit = dit

        # Build LLLite
        logger.info("Building ControlNet-LLLite (Anima)...")
        lllite = ControlNetLLLiteDiT(
            dit,
            cond_emb_dim=args.cond_emb_dim,
            mlp_dim=args.lllite_mlp_dim,
            target_layers=args.lllite_target_layers,
            dropout=getattr(args, "lllite_dropout", None),
            multiplier=args.lllite_multiplier,
            cond_dim=args.lllite_cond_dim,
            cond_resblocks=args.lllite_cond_resblocks,
            use_aspp=args.lllite_use_aspp,
        )

        if getattr(args, "network_weights", None):
            load_lllite_weights(lllite, args.network_weights, strict=False)

        lllite.apply_to()
        self.lllite = lllite

        wrapper = AnimaControlNetLLLiteWrapper(dit, lllite)
        self.wrapper = wrapper

        # Optimizer — only LLLite parameters are trained
        trainable_params = list(lllite.parameters())
        n_trainable = sum(p.numel() for p in trainable_params if p.requires_grad)
        accelerator.print(f"LLLite modules: {len(lllite.lllite_modules)}, trainable params: {n_trainable:,}")

        _, _, optimizer = train_util.get_optimizer(args, trainable_params=trainable_params)
        self.optimizer = optimizer
        self.optimizer_train_fn, self.optimizer_eval_fn = train_util.get_optimizer_train_eval_fn(optimizer, args)

        # Dataloader
        train_dataset_group.set_current_strategies()
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        train_dataset_group.set_max_train_steps(args.max_train_steps)
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)
        self.lr_scheduler = lr_scheduler

        # dtype setup
        dit_weight_dtype = weight_dtype
        if args.full_fp16:
            accelerator.print("enable full fp16 training.")
        elif args.full_bf16:
            accelerator.print("enable full bf16 training.")
        self._dit_weight_dtype = dit_weight_dtype

        dit.to(dit_weight_dtype)
        dit.to(accelerator.device)

        # LLLite trains in fp32 unless full_*16 is set
        lllite_dtype = torch.float32
        if args.full_fp16 or args.full_bf16:
            lllite_dtype = weight_dtype
        lllite.to(lllite_dtype)
        lllite.to(accelerator.device)

        if not args.cache_text_encoder_outputs and self.qwen3_text_encoder is not None:
            self.qwen3_text_encoder.to(accelerator.device)
        if not args.cache_latents:
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=weight_dtype)

        clean_memory_on_device(accelerator.device)

        wrapper, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            wrapper, optimizer, train_dataloader, lr_scheduler
        )
        self.wrapper = wrapper
        self.optimizer = optimizer
        self._train_dataloader = train_dataloader
        self.lr_scheduler = lr_scheduler

        if args.full_fp16:
            train_util.patch_accelerator_for_fp16_training(accelerator)

        train_util.resume_from_local_or_hf_if_specified(accelerator, args)

        # Epoch count
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        self.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # Noise scheduler
        noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
        self._noise_scheduler_copy = copy.deepcopy(noise_scheduler)

        if accelerator.is_main_process:
            accelerator.init_trackers("anima_controlnet_lllite", config=train_util.get_sanitized_config_or_none(args))

        self.metadata = {
            "ss_lllite_arch_version": LLLITE_ARCH_VERSION,
            "ss_cond_emb_dim": str(args.cond_emb_dim),
            "ss_lllite_mlp_dim": str(args.lllite_mlp_dim),
            "ss_lllite_target_layers": args.lllite_target_layers,
        }

        self._training_gen = self._make_training_generator(current_step)
        return self._training_loop_fn

    def _make_training_generator(self, current_step):
        accelerator = self.accelerator
        args = self.args

        for epoch in range(self.num_train_epochs):
            self.current_epoch.value = epoch + 1

            self.wrapper.train()
            if args.gradient_checkpointing:
                accelerator.unwrap_model(self.wrapper).dit.train()
            else:
                accelerator.unwrap_model(self.wrapper).dit.eval()

            for step, batch in enumerate(self._train_dataloader):
                current_step.value = self.global_step

                with accelerator.accumulate(self.wrapper):
                    # Latents
                    if "latents" in batch and batch["latents"] is not None:
                        latents = batch["latents"].to(accelerator.device, dtype=self._dit_weight_dtype)
                        if latents.ndim == 5:
                            latents = latents.squeeze(2)
                    else:
                        with torch.no_grad():
                            images = batch["images"].to(accelerator.device, dtype=self._weight_dtype)
                            latents = self.vae.encode_pixels_to_latents(images).to(accelerator.device, dtype=self._dit_weight_dtype)
                        if torch.any(torch.isnan(latents)):
                            latents = torch.nan_to_num(latents, 0, out=latents)

                    # Text encoder outputs
                    text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
                    if text_encoder_outputs_list is not None:
                        caption_dropout_rates = text_encoder_outputs_list[-1]
                        text_encoder_outputs_list = text_encoder_outputs_list[:-1]
                        text_encoder_outputs_list = self._text_encoding_strategy.drop_cached_text_encoder_outputs(
                            *text_encoder_outputs_list, caption_dropout_rates=caption_dropout_rates
                        )
                        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoder_outputs_list
                    else:
                        with torch.no_grad():
                            prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = self._text_encoding_strategy.encode_tokens(
                                self._tokenize_strategy, [self.qwen3_text_encoder], batch["input_ids_list"]
                            )

                    prompt_embeds = prompt_embeds.to(accelerator.device, dtype=self._dit_weight_dtype)
                    attn_mask = attn_mask.to(accelerator.device)
                    t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
                    t5_attn_mask = t5_attn_mask.to(accelerator.device)

                    # Noise + timesteps
                    noise = torch.randn_like(latents)
                    noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
                        args, self._noise_scheduler_copy, latents, noise, accelerator.device, self._dit_weight_dtype
                    )
                    timesteps = timesteps / 1000.0
                    if torch.any(torch.isnan(noisy_model_input)):
                        noisy_model_input = torch.nan_to_num(noisy_model_input, 0, out=noisy_model_input)

                    bs = latents.shape[0]
                    h_latent, w_latent = latents.shape[-2], latents.shape[-1]
                    padding_mask = torch.zeros(bs, 1, h_latent, w_latent, dtype=self._dit_weight_dtype, device=accelerator.device)

                    cond_image = batch["conditioning_images"].to(accelerator.device, dtype=self._dit_weight_dtype)
                    noisy_model_input = noisy_model_input.unsqueeze(2)

                    with accelerator.autocast():
                        model_pred = self.wrapper(
                            noisy_model_input,
                            timesteps,
                            prompt_embeds,
                            cond_image=cond_image,
                            padding_mask=padding_mask,
                            source_attention_mask=attn_mask,
                            t5_input_ids=t5_input_ids,
                            t5_attn_mask=t5_attn_mask,
                        )
                    model_pred = model_pred.squeeze(2)

                    target = noise - latents
                    weighting = anima_train_utils.compute_loss_weighting_for_anima(
                        weighting_scheme=args.weighting_scheme, sigmas=sigmas
                    )
                    huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, None)
                    loss = train_util.conditional_loss(model_pred.float(), target.float(), args.loss_type, "none", huber_c)
                    loss = loss.mean([1, 2, 3])

                    if weighting is not None:
                        loss = loss * weighting

                    loss = (loss * batch["loss_weights"]).mean()

                    accelerator.backward(loss)

                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = list(accelerator.unwrap_model(self.wrapper).lllite.parameters())
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                accelerator.unwrap_model(self.wrapper).lllite.clear_cond_image()

                if accelerator.sync_gradients:
                    self.global_step += 1
                    if self.comfy_pbar is not None:
                        self.comfy_pbar.update(1)

                    current_loss = loss.detach().item()
                    if len(accelerator.trackers) > 0:
                        accelerator.log(
                            {"loss": current_loss, "lr": self.lr_scheduler.get_last_lr()[0]},
                            step=self.global_step,
                        )

                    yield

                    if self.global_step >= args.max_train_steps:
                        return

            if len(accelerator.trackers) > 0:
                accelerator.log({"epoch": epoch + 1}, step=self.global_step)

    def _training_loop_fn(self, break_at_steps, epoch):
        while self.global_step < break_at_steps:
            try:
                next(self._training_gen)
                if self.global_step >= self.args.max_train_steps:
                    break
            except StopIteration:
                break
        return self.global_step

    def save_model(self, ckpt_name, global_step, epoch, force_sync_upload=False):
        save_path = os.path.join(self.args.output_dir, ckpt_name)
        unwrapped_lllite = self.accelerator.unwrap_model(self.wrapper).lllite

        sai_metadata = train_util.get_sai_model_spec(None, self.args, False, False, False, is_stable_diffusion_ckpt=True)
        sai_metadata["modelspec.architecture"] = "anima-preview/control-net-lllite"
        sai_metadata["lllite.version"] = LLLITE_ARCH_VERSION
        sai_metadata["lllite.cond_emb_dim"] = str(self.args.cond_emb_dim)
        sai_metadata["lllite.mlp_dim"] = str(self.args.lllite_mlp_dim)
        sai_metadata["lllite.target_layers"] = self.args.lllite_target_layers
        sai_metadata["lllite.target_atomics"] = unwrapped_lllite.target_atomics_str
        sai_metadata["lllite.cond_dim"] = str(self.args.lllite_cond_dim)
        sai_metadata["lllite.cond_resblocks"] = str(self.args.lllite_cond_resblocks)
        sai_metadata["lllite.use_aspp"] = "true" if self.args.lllite_use_aspp else "false"
        if self.args.lllite_use_aspp:
            sai_metadata["lllite.aspp_dilations"] = ",".join(str(d) for d in unwrapped_lllite.aspp_dilations)

        save_lllite_model(save_path, unwrapped_lllite, dtype=self._save_dtype, metadata=sai_metadata)
        logger.info(f"LLLite checkpoint saved: {save_path}")

    def sample_images(self, epoch, global_step, validation_settings):
        qwen3_te = self.qwen3_text_encoder if not self.args.cache_text_encoder_outputs else None

        save_dir = os.path.join(self.args.output_dir, "sample")
        existing_files = set(os.listdir(save_dir)) if os.path.exists(save_dir) else set()

        anima_train_utils.sample_images(
            self.accelerator,
            self.args,
            epoch,
            global_step,
            self.dit,
            self.vae,
            qwen3_te,
            self._tokenize_strategy,
            self._text_encoding_strategy,
            self._sample_prompts_te_outputs,
            force=True,
            validation_settings=validation_settings,
        )

        clean_memory_on_device(self.accelerator.device)

        if not os.path.exists(save_dir):
            return None

        new_files = sorted(
            [f for f in os.listdir(save_dir) if f not in existing_files and f.endswith(".png")],
            key=lambda f: os.path.getmtime(os.path.join(save_dir, f)),
        )
        if not new_files:
            return None

        images = []
        for fname in new_files:
            img = Image.open(os.path.join(save_dir, fname)).convert("RGB")
            images.append(torch.from_numpy(np.array(img)).float() / 255.0)

        return torch.stack(images)
