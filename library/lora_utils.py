import os
import re
from typing import Dict, List, Optional, Union
import torch
from tqdm import tqdm
from .device_utils import synchronize_device
from .fp8_optimization_utils import load_safetensors_with_fp8_optimization
from .safetensors_utils import MemoryEfficientSafeOpen, TensorWeightAdapter, WeightTransformHooks, get_split_weight_filenames

try:
    from ..networks.loha import merge_weights_to_tensor as loha_merge
except ImportError:
    loha_merge = None

try:
    from ..networks.lokr import merge_weights_to_tensor as lokr_merge
except ImportError:
    lokr_merge = None

from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def filter_lora_state_dict(
    weights_sd: Dict[str, torch.Tensor],
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    original_key_count = len(weights_sd.keys())
    if include_pattern is not None:
        regex_include = re.compile(include_pattern)
        weights_sd = {k: v for k, v in weights_sd.items() if regex_include.search(k)}
        logger.info(f"Filtered keys with include pattern {include_pattern}: {original_key_count} -> {len(weights_sd.keys())}")

    if exclude_pattern is not None:
        original_key_count_ex = len(weights_sd.keys())
        regex_exclude = re.compile(exclude_pattern)
        weights_sd = {k: v for k, v in weights_sd.items() if not regex_exclude.search(k)}
        logger.info(f"Filtered keys with exclude pattern {exclude_pattern}: {original_key_count_ex} -> {len(weights_sd.keys())}")

    if len(weights_sd) != original_key_count:
        remaining_keys = list(set([k.split(".", 1)[0] for k in weights_sd.keys()]))
        remaining_keys.sort()
        logger.info(f"Remaining LoRA modules after filtering: {remaining_keys}")
        if len(weights_sd) == 0:
            logger.warning("No keys left after filtering.")

    return weights_sd


def load_safetensors_with_lora_and_fp8(
    model_files: Union[str, List[str]],
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]],
    lora_multipliers: Optional[List[float]],
    fp8_optimization: bool,
    calc_device: torch.device,
    move_to_device: bool = False,
    dit_weight_dtype: Optional[torch.dtype] = None,
    target_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
) -> dict[str, torch.Tensor]:
    """
    Merge LoRA weights into the state dict of a model with fp8 optimization if needed.
    """
    if isinstance(model_files, str):
        model_files = [model_files]

    extended_model_files = []
    for model_file in model_files:
        split_filenames = get_split_weight_filenames(model_file)
        if split_filenames is not None:
            extended_model_files.extend(split_filenames)
        else:
            extended_model_files.append(model_file)
    model_files = extended_model_files
    logger.info(f"Loading model files: {model_files}")

    weight_hook = None
    if lora_weights_list is None or len(lora_weights_list) == 0:
        lora_weights_list = []
        lora_multipliers = []
        list_of_lora_weight_keys = []
    else:
        list_of_lora_weight_keys = []
        for lora_sd in lora_weights_list:
            lora_weight_keys = set(lora_sd.keys())
            list_of_lora_weight_keys.append(lora_weight_keys)

        if lora_multipliers is None:
            lora_multipliers = [1.0] * len(lora_weights_list)
        while len(lora_multipliers) < len(lora_weights_list):
            lora_multipliers.append(1.0)
        if len(lora_multipliers) > len(lora_weights_list):
            lora_multipliers = lora_multipliers[: len(lora_weights_list)]

        logger.info(f"Merging LoRA weights into state dict. multipliers: {lora_multipliers}")

        def weight_hook_func(model_weight_key, model_weight: torch.Tensor, keep_on_calc_device=False):
            nonlocal list_of_lora_weight_keys, lora_weights_list, lora_multipliers, calc_device

            if not model_weight_key.endswith(".weight"):
                return model_weight

            original_device = model_weight.device
            if original_device != calc_device:
                model_weight = model_weight.to(calc_device)

            for lora_weight_keys, lora_sd, multiplier in zip(list_of_lora_weight_keys, lora_weights_list, lora_multipliers):
                lora_name_without_prefix = model_weight_key.rsplit(".", 1)[0]
                found = False
                for prefix in ["lora_unet_", ""]:
                    lora_name = prefix + lora_name_without_prefix.replace(".", "_")
                    down_key = lora_name + ".lora_down.weight"
                    up_key = lora_name + ".lora_up.weight"
                    alpha_key = lora_name + ".alpha"
                    if down_key in lora_weight_keys and up_key in lora_weight_keys:
                        found = True
                        break

                if found:
                    down_weight = lora_sd[down_key]
                    up_weight = lora_sd[up_key]

                    dim = down_weight.size()[0]
                    alpha = lora_sd.get(alpha_key, dim)
                    scale = alpha / dim

                    down_weight = down_weight.to(calc_device)
                    up_weight = up_weight.to(calc_device)

                    original_dtype = model_weight.dtype
                    if original_dtype.itemsize == 1:
                        model_weight = model_weight.to(torch.float16)
                        down_weight = down_weight.to(torch.float16)
                        up_weight = up_weight.to(torch.float16)

                    if len(model_weight.size()) == 2:
                        if len(up_weight.size()) == 4:
                            up_weight = up_weight.squeeze(3).squeeze(2)
                            down_weight = down_weight.squeeze(3).squeeze(2)
                        model_weight = model_weight + multiplier * (up_weight @ down_weight) * scale
                    elif down_weight.size()[2:4] == (1, 1):
                        model_weight = (
                            model_weight
                            + multiplier
                            * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                            * scale
                        )
                    else:
                        conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                        model_weight = model_weight + multiplier * conved * scale

                    if original_dtype.itemsize == 1:
                        model_weight = model_weight.to(original_dtype)

                    lora_weight_keys.remove(down_key)
                    lora_weight_keys.remove(up_key)
                    if alpha_key in lora_weight_keys:
                        lora_weight_keys.remove(alpha_key)
                    continue

                # Check for LoHa/LoKr weights
                for prefix in ["lora_unet_", ""]:
                    lora_name = prefix + lora_name_without_prefix.replace(".", "_")
                    hada_key = lora_name + ".hada_w1_a"
                    lokr_key = lora_name + ".lokr_w1"

                    if hada_key in lora_weight_keys:
                        if loha_merge is None:
                            raise ImportError("LoHa merge requested but networks.loha is not available")
                        model_weight = loha_merge(model_weight, lora_name, lora_sd, lora_weight_keys, multiplier, calc_device)
                        break
                    elif lokr_key in lora_weight_keys:
                        if lokr_merge is None:
                            raise ImportError("LoKr merge requested but networks.lokr is not available")
                        model_weight = lokr_merge(model_weight, lora_name, lora_sd, lora_weight_keys, multiplier, calc_device)
                        break

            if not keep_on_calc_device and original_device != calc_device:
                model_weight = model_weight.to(original_device)
            return model_weight

        weight_hook = weight_hook_func

    state_dict = load_safetensors_with_fp8_optimization_and_hook(
        model_files,
        fp8_optimization,
        calc_device,
        move_to_device,
        dit_weight_dtype,
        target_keys,
        exclude_keys,
        weight_hook=weight_hook,
        disable_numpy_memmap=disable_numpy_memmap,
        weight_transform_hooks=weight_transform_hooks,
    )

    for lora_weight_keys in list_of_lora_weight_keys:
        if len(lora_weight_keys) > 0:
            logger.warning(f"Warning: not all LoRA keys are used: {', '.join(lora_weight_keys)}")

    return state_dict


def load_safetensors_with_fp8_optimization_and_hook(
    model_files: list[str],
    fp8_optimization: bool,
    calc_device: torch.device,
    move_to_device: bool = False,
    dit_weight_dtype: Optional[torch.dtype] = None,
    target_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
    weight_hook: callable = None,
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
) -> dict[str, torch.Tensor]:
    """Load state dict from safetensors files with optional fp8 optimization and weight hook."""
    if fp8_optimization:
        logger.info(
            f"Loading state dict with FP8 optimization. Dtype of weight: {dit_weight_dtype}, hook enabled: {weight_hook is not None}"
        )
        state_dict = load_safetensors_with_fp8_optimization(
            model_files,
            calc_device,
            target_keys,
            exclude_keys,
            move_to_device=move_to_device,
            weight_hook=weight_hook,
            disable_numpy_memmap=disable_numpy_memmap,
            weight_transform_hooks=weight_transform_hooks,
        )
    else:
        logger.info(
            f"Loading state dict without FP8 optimization. Dtype of weight: {dit_weight_dtype}, hook enabled: {weight_hook is not None}"
        )
        state_dict = {}
        for model_file in model_files:
            with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as original_f:
                f = TensorWeightAdapter(weight_transform_hooks, original_f) if weight_transform_hooks is not None else original_f
                for key in tqdm(f.keys(), desc=f"Loading {os.path.basename(model_file)}", leave=False):
                    if weight_hook is None and move_to_device:
                        value = f.get_tensor(key, device=calc_device, dtype=dit_weight_dtype)
                    else:
                        value = f.get_tensor(key)
                        if weight_hook is not None:
                            value = weight_hook(key, value, keep_on_calc_device=move_to_device)
                        if move_to_device:
                            value = value.to(calc_device, dtype=dit_weight_dtype, non_blocking=True)
                        elif dit_weight_dtype is not None:
                            value = value.to(dit_weight_dtype)

                    state_dict[key] = value
        if move_to_device:
            synchronize_device(calc_device)

    return state_dict
