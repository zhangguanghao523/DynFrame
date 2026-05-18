import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from transformers import PreTrainedModel

from ..extras.interactive_ckpt import interactive_ckpt_enable
from ..extras.callbacks import CustomEarlyStoppingCallback, LogCallback, SaveTokenizerCallback, \
    InteractiveCkptCallback
from ..extras.logging import get_logger
from ..hparams import get_infer_args, get_train_args
from ..model import load_model_and_template
from .grpo import run_grpo
from .sft import run_sft


if TYPE_CHECKING:
    from transformers import TrainerCallback

from dotenv import load_dotenv
load_dotenv(dotenv_path="./.env.local")

logger = get_logger(__name__)

try:
    import ml_tracker.integration.transformers
    os.environ['ML_TRACKER_STEP'] = 'global_step'
except ImportError:
    logger.error("can not find MLTrackerCallback,skip it.")


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    training_args.remove_unused_columns = False  # for mllm and rm/kto dataset

    callbacks = [LogCallback()] if callbacks is None else callbacks

    # 开启交互式ckpt
    enable_success = interactive_ckpt_enable()
    if enable_success:
        # 交互式ckpt callbacks
        callbacks.append(InteractiveCkptCallback())

    # All current stages support callbacks
    callbacks.append(SaveTokenizerCallback())

    if finetuning_args.early_stopping_patience is not None:
        callbacks.append(
            CustomEarlyStoppingCallback(
                finetuning_args.early_stopping_patience,
                finetuning_args.early_stopping_threshold,
            )
        )

    if finetuning_args.stage == "sft":
        if finetuning_args.use_turbo:
            from .sft.workflow_turbo import run_sft_turbo
            run_sft_turbo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
        else:
            run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "grpo":
        run_grpo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    else:
        raise ValueError("Unknown task.")


def export_model(args: Optional[Dict[str, Any]] = None):
    model_args, data_args, finetuning_args, _ = get_infer_args(args)

    if model_args.export_dir is None:
        raise ValueError("Please specify `export_dir`.")

    if model_args.adapter_name_or_path is not None and model_args.export_quantization_bit is not None:
        raise ValueError("Please merge adapters before quantizing the model.")

    model, template = load_model_and_template(
        model_args, finetuning_args, template=data_args.template, add_valuehead=False
    )
    tokenizer, processor = template.tokenizer, template.processor

    if getattr(model, "quantization_method", None) and model_args.adapter_name_or_path is not None:
        raise ValueError("Cannot merge adapters to a quantized model.")

    if not isinstance(model, PreTrainedModel):
        raise ValueError("The model is not a `PreTrainedModel`, export aborted.")

    if getattr(model, "quantization_method", None):
        model = model.to("cpu")
    elif hasattr(model.config, "torch_dtype"):
        model = model.to(getattr(model.config, "torch_dtype")).to("cpu")
    else:
        model = model.to(torch.float16).to("cpu")
        setattr(model.config, "torch_dtype", torch.float16)

    model.save_pretrained(
        save_directory=model_args.export_dir,
        max_shard_size="{}GB".format(model_args.export_size),
        safe_serialization=(not model_args.export_legacy_format),
    )
    if model_args.export_hub_model_id is not None:
        model.push_to_hub(
            model_args.export_hub_model_id,
            token=model_args.hf_hub_token,
            max_shard_size="{}GB".format(model_args.export_size),
            safe_serialization=(not model_args.export_legacy_format),
        )

    try:
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        if processor is not None:
            processor.save_pretrained(model_args.export_dir)
        tokenizer.save_pretrained(model_args.export_dir)
        if model_args.export_hub_model_id is not None:
            tokenizer.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)
    except Exception:
        logger.warning("Cannot save tokenizer, please copy the files manually.")


if __name__ == "__main__":
    run_exp()
