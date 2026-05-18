import os
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
from transformers.utils import is_safetensors_available

from ..extras.constants import IGNORE_INDEX, V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.logging import get_logger
from ..hparams import FinetuningArguments, ModelArguments
from ..model import load_model_and_tokenizer, load_valuehead_params


if is_safetensors_available():
    from safetensors.torch import save_file

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, Trainer
    from transformers.modeling_utils import PreTrainedModel
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import DataArguments


logger = get_logger(__name__)


def create_modelcard_and_push(
    trainer: "Trainer",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    return
    kwargs = {
        "tasks": "text-generation",
        "finetuned_from": model_args.model_name_or_path,
        "dataset": [dataset.strip() for dataset in data_args.dataset.split(",")],
        "tags": ["llama-factory", finetuning_args.finetuning_type],
    }
    if not training_args.do_train:
        pass
    elif training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(license="other", **kwargs)  # prevent from connecting to hub


def create_ref_model(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    data_args: "DataArguments",
    add_valuehead: Optional[bool] = False
) -> Union["PreTrainedModel", "AutoModelForCausalLMWithValueHead"]:
    r"""
    Creates reference model for PPO training. Evaluation mode is not supported.

    The valuehead parameter is randomly initialized since it is useless for PPO training.
    """
    if finetuning_args.ref_model is not None:
        ref_model_args = ModelArguments.copyfrom(
            model_args,
            model_name_or_path=finetuning_args.ref_model,
            adapter_name_or_path=finetuning_args.ref_model_adapters,
            quantization_bit=finetuning_args.ref_model_quantization_bit,
        )
        ref_finetuning_args = FinetuningArguments(finetuning_type="lora")
        ref_model, *_ = load_model_and_tokenizer(
            ref_model_args, ref_finetuning_args, is_trainable=False, add_valuehead=add_valuehead
        )
        logger.info("Created reference model from {}".format(finetuning_args.ref_model))
    else:
        if finetuning_args.finetuning_type == "lora":
            ref_model = None
        else:
            ref_model_args = ModelArguments.copyfrom(model_args)
            ref_finetuning_args = FinetuningArguments()
            ref_model, *_ = load_model_and_tokenizer(
                ref_model_args, ref_finetuning_args, is_trainable=False, add_valuehead=add_valuehead
            )
            logger.info("Created reference model from the model itself.")

    return ref_model


def get_batch_logps(
    logits: "torch.Tensor", labels: "torch.Tensor", label_pad_token_id: int = IGNORE_INDEX, per_token: bool = False
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    r"""
    Computes the log probabilities of the given labels under the given logits.

    Returns:
        logps: A tensor of shape (batch_size,) containing the sum of log probabilities.
        valid_length: A tensor of shape (batch_size,) containing the number of non-masked tokens.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batchsize x seqlen) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id
    labels[labels == label_pad_token_id] = 0  # dummy token
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    if per_token:
        return per_token_logps * loss_mask, loss_mask
    return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)


def save_trl_vhead_model(
    model: "AutoModelForCausalLMWithValueHead",
    output_dir: str,
    state_dict=None,
    dup_head_weight: bool = True,
    save_safetensors: bool = False,
):
    if state_dict is None:
        state_dict = model.state_dict()
    fixed_state_dict = {}
    vhead_state_dict = {}
    # fix zero3 wrapped pretrained model
    for name, param in state_dict.items():
        if name.startswith("v_head."):
            vhead_state_dict[name] = param
            if not dup_head_weight:
                continue
        fixed_state_dict[name.replace("pretrained_model.", "", 1)] = param

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving model checkpoint to {output_dir}")
    if model.is_peft_model:
        fixed_state_dict = None  # not override the state_dict of peft model
    model.pretrained_model.save_pretrained(
        output_dir, state_dict=fixed_state_dict, safe_serialization=save_safetensors
    )
    if save_safetensors:
        save_file(vhead_state_dict, os.path.join(output_dir, V_HEAD_SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
    else:
        torch.save(vhead_state_dict, os.path.join(output_dir, V_HEAD_WEIGHTS_NAME))
    logger.info("Value head model saved at: {}".format(output_dir))