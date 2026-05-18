import contextlib
import gc
import os
import types
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
from transformers import InfNanRemoveLogitsProcessor, LogitsProcessorList
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import (
    is_torch_bf16_gpu_available,
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)

from .logging import get_logger


_is_fp16_available = is_torch_npu_available() or is_torch_cuda_available()
try:
    _is_bf16_available = is_torch_bf16_gpu_available()
except Exception:
    _is_bf16_available = False


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments

    from llmtuner.hparams import ModelArguments


logger = get_logger(__name__)


class AverageMeter:
    r"""
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def get_current_device() -> torch.device:
    r"""
    Gets the current available device.
    """
    if is_torch_xpu_available():
        device = "xpu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_npu_available():
        device = "npu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_mps_available():
        device = "mps:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_cuda_available():
        device = "cuda:{}".format(os.environ.get("LOCAL_RANK", "0"))
    else:
        device = "cpu"

    return torch.device(device)


def get_device_count() -> int:
    return torch.cuda.device_count()


def get_logits_processor() -> "LogitsProcessorList":
    r"""
    Gets logits processor that removes NaN and Inf logits.
    """
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    return logits_processor


def infer_optim_dtype(model_dtype: torch.dtype) -> torch.dtype:
    r"""
    Infers the optimal dtype according to the model_dtype and device compatibility.
    """
    if _is_bf16_available and model_dtype == torch.bfloat16:
        return torch.bfloat16
    elif _is_fp16_available:
        return torch.float16
    else:
        return torch.float32


def torch_gc() -> None:
    r"""
    Collects GPU memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def try_download_model_from_ms(model_args: "ModelArguments") -> None:
    if not use_modelscope() or os.path.exists(model_args.model_name_or_path):
        return

    try:
        from modelscope import snapshot_download

        revision = "master" if model_args.model_revision == "main" else model_args.model_revision
        model_args.model_name_or_path = snapshot_download(
            model_args.model_name_or_path, revision=revision, cache_dir=model_args.cache_dir
        )
    except ImportError:
        raise ImportError("Please install modelscope via `pip install modelscope -U`")


def use_modelscope() -> bool:
    return bool(int(os.environ.get("USE_MODELSCOPE_HUB", "0")))


def get_resume_checkpoint(training_args: "Seq2SeqTrainingArguments") -> Optional[str]:
    resume_from_checkpoint = training_args.resume_from_checkpoint
    if isinstance(resume_from_checkpoint, str) and os.path.exists(resume_from_checkpoint):
        return resume_from_checkpoint

    if (
        (resume_from_checkpoint is None or resume_from_checkpoint is True)
        and not training_args.overwrite_output_dir
        and os.path.exists(training_args.output_dir)
    ):
        resume_from_checkpoint = get_last_checkpoint(training_args.output_dir)

    if resume_from_checkpoint is None:
        logger.warning("No valid checkpoint found in output directory. New training will be started.")
    else:
        logger.warning(
            f"Resume training from checkpoint: {resume_from_checkpoint}. "
            "Make sure the checkpoint is compatible with current training arguments!"
        )
    return resume_from_checkpoint


def patch_training_args(training_args: "Seq2SeqTrainingArguments"):
    @contextlib.contextmanager
    def _main_process_first(self, local=True, desc="work"):
        import torch.distributed as dist
        from transformers.utils import is_sagemaker_mp_enabled, is_torch_available, is_torch_xla_available

        if is_torch_xla_available():
            import torch_xla.core.xla_model as xm
        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp

        if is_torch_available() and self.world_size > 1:
            main_process_desc = "main local process" if local else "main process"
            if self.distributed_state is not None:
                local_rank = int(os.getenv("LOCAL_PROCESS_RANK", os.getenv("LOCAL_RANK", "0")))
                is_main_process = local_rank == 0 if local else self.distributed_state.is_main_process
            elif is_sagemaker_mp_enabled():
                is_main_process = smp.rank() == 0
            logger.warning(f"rank{dist.get_rank()} {is_main_process}: {main_process_desc} performing {desc}")
            try:
                if not is_main_process:
                    # tell all replicas to wait
                    logger.debug(f"{self.process_index}: waiting for the {main_process_desc} to perform {desc}")

                    if is_torch_xla_available():
                        xm.rendezvous(desc)
                    else:
                        dist.barrier()
                yield
            finally:
                if is_main_process:
                    # the wait is over
                    logger.debug(f"{self.process_index}: {main_process_desc} completed {desc}, releasing all replicas")
                    if is_torch_xla_available():
                        xm.rendezvous(desc)
                    else:
                        dist.barrier()
        else:
            yield

    training_args.main_process_first = types.MethodType(_main_process_first, training_args)

def is_float_tensor(tensor):
    if not isinstance(tensor, torch.Tensor):
        return False
    return tensor.dtype in (torch.float, torch.float16, torch.bfloat16)


def recursive_converter(converter, value):
    if isinstance(value, list):
        new_value = []
        for v in value:
            new_value += [recursive_converter(converter, v)]
        return new_value
    else:
        return converter(value)


def to_device(data: Dict, device: torch.device):
    def cast_tensor(v):
        if isinstance(v, torch.Tensor):
            return v.to(device=device)
        else:
            return v

    new_data = {}
    for k, v in data.items():
        new_data[k] = recursive_converter(cast_tensor, v)
    return new_data


