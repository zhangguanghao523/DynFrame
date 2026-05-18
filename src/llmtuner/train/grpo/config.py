"""
Configuration mapping for GRPO training.

This module maps LLaMA-Factory's configuration system to TRL's GRPOConfig.
"""

from trl import GRPOConfig
from typing import TYPE_CHECKING, Optional
from ...extras.logging import get_logger

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments
    from ...hparams import ModelArguments, FinetuningArguments, GeneratingArguments

logger = get_logger(__name__)


def create_grpo_config(
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    model_args: "ModelArguments",
) -> GRPOConfig:
    """
    Map LLaMA-Factory configuration to TRL GRPOConfig.

    Args:
        training_args: Training arguments from LLaMA-Factory
        finetuning_args: Finetuning arguments with GRPO-specific parameters
        generating_args: Generation arguments
        model_args: Model arguments

    Returns:
        GRPOConfig instance configured for training
    """

    # Basic training parameters from training_args
    grpo_config_kwargs = {
        "output_dir": training_args.output_dir,
        "num_train_epochs": training_args.num_train_epochs,
        "max_steps": training_args.max_steps,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "learning_rate": training_args.learning_rate,
        "warmup_steps": training_args.warmup_steps,
        "warmup_ratio": training_args.warmup_ratio,
        "logging_steps": training_args.logging_steps,
        "save_steps": training_args.save_steps,
        "eval_steps": training_args.eval_steps,
        "save_total_limit": training_args.save_total_limit,
        "load_best_model_at_end": training_args.load_best_model_at_end,
        "metric_for_best_model": training_args.metric_for_best_model,
        "greater_is_better": training_args.greater_is_better,
        "seed": training_args.seed,
        "report_to": training_args.report_to,
        "run_name": training_args.run_name,
        "disable_tqdm": training_args.disable_tqdm,
        "log_level": training_args.log_level,
        "lr_scheduler_type": training_args.lr_scheduler_type,
        "lr_scheduler_kwargs": training_args.lr_scheduler_kwargs,
        "optim": training_args.optim,
        "optim_args": training_args.optim_args,
        "adam_beta1": training_args.adam_beta1,
        "adam_beta2": training_args.adam_beta2,
        "adam_epsilon": training_args.adam_epsilon,
        "weight_decay": training_args.weight_decay,
        "max_grad_norm": training_args.max_grad_norm,
        "gradient_checkpointing": training_args.gradient_checkpointing,
        "bf16": training_args.bf16,
        "fp16": training_args.fp16,
        "tf32": training_args.tf32,
        "dataloader_num_workers": training_args.dataloader_num_workers,
        "remove_unused_columns": training_args.remove_unused_columns,
    }

    # GRPO-specific parameters from finetuning_args
    num_generations = getattr(finetuning_args, 'grpo_num_generations', 8)
    grpo_config_kwargs.update({
        "num_generations": num_generations,
        "num_iterations": getattr(finetuning_args, 'grpo_num_iterations', 1),
        "beta": getattr(finetuning_args, 'grpo_beta', 0.0),  # No KL penalty by default
        "loss_type": getattr(finetuning_args, 'grpo_loss_type', 'dapo'),
        "epsilon": getattr(finetuning_args, 'grpo_epsilon', 0.2),
        "scale_rewards": getattr(finetuning_args, 'grpo_scale_rewards', 'group'),
        "mask_truncated_completions": getattr(finetuning_args, 'grpo_mask_truncated_completions', False),
        "multi_objective_aggregation": getattr(finetuning_args, 'multi_objective_aggregation', 'sum_then_normalize'),
        # NOTE: Do NOT set generation_batch_size here!
        # Let TRL auto-calculate it based on: per_device_batch_size * num_gpus * gradient_accumulation_steps
        # Setting it to num_generations causes errors when num_generations < global_batch_size

        # ZeRO-3 compatibility: gather model weights for generation
        # This is required for ZeRO Stage 3 to work correctly during generation
        # Without this, embedding weights won't be properly gathered, causing "'weight' must be 2-D" error
        "ds3_gather_for_generation": True,
    })

    # Generation parameters from generating_args
    grpo_config_kwargs.update({
        "temperature": getattr(generating_args, 'temperature', 1.0),
        "top_p": getattr(generating_args, 'top_p', 1.0),
        "top_k": getattr(generating_args, 'top_k', None),
        "max_completion_length": getattr(generating_args, 'max_new_tokens', 256),
        "repetition_penalty": getattr(generating_args, 'repetition_penalty', 1.0),
        # NOTE: DO NOT suppress video/vision tokens!
        # These tokens (<|vision_start|>, <|video_pad|>, <|vision_end|>) are part of the model's
        # learned format for fine-grained video understanding with dynamic timestamps.
        # When the model generates these tokens, it's "re-referencing" specific video frames
        # for detailed temporal understanding, which is essential for VMCOT reasoning.
    })

    # Note: reward_weights will be set separately in the trainer based on the reward functions

    # vLLM configuration if enabled
    use_vllm = getattr(model_args, 'use_vllm', False)
    if use_vllm:
        grpo_config_kwargs.update({
            "use_vllm": True,
            "vllm_mode": getattr(model_args, 'vllm_mode', 'server'),
            "vllm_server_host": getattr(model_args, 'vllm_server_host', '0.0.0.0'),
            "vllm_server_port": getattr(model_args, 'vllm_server_port', 8000),
            "vllm_gpu_memory_utilization": getattr(model_args, 'vllm_gpu_memory_utilization', 0.3),
        })

    # Create the config
    grpo_config = GRPOConfig(**grpo_config_kwargs)

    # Log configuration summary
    logger.info(f"Created GRPOConfig with:")
    logger.info(f"  - num_generations: {grpo_config.num_generations}")
    logger.info(f"  - loss_type: {grpo_config.loss_type}")
    logger.info(f"  - beta (KL coefficient): {grpo_config.beta}")
    logger.info(f"  - max_completion_length: {grpo_config.max_completion_length}")
    logger.info(f"  - use_vllm: {grpo_config.use_vllm}")
    logger.info(f"  - format_weight: {getattr(finetuning_args, 'grpo_format_weight', 0.3)}")
    logger.info(f"  - accuracy_weight: {getattr(finetuning_args, 'grpo_accuracy_weight', 0.7)}")
    logger.info(f"  - video_tokens: preserved (for fine-grained temporal understanding)")

    return grpo_config