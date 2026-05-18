"""
GRPO trainer adapter for LLaMA-Factory.

This module provides an adapter to integrate TRL's GRPOTrainer with LLaMA-Factory's
training pipeline, handling configuration mapping and custom reward functions.
"""

from typing import TYPE_CHECKING, Optional, List, Dict, Any, Callable
import os
import torch
from trl import GRPOTrainer
from transformers import PreTrainedModel, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from .config import create_grpo_config
from .rewards import create_reward_functions, get_reward_weights
from .video_aware_trainer import VideoAwareGRPOTrainer
from ...extras.logging import get_logger
from ...extras.ploting import plot_loss

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase, Seq2SeqTrainingArguments
    from datasets import Dataset
    from ...hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments

logger = get_logger(__name__)


class LLamaFactoryGRPOTrainer:
    """
    Adapter to integrate TRL's GRPOTrainer with LLaMA-Factory.

    This class wraps TRL's GRPOTrainer to provide compatibility with LLaMA-Factory's
    training infrastructure while leveraging TRL's efficient GRPO implementation.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        tokenizer: "PreTrainedTokenizerBase",
        train_dataset: "Dataset",
        eval_dataset: Optional["Dataset"] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        processor: Optional[Any] = None,
        ref_model: Optional[PreTrainedModel] = None,
    ):
        """
        Initialize the GRPO trainer adapter.

        Args:
            model: The model to train
            model_args: Model arguments from LLaMA-Factory
            data_args: Data arguments from LLaMA-Factory
            training_args: Training arguments from LLaMA-Factory
            finetuning_args: Finetuning arguments with GRPO-specific parameters
            generating_args: Generation arguments
            tokenizer: Tokenizer/processor for the model
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            callbacks: Optional list of trainer callbacks
        """
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.finetuning_args = finetuning_args
        self.generating_args = generating_args

        # Create GRPO config from LLaMA-Factory arguments
        grpo_config = create_grpo_config(
            training_args,
            finetuning_args,
            generating_args,
            model_args
        )

        # Create separate reward functions and weights
        reward_funcs = create_reward_functions(finetuning_args)
        reward_weights = get_reward_weights(finetuning_args)

        # Set reward weights in the config
        grpo_config.reward_weights = reward_weights

        # Initialize VideoAwareGRPOTrainer with custom reward functions
        # This trainer extends TRL's GRPOTrainer to handle video inputs
        # 🎯 传入 ref_model 以跳过 TRL 的自动创建（避免 transformers 找不到自定义类的错误）
        self.trainer = VideoAwareGRPOTrainer(
            model=model,
            ref_model=ref_model,  # 传入我们加载的 ref_model（与训练模型使用相同的类）
            reward_funcs=reward_funcs,  # List of separate reward functions
            args=grpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
        )

        # Store the processor and template for video handling
        if processor is not None:
            self.trainer.processor = processor
        # Pass the template for proper video encoding
        from ...model import load_template
        template = load_template(model_args, template=data_args.template)
        self.trainer.template = template

        logger.info("Initialized LLaMA-Factory GRPO trainer adapter")

    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        Start GRPO training.

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        logger.info("Starting GRPO training...")

        # Train using TRL's GRPOTrainer
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Log metrics
        if train_result.metrics:
            logger.info(f"Training metrics: {train_result.metrics}")

        return train_result

    def save_model(self, output_dir: Optional[str] = None):
        """
        Save the trained model.

        Args:
            output_dir: Directory to save the model (defaults to training_args.output_dir)
        """
        if output_dir is None:
            output_dir = self.training_args.output_dir

        logger.info(f"Saving model to {output_dir}")

        # Save using TRL's method
        self.trainer.save_model(output_dir)

        # Save tokenizer
        if hasattr(self.trainer, "processing_class") and self.trainer.processing_class is not None:
            self.trainer.processing_class.save_pretrained(output_dir)

    def save_state(self):
        """Save trainer state."""
        self.trainer.save_state()

    def evaluate(self, eval_dataset: Optional["Dataset"] = None):
        """
        Evaluate the model.

        Args:
            eval_dataset: Optional evaluation dataset

        Returns:
            Evaluation metrics
        """
        if eval_dataset is None and self.trainer.eval_dataset is None:
            logger.warning("No evaluation dataset provided")
            return {}

        return self.trainer.evaluate(eval_dataset=eval_dataset)

    def is_world_process_zero(self) -> bool:
        """Check if this is the main process."""
        return self.trainer.accelerator.is_main_process

    def get_output_dir(self) -> str:
        """Get the output directory path."""
        return self.training_args.output_dir

    def plot_loss(self):
        """Plot training loss if enabled."""
        if self.is_world_process_zero() and self.finetuning_args.plot_loss:
            plot_loss(self.training_args.output_dir, keys=["loss", "reward"])

    @property
    def state(self):
        """Get trainer state."""
        return self.trainer.state

    @property
    def model(self):
        """Get the model being trained."""
        return self.trainer.model

    @property
    def tokenizer(self):
        """Get the tokenizer/processor."""
        return self.trainer.processing_class