"""
GRPO workflow integration for LLaMA-Factory.

This module provides the main entry point for GRPO training within the LLaMA-Factory
training pipeline.
"""

from typing import TYPE_CHECKING, List, Optional
from transformers import DataCollatorWithPadding, TrainerCallback

from ...data import get_dataset
from ...extras.callbacks import SaveProcessorCallback
from ...model import load_model, load_template
from .trainer import LLamaFactoryGRPOTrainer
from ...extras.logging import get_logger

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments
    from ...hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments

logger = get_logger(__name__)


class HighPrecisionLossCallback(TrainerCallback):
    """
    Callback to display loss with higher precision (6 decimal places).

    GRPO loss values are often very small (close to 0) due to:
    1. The GRPO/DAPO loss formula using ratio clipping
    2. Mixed positive/negative advantages that cancel out

    This callback ensures we can see the actual loss values instead of rounded 0.0.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.is_world_process_zero:
            # Format numeric values with high precision
            formatted = {}
            for k, v in logs.items():
                if isinstance(v, float):
                    # Use 6 decimal places for better visibility
                    formatted[k] = f"{v:.6f}"
                else:
                    formatted[k] = v

            # Print high-precision log
            print(f"📊 [HIGH PRECISION] Step {state.global_step}: {formatted}")


def run_grpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    """
    Run GRPO training integrated with LLaMA-Factory.

    Args:
        model_args: Model configuration arguments
        data_args: Data configuration arguments
        training_args: Training configuration arguments
        finetuning_args: Finetuning configuration arguments
        generating_args: Generation configuration arguments
        callbacks: Optional list of trainer callbacks
    """
    logger.info("Starting GRPO training workflow")

    # Load template and tokenizer
    template = load_template(
        model_args,
        template=data_args.template,
        enable_thinking=data_args.enable_thinking
    )
    tokenizer, processor = template.tokenizer, template.processor

    # Set padding side for generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning("Pad token not set, using eos_token as pad_token")

    # Load dataset for GRPO
    # GRPO needs raw text prompts, not tokenized data
    # So we'll load the dataset directly without tokenization
    from ...data.parser import get_dataset_list
    from datasets import load_dataset

    dataset_list = get_dataset_list(data_args)
    datasets_to_concatenate = []
    print("DEBUG: ")
    logger.info(f"Loading {len(dataset_list)} datasets for GRPO training")

    for dataset_attr in dataset_list:
        logger.info(f"Loading dataset: {dataset_attr}")
        try:
            if dataset_attr.load_from == "file":
                logger.info(f"Loading from file: {dataset_attr.dataset_name}")
                dataset = load_dataset(
                    "json",
                    data_files=dataset_attr.dataset_name,
                    split="train",
                    num_proc=data_args.preprocessing_num_workers,
                )
            else:
                # Handle other dataset loading methods if needed
                logger.info(f"Loading from hub: {dataset_attr.dataset_name}")
                dataset = load_dataset(dataset_attr.dataset_name, split="train")

            logger.info(f"Dataset columns: {dataset.column_names}")
            logger.info(f"Dataset size: {len(dataset)}")

            # Log if video column is present
            if "video" in dataset.column_names:
                logger.info(f"Video column detected in dataset")

            # Ensure the dataset has the required fields
            if "prompt" in dataset.column_names and "answer" in dataset.column_names:
                datasets_to_concatenate.append(dataset)
                logger.info(f"Added dataset with {len(dataset)} examples")
            else:
                logger.warning(f"Dataset {dataset_attr.dataset_name} missing required fields: prompt={('prompt' in dataset.column_names)}, answer={('answer' in dataset.column_names)}")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_attr.dataset_name}: {e}")

    if len(datasets_to_concatenate) == 0:
        raise ValueError("No valid datasets found for GRPO training")

    from datasets import concatenate_datasets
    train_dataset = concatenate_datasets(datasets_to_concatenate)

    # Create the dataset module format expected by the trainer
    dataset_module = {
        "train_dataset": train_dataset
    }

    # GRPO Fix: Ensure we use the enhanced Qwen3VL model for video processing
    # The enhanced model properly handles video tokens and attention masks
    if hasattr(model_args, 'enable_dynamic_video'):
        logger.info(f"Current enable_dynamic_video setting: {model_args.enable_dynamic_video}")
        if not model_args.enable_dynamic_video:
            logger.warning("enable_dynamic_video was False, setting to True for GRPO video training")
            model_args.enable_dynamic_video = True
    else:
        logger.info("enable_dynamic_video not found in model_args, setting to True")
        model_args.enable_dynamic_video = True

    # Load model (no value head needed for GRPO)
    model = load_model(
        tokenizer,
        model_args,
        finetuning_args,
        is_trainable=training_args.do_train,
        add_valuehead=False  # GRPO doesn't need value head
    )

    # 🎯 Load ref_model for KL divergence calculation (when beta != 0 and not using LoRA)
    # 重要：ref_model 必须和训练模型使用相同的模型类（Qwen3VLForConditionalGenerationWithDynamicVideo）
    # 这样计算 KL 散度时 logps 才一致
    # TRL 自动创建 ref_model 会失败（因为自定义类不在 transformers 中），所以我们自己加载
    ref_model = None
    grpo_beta = getattr(finetuning_args, 'grpo_beta', 0.0)
    is_lora = finetuning_args.finetuning_type == "lora"

    if grpo_beta != 0.0 and not is_lora:
        logger.info(f"🔧 Loading ref_model for KL divergence (beta={grpo_beta})")
        from copy import deepcopy

        # ref_model 使用相同的 model_args（包括 enable_dynamic_video=True）
        # 但 finetuning_type 设为 full，且 is_trainable=False（冻结）
        ref_finetuning_args = deepcopy(finetuning_args)
        ref_finetuning_args.finetuning_type = "full"  # Load as full model

        ref_model = load_model(
            tokenizer,
            model_args,  # 使用相同的 model_args，保持模型类一致
            ref_finetuning_args,
            is_trainable=False,  # 冻结，不训练
            add_valuehead=False
        )
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        logger.info(f"✅ ref_model loaded (same model class as training model)")

    # Prepare callbacks
    if callbacks is None:
        callbacks = []

    # Add high-precision loss logging callback
    # This is crucial for GRPO where loss values are often very small
    callbacks.append(HighPrecisionLossCallback())
    logger.info("Added HighPrecisionLossCallback for detailed loss monitoring")

    # Add processor saving callback if using multimodal model
    if processor is not None:
        callbacks.append(SaveProcessorCallback(processor))

    # Create trainer adapter
    trainer = LLamaFactoryGRPOTrainer(
        model=model,
        ref_model=ref_model,  # 传入我们加载的 ref_model
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        tokenizer=tokenizer,
        train_dataset=dataset_module["train_dataset"],
        eval_dataset=dataset_module.get("eval_dataset"),
        callbacks=callbacks,
        processor=processor,
    )

    # Start training
    if training_args.do_train:
        logger.info("Starting GRPO training...")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

        # Save model
        trainer.save_model()

        # Save trainer state
        trainer.save_state()

        # Plot loss if enabled
        trainer.plot_loss()

        logger.info("GRPO training completed successfully")

    # Evaluation
    if training_args.do_eval:
        logger.info("Starting evaluation...")
        metrics = trainer.evaluate()
        logger.info(f"Evaluation metrics: {metrics}")

    logger.info("GRPO workflow completed")