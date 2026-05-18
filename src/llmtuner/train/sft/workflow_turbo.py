from typing import TYPE_CHECKING, List, Optional

from ...data import get_dataset
from ...data.collator import MultiModalDataCollatorForSeq2Seq
from ...extras.callbacks import SaveProcessorCallback
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import get_logits_processor
from ...extras.packages import is_turbo_available
from ...extras.ploting import plot_loss
from ...model import load_template, load_turbo_model


if is_turbo_available():
    from transformers_turbo import TurboTrainer

logger = get_logger(__name__)


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def run_sft_turbo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    assert not training_args.predict_with_generate, "Turbo does not support predict_with_generate."
    data_args.neat_packing = training_args.sequence_packing = data_args.neat_packing or training_args.sequence_packing
    data_args.packing = data_args.neat_packing or data_args.packing
    cp_size = training_args.context_parallel_size or 1
    if data_args.neat_packing and cp_size > 1:
        data_args.sequence_length_pad_multiple = data_args.sequence_length_pad_multiple or 2 * cp_size
        pad_to = data_args.sequence_length_pad_multiple
        assert (pad_to % (2 * cp_size) == 0), f"When context_parallel_size > 1 ans packing=true, sequence_length_pad_multiple should be the multiple of 2 * cp_size ({2 * cp_size})."

    template = load_template(model_args, template=data_args.template, enable_thinking=data_args.enable_thinking)
    tokenizer, processor = template.tokenizer, template.processor
    data_args.cutoff_len += 1  # turbo will shift the input_ids and labels
    dataset_module = get_dataset(
        tokenizer, model_args, data_args, training_args, stage="sft", template=template
    )
    data_args.cutoff_len -= 1
    model = load_turbo_model(
        tokenizer, training_args, model_args, finetuning_args, is_trainable=training_args.do_train
    )

    pad_of = 128 * (training_args.tensor_model_parallel_size or 1) * cp_size
    if (training_args.expert_model_parallel_size or 1) > 1 and not training_args.variable_seq_lengths:
        max_len = data_args.cutoff_len  # all pad to max when expert parallel
        pad_of = max_len + (-max_len) % pad_of

    data_collator = MultiModalDataCollatorForSeq2Seq(
        template=template,
        tokenizer=tokenizer,
        is_turbo=True,
        pad_to_multiple_of=pad_of,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    callbacks = [SaveProcessorCallback(processor)] + callbacks

    if data_args.dynamic_probs:
        raise ValueError("Turbo does not support dynamic probs.")
    if training_args.do_predict:
        raise ValueError("Turbo does not support predict.")

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams

    # Initialize our Trainer
    trainer = TurboTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])
