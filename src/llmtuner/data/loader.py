import inspect
import ast
import os
import sys
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Sequence, Union

from datasets import (
    DatasetDict,
    IterableDataset,
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
    load_from_disk,
)

from ..extras.constants import FILEEXT2TYPE
from ..extras.file_system import get_storage_options, has_tokenized_data
from ..extras.logging import get_logger
from .aligner import align_dataset
from .parser import get_dataset_list
from .preprocess import get_dataset_processor
from .template import get_template_and_fix_tokenizer
from .utils import get_sample_data, split_dataset


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments
    from transformers.tokenization_utils import PreTrainedTokenizer

    from ..hparams import DataArguments, ModelArguments
    from .parser import DatasetAttr
    from .template import Template


logger = get_logger(__name__)


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    template: "Template",
):
    logger.info("Loading dataset {}...".format(dataset_attr))

    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path: str = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
                if data_path is None:
                    data_path = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
                elif data_path != FILEEXT2TYPE.get(file_name.split(".")[-1], None):
                    raise ValueError("File types should be identical.")
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
            data_path = FILEEXT2TYPE.get(local_path.split(".")[-1], None)
        elif os.path.isfile(dataset_attr.dataset_name):  # is file
            data_files.append(dataset_attr.dataset_name)
            data_path = FILEEXT2TYPE.get(dataset_attr.dataset_name.split(".")[-1], None)
        else:
            raise ValueError("File not found.")

        if data_path is None:
            raise ValueError("File extension must be txt, csv, json or jsonl.")

    else:
        raise NotImplementedError

    if "trust_remote_code" in inspect.signature(load_dataset).parameters:  # for datasets==2.16.0
        kwargs = {"trust_remote_code": True}
    else:
        kwargs = {}

    if not data_args.streaming:
        kwargs["num_proc"] = data_args.preprocessing_num_workers

    dataset = load_dataset(
        path=data_path,
        name=data_name,
        data_dir=data_dir,
        data_files=data_files,
        split=data_args.split,
        cache_dir=model_args.cache_dir,
        token=model_args.hf_hub_token,
        streaming=data_args.streaming,
        **kwargs,
    )
    if data_args.streaming and not isinstance(dataset, IterableDataset):
        dataset = dataset.to_iterable_dataset(num_shards=training_args.dataloader_num_workers)

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args, training_args, template)


def merge_dataset(
    all_datasets: List[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    if len(all_datasets) == 1:
        return all_datasets[0]
    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning("The samples between different datasets will not be mixed in streaming mode.")
        return concatenate_datasets(all_datasets)
    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning("We recommend using `mix_strategy=concat` in non-streaming mode.")
        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=training_args.seed,
            stopping_strategy="first_exhausted" if data_args.mix_strategy.endswith("under") else "all_exhausted",
        )
    else:
        raise ValueError("Unknown mixing strategy.")


def dict_datasets(all_datasets):
    if len(all_datasets) == 0:
        return None
    if len(all_datasets) == 1:
        return all_datasets[0]
    return {f"dataset_{i}": all_datasets[i] for i in range(len(all_datasets))}


def _get_merged_dataset(
    dataset_attrs: Optional[Sequence["DatasetAttr"]],
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto", "grpo"],
    template: "Template",
    as_dict: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Gets the merged datasets in the standard format.
    """
    if dataset_attrs is None or len(dataset_attrs) == 0:
        return None

    datasets = []
    for dataset_attr in dataset_attrs:
        if dataset_attr.ranking is True:
            raise ValueError("Ranking datasets are not supported in current training stages.")

        datasets.append(_load_single_dataset(dataset_attr, model_args, data_args, training_args, template))
    if as_dict:
        return dict_datasets(datasets)
    return merge_dataset(datasets, data_args, training_args)


def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto", "grpo"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    is_eval: bool = False,
    get_predict: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    if dataset is None:
        return None
    predict_with_generate = False
    if training_args.predict_with_generate:
        if training_args.do_predict:
            # 单独走predict会触发
            predict_with_generate = True
        if training_args.do_train and training_args.do_eval and get_predict:
            # 边train边eval时,predict数据集会触发
            predict_with_generate = True

    dataset_processor = get_dataset_processor(data_args, stage, template, tokenizer, predict_with_generate)
    # 在用户配置--image后，输入数据会加入image列，经过preprocess_func处理后image列需要保留
    reserve_column_name = set()
    if data_args.image:
        reserve_column_name.add('image')
    if data_args.video:
        reserve_column_name.add('video')
        reserve_column_name.add('video_configs')  # 关键修复
    if data_args.audio:
        reserve_column_name.add('audio')

    sample_data = get_sample_data(dataset=dataset, use_subprocess=True)
    column_names = sorted(sample_data.keys() - reserve_column_name)

    # 🎯 Debug: 追踪video_configs字段处理
    import sys
    print(f"🔍🔍🔍 [DEBUG] sample_data keys: {list(sample_data.keys())} 🔍🔍🔍", file=sys.stderr)
    print(f"🔍🔍🔍 [DEBUG] reserve_column_name: {reserve_column_name} 🔍🔍🔍", file=sys.stderr)
    print(f"🔍🔍🔍 [DEBUG] columns_to_remove: {column_names} 🔍🔍🔍", file=sys.stderr)
    has_video_configs = 'video_configs' in sample_data.keys()
    will_remove_video_configs = 'video_configs' in column_names
    print(f"🔍🔍🔍 [DEBUG] has video_configs: {has_video_configs}, will remove: {will_remove_video_configs} 🔍🔍🔍", file=sys.stderr)
    kwargs = {}
    if not data_args.streaming:
        local_process_index = int(os.getenv("LOCAL_PROCESS_RANK", training_args.local_process_index))
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (local_process_index != 0),
            desc="Running tokenizer on dataset",
        )

    dataset = dataset.map(dataset_processor.preprocess_dataset, batched=True, remove_columns=column_names, **kwargs)

    if training_args.should_log:
        try:
            if predict_with_generate:
                print("generate example:")
            elif is_eval:
                print("eval example:")
            else:
                print("training example:")
            sample_data = get_sample_data(dataset=dataset, use_subprocess=True)
            dataset_processor.print_data_example(sample_data)
        except StopIteration:
            raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    if stage == "grpo":
        # GRPO needs the raw prompts and answers, so we keep them
        # But we also need to remove labels column like PPO
        if "labels" in dataset.column_names:
            dataset = dataset.remove_columns(["labels"])

    return dataset


def _get_sequence_parallel_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    model_args: "ModelArguments",
    training_args: "Seq2SeqTrainingArguments",
    tokenizer: "PreTrainedTokenizer",
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    if data_args.shuffle_for_sequence_parallel:
        dataset = dataset.shuffle(seed=training_args.seed)
    return dataset


def _get_dataset_dict(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto", "grpo"],
    template: "Template",
    predict_with_generate: bool = False,
) -> "DatasetDict":
    if template is None:
        template = get_template_and_fix_tokenizer(
            tokenizer, data_args.template, processor_args=model_args, enable_thinking=data_args.enable_thinking
        )
    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")
    if data_args.tokenized_path is not None:
        tokenized_paths = data_args.tokenized_path.split(",")
        if has_tokenized_data(tokenized_paths[0]):
            logger.warning("Loading dataset from disk will ignore other data arguments.")
            dataset_dict = defaultdict(list)
            for tokenized_path in tokenized_paths:
                load_dict = load_from_disk(tokenized_path, storage_options=get_storage_options(tokenized_path))
                for k, v in load_dict.items():
                    dataset_dict[k].append(v)
            dataset_dict = {k: concatenate_datasets(v) if len(v) > 1 else v[0] for k, v in dataset_dict.items()}
            return dataset_dict
        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

        if training_args.world_size > 1:
            logger.warning("it's not recommended to use preprocess datasets with world_size > 1.")
            if not training_args.distributed_state.is_main_process:
                time.sleep(10 * 60 * 60)  # when rank0 exit, other rank will also exit

    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset"):
        dataset_attrs = get_dataset_list(data_args, get_eval=False)
        eval_attrs = get_dataset_list(data_args, get_eval=True)
        dataset = _get_merged_dataset(dataset_attrs, model_args, data_args, training_args, stage, template=template)
        eval_dataset = _get_merged_dataset(
            eval_attrs, model_args, data_args, training_args, stage, template=template, as_dict=True
        )
        if predict_with_generate:
            predict_dataset = _get_merged_dataset(
                eval_attrs, model_args, data_args, training_args, stage, template=template, as_dict=True
            )
        else:
            predict_dataset = None

    with training_args.main_process_first(desc="pre-process dataset"):
        dataset = _get_preprocessed_dataset(
            dataset, data_args, training_args, stage, template, tokenizer, is_eval=False, get_predict=False
        )
        if isinstance(eval_dataset, dict):
            for k, v in eval_dataset.items():
                eval_dataset[k] = _get_preprocessed_dataset(
                    v, data_args, training_args, stage, template, tokenizer, is_eval=True, get_predict=False
                )
        else:
            eval_dataset = _get_preprocessed_dataset(
                eval_dataset, data_args, training_args, stage, template, tokenizer, is_eval=True, get_predict=False
            )
        if isinstance(predict_dataset, dict):
            for k, v in predict_dataset.items():
                predict_dataset[k] = _get_preprocessed_dataset(
                    v, data_args, training_args, stage, template, tokenizer, is_eval=False, get_predict=True
                )
        else:
            predict_dataset = _get_preprocessed_dataset(
                predict_dataset, data_args, training_args, stage, template, tokenizer, is_eval=False, get_predict=True
            )

        if model_args.sequence_parallel_size > 1:
            dataset = _get_sequence_parallel_dataset(
                dataset, data_args, model_args, training_args, tokenizer, is_eval=False
            )
            if eval_dataset is not None:
                eval_dataset = _get_sequence_parallel_dataset(
                    eval_dataset, data_args, model_args, training_args, tokenizer, is_eval=True
                )

        if data_args.val_size > 1e-6:
            assert eval_dataset is None, "The validation dataset is not supported when `val_size` is set."
            dataset_dict = split_dataset(dataset, data_args, seed=training_args.seed)
        else:
            dataset_dict = {}
            if dataset is not None:
                if data_args.streaming:
                    dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)

                dataset_dict["train"] = dataset

            if eval_dataset is not None:
                if isinstance(eval_dataset, dict):
                    for k, v in eval_dataset.items():
                        dataset_dict[f"validation_{k}"] = v
                else:
                    if data_args.streaming:
                        eval_dataset = eval_dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
                    dataset_dict["validation"] = eval_dataset

            if predict_dataset is not None:
                if isinstance(predict_dataset, dict):
                    for k, v in predict_dataset.items():
                        dataset_dict[f"test_{k}"] = v
                else:
                    if data_args.streaming:
                        predict_dataset = predict_dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
                    dataset_dict["test"] = predict_dataset

            dataset_dict = DatasetDict(dataset_dict)
        if data_args.tokenized_path is not None:
            dataset_dict.save_to_disk(
                data_args.tokenized_path,
                storage_options=get_storage_options(data_args.tokenized_path),
                num_proc=data_args.preprocessing_num_workers,
            )
            logger.info("Saved tokenized dataset to {}".format(data_args.tokenized_path))
            if training_args.world_size > 1:
                logger.warning("because world_size > 1, the program exit with error code -1. ignore it.")
                sys.exit(-1)
            sys.exit(0)
        return dataset_dict


def get_dataset(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto", "grpo"],
    template: "Template",
    predict_with_generate: bool = False,
) -> Dict[str, Union["Dataset", "IterableDataset", "DatasetDict"]]:
    dataset_dict = _get_dataset_dict(
        tokenizer, model_args, data_args, training_args, stage, template, predict_with_generate
    )
    dataset_module = {}
    dataset_module["eval_dataset"] = {
        k.replace("validation_", ""): v for k, v in dataset_dict.items() if k.startswith("validation_")
    }
    dataset_module["predict_dataset"] = {
        k.replace("test_", ""): v for k, v in dataset_dict.items() if k.startswith("test_")
    }
    if len(dataset_module["eval_dataset"]) == 0:
        dataset_module.pop("eval_dataset")
    if len(dataset_module["predict_dataset"]) == 0:
        dataset_module.pop("predict_dataset")

    if "train" in dataset_dict:
        dataset_module["train_dataset"] = dataset_dict["train"]
    if "validation" in dataset_dict:
        dataset_module["eval_dataset"] = dataset_dict["validation"]
    if "test" in dataset_dict:
        dataset_module["predict_dataset"] = dataset_dict["test"]
    return dataset_module
