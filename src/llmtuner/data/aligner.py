import os
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Union

from ..extras.constants import AUDIO_PLACEHOLDER, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from ..extras.logging import get_logger
from .utils import Role, get_sample_data


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments

    from ..hparams import DataArguments
    from .parser import DatasetAttr
    from .template import Template


logger = get_logger(__name__)


def convert_multimodal(multimodals: List[Any], multimodal_folder: str, dataset_attr: "DatasetAttr") -> List[Any]:
    r"""
    Optionally concatenates image path to dataset dir when loading from local disk.
    """
    if multimodals is None:
        return []
    if not isinstance(multimodals, list):
        multimodals = [multimodals]
    elif len(multimodals) == 0:
        return []
    else:
        multimodals = multimodals[:]

    def convert_path(obj: Any) -> Any:
        if multimodal_folder is not None and isinstance(obj, str) and not obj.startswith(("http://", "https://")):
            return os.path.join(multimodal_folder, obj)
        else:
            return obj

    if dataset_attr.load_from in ["script", "file"]:
        for i in range(len(multimodals)):
            if isinstance(multimodals[i], list):
                multimodals[i] = [convert_path(o) for o in multimodals[i]]
            else:
                multimodals[i] = convert_path(multimodals[i])

    return multimodals


def convert_placeholder_token(example: Dict[str, Any], multimodel_token_maps: dict[str, str]):
    import sys

    # 🎯 **关键修复**：先解析视频配置（在token替换之前！）
    # print(f"🔧🔧🔧 [DEBUG] aligner: 开始处理，先解析视频配置再替换token 🔧🔧🔧", file=sys.stderr)

    # Debug: 打印条件检查信息
    has_video = bool(example.get("video"))
    has_video_frames = bool(example.get("video_frames"))
    has_video_token = bool(multimodel_token_maps.get("video"))

    # print(f"🔍🔍🔍 [DEBUG] aligner条件检查: video={has_video}, video_frames={has_video_frames}, video_token={has_video_token} 🔍🔍🔍", file=sys.stderr)

    if (example.get("video") or example.get("video_frames")) and multimodel_token_maps.get("video"):
        try:
            from .enhanced_video_token_parser import build_video_config_map

            # print(f"📝📝📝 [DEBUG] aligner: 开始解析视频配置，prompt={len(example.get('prompt', []))}, response={len(example.get('response', []))} 📝📝📝", file=sys.stderr)

            # 构建conversations格式（使用原始内容，未替换token）
            conversations = []
            # 添加prompt对话
            for msg in example["prompt"]:
                conversations.append({
                    'from': 'human' if msg["role"] == 'user' else msg["role"],
                    'value': msg["content"]  # 🎯 这里是原始content，包含<video>或<span>...</span><fps>...</fps><video>
                })
            # 添加response对话
            for msg in example["response"]:
                conversations.append({
                    'from': 'gpt' if msg["role"] == 'assistant' else msg["role"],
                    'value': msg["content"]  # 🎯 这里是原始content
                })

            # print(f"💬💬💬 [DEBUG] aligner: 构建了{len(conversations)}条对话 💬💬💬", file=sys.stderr)

            # # Debug: 打印conversations内容
            # for i, conv in enumerate(conversations):
            #     content_preview = conv['value'][:100] if len(conv['value']) > 100 else conv['value']
            #     print(f"📖📖📖 [DEBUG] 对话{i}: from={conv['from']}, content='{content_preview}...' 📖📖📖", file=sys.stderr)

            # 解析视频配置
            video_configs = build_video_config_map(conversations)

            # print(f"🔍🔍🔍 [DEBUG] aligner: build_video_config_map返回结果: {video_configs} 🔍🔍🔍", file=sys.stderr)
            if video_configs:
                example["video_configs"] = video_configs
                # print(f"🎯🎯🎯 [DEBUG] aligner: 解析到{len(video_configs)}个视频配置 🎯🎯🎯", file=sys.stderr)
            else:
                example["video_configs"] = None
                # print(f"🤔🤔🤔 [DEBUG] aligner: 未找到视频配置 🤔🤔🤔", file=sys.stderr)

        except ImportError as e:
            print(f"⚠️⚠️⚠️ [WARNING] aligner: 无法导入视频token解析器: {e} ⚠️⚠️⚠️", file=sys.stderr)
            example["video_configs"] = None
        except Exception as e:
            print(f"⚠️⚠️⚠️ [WARNING] aligner: 视频配置解析失败: {e} ⚠️⚠️⚠️", file=sys.stderr)
            import traceback
            traceback.print_exc()
            example["video_configs"] = None
    else:
        print(f"❌❌❌ [DEBUG] aligner: 跳过视频配置解析 ❌❌❌", file=sys.stderr)
        example["video_configs"] = None

    # 现在进行token替换（在视频配置解析之后）
    # print(f"🔄🔄🔄 [DEBUG] aligner: 视频配置解析完成，现在进行token替换 🔄🔄🔄", file=sys.stderr)

    def replace_token(text: str):
        if multimodel_token_maps["image"] is not None and "image" in example:
            text = text.replace(IMAGE_PLACEHOLDER, multimodel_token_maps["image"])
        if multimodel_token_maps["video"] is not None and ("video" in example or "video_frames" in example):
            text = text.replace(VIDEO_PLACEHOLDER, multimodel_token_maps["video"])
        if multimodel_token_maps["audio"] is not None and "audio" in example:
            text = text.replace(AUDIO_PLACEHOLDER, multimodel_token_maps["audio"])
        return text

    def replace_content_list(content_list: List[Dict[str, str]]):
        return [{"role": content["role"], "content": replace_token(content["content"])} for content in content_list]

    example["prompt"] = replace_content_list(example["prompt"])
    example["response"] = replace_content_list(example["response"])
    example["system"] = replace_token(example["system"])
    example["tools"] = replace_token(example["tools"])

    # print(f"✅✅✅ [DEBUG] aligner: token替换完成 ✅✅✅", file=sys.stderr)

    return example


@dataclass
class AlpacaDatasetConverter:
    dataset_attr: "DatasetAttr"
    data_args: "DataArguments"
    multimodel_token_maps: dict[str, str]

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        r"""
        Converts alpaca format dataset to the standard format.
        """
        convert_images = partial(
            convert_multimodal, dataset_attr=self.dataset_attr, multimodal_folder=self.data_args.image_folder
        )
        convert_videos = partial(
            convert_multimodal, dataset_attr=self.dataset_attr, multimodal_folder=self.data_args.video_folder
        )
        convert_audios = partial(
            convert_multimodal, dataset_attr=self.dataset_attr, multimodal_folder=self.data_args.audio_folder
        )

        prompt = []
        if self.dataset_attr.history and isinstance(example[self.dataset_attr.history], list):
            for old_prompt, old_response in example[self.dataset_attr.history]:
                prompt.append({"role": Role.USER.value, "content": old_prompt})
                prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

        content = []
        if self.dataset_attr.prompt and example[self.dataset_attr.prompt]:
            content.append(example[self.dataset_attr.prompt])

        if self.dataset_attr.query and example[self.dataset_attr.query]:
            content.append(example[self.dataset_attr.query])

        prompt.append({"role": Role.USER.value, "content": "\n".join(content)})  # "prompt\nquery"

        if self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            response = [{"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], str)
            and isinstance(example[self.dataset_attr.rejected], str)
        ):  # pairwise example
            response = [
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.chosen]},
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.rejected]},
            ]
        elif self.dataset_attr.response and isinstance(
            example[self.dataset_attr.response], (int, str)
        ):  # normal example
            response = [{"role": Role.ASSISTANT.value, "content": str(example[self.dataset_attr.response])}]
        else:  # unsupervised
            response = []

        output = {
            "prompt": prompt,
            "response": response,
            "system": example[self.dataset_attr.system] if self.dataset_attr.system else "",
            "tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "image": convert_images(example[self.dataset_attr.image]) if self.dataset_attr.image else [],
            "video": convert_videos(example[self.dataset_attr.video]) if self.dataset_attr.video else [],
            "video_frames": convert_videos(example[self.dataset_attr.video_frames]) if self.dataset_attr.video_frames else [],
            "audio": convert_audios(example[self.dataset_attr.audio]) if self.dataset_attr.audio else [],
        }
        return convert_placeholder_token(output, self.multimodel_token_maps)


@dataclass
class SharegptDatasetConverter:
    dataset_attr: "DatasetAttr"
    data_args: "DataArguments"
    multimodel_token_maps: dict[str, str]

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        r"""
        Converts sharegpt format dataset to the standard format.
        """
        convert_images = partial(
            convert_multimodal, dataset_attr=self.dataset_attr, multimodal_folder=self.data_args.image_folder
        )
        convert_videos = partial(
            convert_multimodal, dataset_attr=self.dataset_attr, multimodal_folder=self.data_args.video_folder
        )
        convert_audios = partial(
            convert_multimodal, dataset_attr=self.dataset_attr, multimodal_folder=self.data_args.audio_folder
        )
        tag_mapping = {
            self.dataset_attr.user_tag: Role.USER.value,
            self.dataset_attr.assistant_tag: Role.ASSISTANT.value,
            self.dataset_attr.observation_tag: Role.OBSERVATION.value,
            self.dataset_attr.function_tag: Role.FUNCTION.value,
            self.dataset_attr.system_tag: Role.SYSTEM.value,
        }
        odd_tags = (self.dataset_attr.user_tag, self.dataset_attr.observation_tag)
        even_tags = (self.dataset_attr.assistant_tag, self.dataset_attr.function_tag)
        accept_tags = (odd_tags, even_tags)

        messages = example[self.dataset_attr.messages]
        if (
            self.dataset_attr.system_tag
            and len(messages) != 0
            and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag
        ):
            system = messages[0][self.dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = example[self.dataset_attr.system] if self.dataset_attr.system else ""

        aligned_messages = []
        broken_data = False
        for turn_idx, message in enumerate(messages):
            if message[self.dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning("Invalid role tag in {}.".format(messages))
                broken_data = True
                break

            aligned_messages.append(
                {
                    "role": tag_mapping[message[self.dataset_attr.role_tag]],
                    "content": message[self.dataset_attr.content_tag],
                }
            )

        if (not self.dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
            self.dataset_attr.ranking and len(aligned_messages) % 2 == 0
        ):
            logger.warning("Invalid message count in {}.".format(messages))
            broken_data = True

        if broken_data:
            logger.warning("Skipping this abnormal example.")
            prompt, response = [], []
        elif self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], dict)
            and isinstance(example[self.dataset_attr.rejected], dict)
        ):  # pairwise example
            chosen = example[self.dataset_attr.chosen]
            rejected = example[self.dataset_attr.rejected]
            if (
                chosen[self.dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[self.dataset_attr.role_tag] not in accept_tags[-1]
            ):
                logger.warning("Invalid role tag in {}.".format([chosen, rejected]))
                broken_data = True

            prompt = aligned_messages
            response = [
                {
                    "role": tag_mapping[chosen[self.dataset_attr.role_tag]],
                    "content": chosen[self.dataset_attr.content_tag],
                },
                {
                    "role": tag_mapping[rejected[self.dataset_attr.role_tag]],
                    "content": rejected[self.dataset_attr.content_tag],
                },
            ]
        else:  # normal example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        output = {
            "prompt": prompt,
            "response": response,
            "system": system,
            "tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "image": convert_images(example[self.dataset_attr.image]) if self.dataset_attr.image else [],
            "video": convert_videos(example[self.dataset_attr.video]) if self.dataset_attr.video else [],
            "video_frames": convert_videos(example[self.dataset_attr.video_frames]) if self.dataset_attr.video_frames else [],
            "audio": convert_audios(example[self.dataset_attr.audio]) if self.dataset_attr.audio else [],
        }
        return convert_placeholder_token(output, self.multimodel_token_maps)


DATASET_CONVERTERS = {
    "alpaca": AlpacaDatasetConverter,
    "sharegpt": SharegptDatasetConverter,
}


def get_dataset_converter(name: str, dataset_attr: "DatasetAttr", data_args: "DataArguments", multimodel_token_maps: dict[str, str]):
    r"""Get a dataset converter."""
    if name not in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} not found.")

    return DATASET_CONVERTERS[name](dataset_attr, data_args, multimodel_token_maps)


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    template: "Template",
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Aligned dataset:
        prompt: [{"role": "user", "content": "..."}] * (2T - 1)
        response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
        system: "..."
        tools: "..."
    """
    multimodel_token_maps = {
        "image": template.image_token,
        "video": template.video_token,
        "audio": template.audio_token,
    }
    dataset_converter = get_dataset_converter(dataset_attr.formatting, dataset_attr, data_args, multimodel_token_maps)

    sample_data = get_sample_data(dataset=dataset, use_subprocess=True)
    column_names = set(sample_data.keys())
    feature_keys = {"prompt", "response", "system", "tools", "image", "video", "video_frames", "audio", "video_configs"}
    kwargs = {}
    if not data_args.streaming:
        local_process_index = int(os.getenv("LOCAL_PROCESS_RANK", training_args.local_process_index))
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (local_process_index != 0),
            desc="Converting format of dataset",
        )

    dataset = dataset.map(
        dataset_converter,
        batched=False,
        remove_columns=sorted(column_names - feature_keys),
        **kwargs,
    )
    return dataset
