import asyncio
import itertools
import os
from enum import Enum, unique
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..data.template import get_template_and_fix_tokenizer
from ..extras.logging import get_logger
from ..extras.misc import down_model, to_device
from ..extras.packages import is_vllm_version_greater_than
from ..model.loader import load_model_and_template, load_model_config, load_tokenizer
from .constant import OutputType
from .data import OutputDataAfterEngine
from .ray_tool import ray_cluster_init_local, ray_cluster_init_remote


if TYPE_CHECKING:
    from .infer_args import InferArguments

logger = get_logger(__name__)


def change_dict_to_list(data_dict):
    if not data_dict:
        return []
    column_length_list = [len(values) for key, values in data_dict.items()]
    assert len(set(column_length_list)) == 1, \
        f"data length not same! get length {column_length_list}"

    data_list_of_dicts = [dict(zip(data_dict.keys(), row)) for row in zip(*data_dict.values())]
    return data_list_of_dicts


def change_list_to_dict(data_list):
    if not data_list:
        return {}
    column_length_list = [len(column) for column in data_list]
    assert len(set(column_length_list)) == 1,\
        f"data length not same! get length {column_length_list}"

    keys = data_list[0].keys()
    data_list_of_dicts = {k: [d.get(k) for d in data_list] for k in keys}
    return data_list_of_dicts


def model_args_refine(model_args):
    model_args.device_map = "balanced"
    model_args.model_name_or_path = down_model(model_args.model_name_or_path, used_for="model_name_or_path")
    if model_args.adapter_name_or_path is not None:
        model_args.adapter_name_or_path = [down_model(item,used_for="adapter_name_or_path") for item in model_args.adapter_name_or_path]
    return model_args


def change_column_to_row(data_list):
    if not data_list:
        return data_list
    column_length_list = [len(column) for column in data_list]
    assert len(set(column_length_list)) == 1, f"data length not same! get length {column_length_list}"
    return list(map(list, zip(*data_list)))


def run_coroutine(coroutine):
    loop = asyncio.get_event_loop()
    task = loop.create_task(coroutine)
    return loop.run_until_complete(task)


@unique
class CallMode(str, Enum):
    OFFLINE_INFERENCE = "offline_inference"
    CHAT_API = "chat_api"


class BaseGenerateEngine(object):
    def __init__(self, model, tokenizer, processor, config, template, generating_args, infer_args, call_mode = CallMode.OFFLINE_INFERENCE):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.template = template
        self.gen_kwargs = generating_args.to_dict()
        self.infer_args = infer_args
        self.print_debug_log = True

        self.num_return = None

        self.fail_input_ids_index = []
        self.gen_kwargs_refine()

        self.output_type = self.confirm_output_type(config, self.gen_kwargs)

        self.tokenizer_refine()
        self.call_mode = call_mode
        if not self.call_mode:
            self.call_mode = CallMode.OFFLINE_INFERENCE

    def confirm_output_type(self, config, generating_args):
        return OutputType.Sequence

    def get_call_mode(self):
        return self.call_mode

    def get_num_return(self):
        return self.num_return

    def get_processor(self):
        return self.processor

    def get_template(self):
        return self.template

    def get_tokenizer(self):
        return self.tokenizer

    def get_model_dtype(self):
        raise Exception('BaseGenerateEngine do not impl get_model_dtype')

    def get_output_type(self):
        return self.output_type

    def tokenizer_refine(self):
        if self.output_type in (OutputType.Sequence, OutputType.TokensProbabilities):
            self.tokenizer.padding_side = "left"  # restore padding side
            self.tokenizer.init_kwargs["padding_side"] = "left"
        else:
            self.tokenizer.padding_side = "right"
            self.tokenizer.init_kwargs["padding_side"] = "right"

    def gen_kwargs_refine(self):
        self.num_return = self.gen_kwargs["num_return_sequences"] if "num_return_sequences" in self.gen_kwargs else 1

        # config token id after get_template_and_fix_tokenizer
        stop_token_ids = self.tokenizer.convert_tokens_to_ids(self.template.stop_words) + [self.tokenizer.eos_token_id]
        self.gen_kwargs["eos_token_id"] = list({token_id for token_id in stop_token_ids if token_id is not None})
        self.gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        logger.info(f"after refine gen_kwargs: {self.gen_kwargs}")

    def predict_batch_impl(self, samples):
        logger.warning('BaseGenerateEngine not be inherit!')
        return samples

    def on_predict_batch_begin(self, samples):
        self.fail_input_ids_index = samples.pop('fail_input_ids_index', [])
        return samples

    def on_predict_batch_finish(self, outputs_after_engine):
        outputs = outputs_after_engine.predict
        if self.num_return > 1:
            outputs = [outputs[i:i + self.num_return] for i in range(0, len(outputs), self.num_return)]
        # 如果有失败数据索引，则将失败数据output设为None
        if self.fail_input_ids_index:
            for _index in self.fail_input_ids_index:
                outputs[_index] = None
        outputs_after_engine.predict = outputs
        return outputs_after_engine

    def predict_batch(self, samples):
        samples = self.on_predict_batch_begin(samples)
        outputs_after_engine = self.predict_batch_impl(samples)
        outputs_after_engine = self.on_predict_batch_finish(outputs_after_engine)
        return outputs_after_engine


class HuggingFaceGenerateEngine(BaseGenerateEngine):
    def __init__(self, model_args, infer_args: "InferArguments", finetuning_args, generating_args, call_model):
        model_args = model_args_refine(model_args)
        model_args.compute_dtype = infer_args.compute_dtype
        # add_valuehead设置为None,表示框架自行决定是否加载valuehead
        model, template = load_model_and_template(
            model_args,
            finetuning_args,
            template=infer_args.template,
            is_trainable=False,
            add_valuehead=None,
            enable_thinking=infer_args.enable_thinking,
        )
        tokenizer, processor = template.tokenizer, template.processor
        config = model.config
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = True
        if hasattr(model, "generation_config") and hasattr(model.generation_config, "use_cache"):
            model.generation_config.use_cache = True

        # 🎯 保存model_args用于动态视频功能
        self.model_args = model_args

        super().__init__(model, tokenizer, processor, config, template, generating_args, infer_args, call_model)

        if self.output_type == OutputType.Embedding:
            logger.info('current model is emb_model, init embedding generate environment ...')
            self.emb_pooling_method = config.emb_pooling_method
            self.predict_batch_impl = self.predict_batch_impl_for_emb
        elif self.output_type == OutputType.RewardScores:
            logger.info('current model is rm model, init reward scores generate environment ...')
            self.predict_batch_impl = self.predict_batch_impl_for_reward_scores
        elif self.output_type == OutputType.Audio:
            logger.info('current model is tts model, init audio generate environment ...')
            model.init_tts()
            self.gen_kwargs["audio_output_path"] = infer_args.outputs
        if "presence_penalty" in self.gen_kwargs:
            # vllm mode only
            self.gen_kwargs.pop("presence_penalty")

    def confirm_output_type(self, config, gen_kwargs):
        output_scores = False
        output_embedding = False
        output_reward_scores = False
        output_audios = False

        if self.infer_args.output_scores:
            output_scores = True
        if hasattr(config, "is_embedding_model"):
            output_embedding = True
        if hasattr(config, "is_rm_model"):
            output_reward_scores = True
        if self.infer_args.output_audio:
            output_audios = True

        output_type_flag = sum([output_scores, output_embedding, output_reward_scores, output_audios])
        assert output_type_flag <= 1, f"only support only one special output type, current " \
                                     f"output_scores:{output_scores}, " \
                                     f"output_embedding:{output_embedding}, " \
                                     f"output_reward_scores:{output_reward_scores} " \
                                     f"output_audios:{output_audios}"
        if output_scores:
            return OutputType.TokensProbabilities
        elif output_embedding:
            return OutputType.Embedding
        elif output_reward_scores:
            return OutputType.RewardScores
        elif output_audios:
            return OutputType.Audio
        else:
            return OutputType.Sequence

    def get_model_dtype(self):
        if hasattr(self.config, "is_trl_causal_lm_with_value_head"):
            # RM模型添加value_head之后结构有变化
            return self.model.pretrained_model.dtype
        else:
            return self.model.dtype

    def print_debug_info(self, prompts, input_ids, output_ids, outputs):
        if not self.print_debug_log:
            return
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        if isinstance(output_ids, torch.Tensor):
            output_ids = output_ids.tolist()
        # 🎯 输出所有样本的详细信息，而不仅仅是第一个
        print("\\n🔍🔍🔍 [DEBUG] 推理结果详细信息 - 共{}个样本 🔍🔍🔍".format(len(outputs)))

        for i in range(len(outputs)):
            print("\\n📝📝📝 [DEBUG] ========== 样本 {}/{} ========== 📝📝📝".format(i+1, len(outputs)))

            # 输入信息
            if i < len(prompts):
                print("🎯 input:\\n{}\\n".format(prompts[i]))

            # 输入token信息
            if i < len(input_ids):
                print("🎯 input_ids ({} tokens):\\n{}\\n".format(len(input_ids[i]), input_ids[i]))
                prompt_decoded = self.tokenizer.decode(input_ids[i], skip_special_tokens=False)
                print("🎯 prompt (decoded with special tokens):\\n{}\\n".format(repr(prompt_decoded)))

            # 输出token信息 - 关键部分
            if i < len(output_ids):
                print("🎯 output_ids ({} tokens):\\n{}\\n".format(len(output_ids[i]), output_ids[i]))

                # 不跳过特殊token的解码 - 这是用户特别要求的
                output_noskip = self.tokenizer.decode(output_ids[i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                print("🎯 output_noskip (with special tokens):\\n{}\\n".format(repr(output_noskip)))

            # 最终输出
            if i < len(outputs):
                print("🎯 final_output (clean):\\n{}\\n".format(outputs[i]))

            # 如果是动态视频插入的情况，额外输出视频相关token的解析
            # if i < len(output_ids):
            #     self._debug_video_tokens_in_output(output_ids[i])

            print("📝📝📝 [DEBUG] ========== 样本 {} 结束 ========== 📝📝📝".format(i+1))

        print("🔍🔍🔍 [DEBUG] 所有样本输出完毕 🔍🔍🔍\\n")

        # 🎯 强制保持debug输出开启，便于调试动态视频插入功能
        self.print_debug_log = True

        # if self.call_mode == CallMode.CHAT_API:
        #     self.print_debug_log = True
        # else:
        #     self.print_debug_log = False

    def _debug_video_tokens_in_output(self, output_token_ids):
        """调试输出中的视频相关token"""
        try:
            # 检查是否包含动态视频插入相关的特殊token
            special_tokens = {
                151669: '<span>',
                151670: '</span>',
                151671: '<fps>',
                151672: '</fps>',
                151652: '<|vision_start|>',
                151653: '<|vision_end|>',
                151656: '<|video_pad|>'
            }

            found_special = []
            for i, token_id in enumerate(output_token_ids):
                if token_id in special_tokens:
                    found_special.append((i, token_id, special_tokens[token_id]))

            if found_special:
                print("🎬🎬🎬 [DEBUG] 发现视频相关特殊token: 🎬🎬🎬")
                # for pos, token_id, token_name in found_special:
                #     print("   位置 {}: {} -> {}".format(pos, token_id, token_name))

                # 检查是否有时间戳token (格式: <数字.数字 seconds>)
                decoded_output = self.tokenizer.decode(output_token_ids, skip_special_tokens=False)
                import re
                timestamps = re.findall(r'<(\d+\.\d+) seconds>', decoded_output)
                if timestamps:
                    print("⏰⏰⏰ [DEBUG] 发现时间戳: {} ⏰⏰⏰".format(timestamps))

        except Exception as e:
            print("⚠️ [WARNING] 调试视频token失败: {}".format(e))

    def predict_batch_impl_for_reward_scores(self, samples):
        samples.pop("prompt_data", None)
        if hasattr(self.config, "is_trl_causal_lm_with_value_head"):
            # RM模型添加value_head之后结构有变化
            _device = self.model.pretrained_model.device
        else:
            _device = self.model.device

        inputs = to_device(samples, _device)
        inputs.pop("samples", None)  # original text samples
        outputs = self.model(**inputs)
        if hasattr(self.config, "is_trl_causal_lm_with_value_head"):
            scores_list = outputs[2][:, -1].tolist()
        else:
            scores_list = outputs[0].squeeze(1).tolist()
        outputs = OutputDataAfterEngine(predict=scores_list)
        return outputs

    def predict_batch_impl_for_emb(self, samples):

        samples.pop("prompt_data", None)
        inputs = to_device(samples, self.model.device)
        inputs.pop("samples", None)  # original text samples
        attention_mask = inputs['attention_mask']
        outputs = self.model(**inputs)

        if self.emb_pooling_method == 'CLS':
            output_embeddings = outputs[0][:, 0]
        elif self.emb_pooling_method == 'LAST':
            sequence_lengths = attention_mask.sum(dim=1) - 1
            last_hidden_states = outputs[0]
            batch_size = last_hidden_states.shape[0]
            output_embeddings = last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        else:
            raise Exception('Unknown pooling method')
        # normalize embeddings
        output_embeddings = torch.nn.functional.normalize(output_embeddings, p=2, dim=1)
        outputs = OutputDataAfterEngine(predict=output_embeddings.tolist())
        return outputs

    def predict_batch_impl(self, samples):
        prompt_data = samples.pop("prompt_data")
        input_ids = samples['input_ids']
        pad_id = self.tokenizer.pad_token_id
        max_len = max(map(len, input_ids))

        attention_mask = [[0] * (max_len - len(input_id)) + [1] * len(input_id) for input_id in input_ids]
        input_ids = [[pad_id] * (max_len - len(input_id)) + input_id for input_id in input_ids]
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        samples['attention_mask'] = attention_mask
        samples['input_ids'] = input_ids

        generated_tokens_with_probs = []
        inputs = to_device(samples, self.model.device)

        # 🎯 为增强版Qwen3VL设置视频上下文
        if hasattr(self.model, 'set_video_context') and 'original_video_paths' in samples:
            self.model.set_video_context(
                samples['original_video_paths'],
                self.processor,
                self.model_args
            )

        if self.output_type == OutputType.TokensProbabilities:
            self.gen_kwargs['output_scores'] = True

        inputs.pop("samples", None)  # original text samples
        inputs.pop("original_video_paths", None)  # 移除视频路径，避免传递给model.generate
        outputs = self.model.generate(
            return_dict_in_generate=True,
            **inputs,
            **self.gen_kwargs,
        )
        if self.output_type == OutputType.TokensProbabilities:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True)
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs.sequences[:, input_length:]
            for generated_token, transition_score in zip(generated_tokens, transition_scores):
                generated_tokens_with_prob = []
                for token, score in zip(generated_token, transition_score):
                    token = int(token.cpu())
                    if token == self.tokenizer.pad_token_id:
                        continue
                    generated_tokens_with_prob.append({
                        "id": str(token),
                        "text": self.tokenizer.decode(token),
                        "prob": str(np.round(np.exp(score.cpu().numpy()), 6)),
                    })
                generated_tokens_with_probs.append(generated_tokens_with_prob)

        multimodal_output_path = getattr(outputs, "multimodal_output_path", None)
        has_input_id_as_prefix = lambda input_ids, output_ids: (len(output_ids) > len(input_ids)) \
            and (all(input_id == output_id for input_id, output_id in zip(input_ids, output_ids)))
        output_end_index = lambda output, start_index=0: output[start_index:].index(self.tokenizer.eos_token_id) + start_index \
            if self.tokenizer.eos_token_id in output[start_index:] else None
        truncate_output = lambda output, start_index=0: output[start_index:output_end_index(output, start_index)]
        output_ids = outputs["sequences"].tolist()
        output_tensor_ids = outputs["sequences"]
        output_finish_reasons = []
        output_lengths = []
        outputs = [
            self.tokenizer.decode(
                truncate_output(output, len(input_ids[j // self.num_return]))
                if has_input_id_as_prefix(input_ids[j // self.num_return], output) else truncate_output(output),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for j, output in enumerate(output_ids)
        ]
        if self.call_mode == CallMode.CHAT_API:
            for i in range(len(outputs)): 
                eos_index = (output_tensor_ids[i] == self.tokenizer.eos_token_id).nonzero()
                response_length = (eos_index[-1].item() + 1) if len(eos_index) else len(output_tensor_ids[i])
                output_lengths.append(response_length)
                output_finish_reasons.append("stop" if len(eos_index) else "length")

        self.print_debug_info(prompt_data, input_ids, output_ids, outputs)

        outputs = OutputDataAfterEngine(
            predict=outputs,
            generated_tokens_with_probs=generated_tokens_with_probs,
            predict_finish_reasons=output_finish_reasons,
            predict_response_lengths=output_lengths,
            audio_path=multimodal_output_path
        )
        return outputs


class VllmGenerateEngine(BaseGenerateEngine):
    def __init__(self, model_args, infer_args: "InferArguments", generating_args, call_mode):
        model_args = model_args_refine(model_args)

        tokenizer, processor = load_tokenizer(model_args)
        template = get_template_and_fix_tokenizer(
            tokenizer,
            name=infer_args.template,
            processor=processor,
            processor_args=model_args,
            enable_thinking=infer_args.enable_thinking,
            vllm_mode=True,
        )
        config = load_model_config(tokenizer=tokenizer, model_args=model_args, is_trainable=False)
        llm_args = {
            "model": model_args.model_name_or_path,
            "gpu_memory_utilization": infer_args.gpu_memory_utilization,
            "tensor_parallel_size": infer_args.tensor_parallel_size,
            "pipeline_parallel_size": infer_args.pipeline_parallel_size,
            "enable_expert_parallel": infer_args.enable_expert_parallel,
            "max_num_seqs": infer_args.vllm_max_num_seqs,
            "trust_remote_code": True,
            "enable_chunked_prefill": False,
            "max_model_len": infer_args.max_model_len,
            "enable_lora": True if model_args.adapter_name_or_path else False,
            "max_lora_rank": infer_args.max_lora_rank,
        }
        if infer_args.compute_dtype:
            dtype_str = str(infer_args.compute_dtype).lstrip("torch.")
            llm_args["dtype"] = dtype_str
        if is_vllm_version_greater_than("0.6.1"):
            llm_args["disable_custom_all_reduce"] = infer_args.disable_custom_all_reduce
            llm_args["max_seq_len_to_capture"] = infer_args.max_seq_len_to_capture
            llm_args["num_scheduler_steps"] = infer_args.num_scheduler_steps
            llm_args["enable_prefix_caching"] = infer_args.enable_prefix_caching
        if not is_vllm_version_greater_than("0.7.0"):
            llm_args["worker_use_ray"] = True
        if is_vllm_version_greater_than("0.7.2"):
            llm_args["distributed_executor_backend"] = "ray"
            if not is_vllm_version_greater_than("0.10.0"):
                llm_args["device"] = "cuda"

        if is_vllm_version_greater_than("0.11.0"):
            llm_args.pop('max_seq_len_to_capture', None)
            llm_args.pop('num_scheduler_steps', None)

        if getattr(config, "visual", None) or getattr(config, "vision_config", None):
            if is_vllm_version_greater_than("0.11.0"):
                mm_processor_kwargs={
                    "min_pixels": model_args.min_resolution,
                    "max_pixels": model_args.image_resolution,
                    "fps": model_args.video_fps,
                }
                llm_args["mm_processor_kwargs"] = mm_processor_kwargs
                
            if is_vllm_version_greater_than("0.7.2"):
                llm_args["limit_mm_per_prompt"] = {"image": infer_args.limit_mm_per_prompt}
            else:
                if getattr(config, "model_type", None) == "qwen2_vl":
                    rope_scaling = {
                        "type": "mrope",
                        "mrope_section": [16, 24, 24],
                    }
                    llm_args["rope_scaling"] = rope_scaling
                llm_args["enforce_eager"] = True

        model = self.create_model(llm_args)

        self.model_args = model_args
        self.vllm_sampling_params = None
        self.vllm_lora_request = None
        super().__init__(model, tokenizer, processor, config, template, generating_args, infer_args, call_mode)

    def confirm_output_type(self, config, gen_kwargs):
        if self.infer_args.output_scores:
            return OutputType.TokensProbabilities
        if hasattr(config, "is_embedding_model"):
            raise Exception(f"{self.__class__.__name__} do not support output_type:{OutputType.Embedding}")
        if hasattr(config, "is_rm_model"):
            raise Exception(f"{self.__class__.__name__} do not support output_type:{OutputType.RewardScores}")
        return OutputType.Sequence

    def create_model(self, args):
        from vllm import LLM
        return LLM(**args)

    def get_model_dtype(self):
        return self.model.llm_engine.model_config.dtype

    def create_lora_request_for_vllm(self, adapter_name_or_path):
        from vllm.lora.request import LoRARequest
        assert len(adapter_name_or_path) == 1, "only support single lora adapter"
        return LoRARequest("adapter", 1, adapter_name_or_path[0])

    def create_sampling_params_for_vllm(self):
        from vllm import SamplingParams

        if not self.gen_kwargs["do_sample"]:
            self.gen_kwargs["temperature"] = 0.0

        basis_sampling_params = {
            'max_tokens': self.gen_kwargs["max_new_tokens"],
            'stop_token_ids': self.gen_kwargs["eos_token_id"],
            'repetition_penalty': self.gen_kwargs["repetition_penalty"],
            'presence_penalty': self.infer_args.presence_penalty,
            'n': self.gen_kwargs["num_return_sequences"],
        }

        if self.gen_kwargs["num_beams"] > 1:
            beams_sampling_params = {
                'best_of': self.gen_kwargs["num_beams"],
                'use_beam_search': True,
                'temperature': 0.0
            }
            basis_sampling_params.update(beams_sampling_params)
        else:
            do_sample_sampling_params = {
                'temperature': self.gen_kwargs["temperature"],
                'top_p': self.gen_kwargs["top_p"],
                'top_k': self.gen_kwargs["top_k"]
            }
            basis_sampling_params.update(do_sample_sampling_params)

        if self.output_type == OutputType.TokensProbabilities:
            basis_sampling_params['logprobs'] = 1

        if 'stop' in self.gen_kwargs:
            basis_sampling_params['stop'] = self.gen_kwargs['stop']
            # 需要移除，否则会覆盖。
            basis_sampling_params.pop('stop_token_ids')

        return SamplingParams(**basis_sampling_params)

    def parse_logprob_from_vllm_output(self, vllm_outputs):
        prob_list = []
        for j, row_outputs in enumerate(vllm_outputs):
            for jj, single_outputs in enumerate(row_outputs.outputs):
                row_prob_list = []
                for logprob_items in single_outputs.logprobs:
                    cur_logprob_item_key = list(logprob_items.keys())[0]
                    cur_logprob_item = logprob_items[cur_logprob_item_key]
                    row_prob_list.append({
                        "id": str(cur_logprob_item_key),
                        "text": cur_logprob_item.decoded_token,
                        "prob": cur_logprob_item.logprob,
                    })
                prob_list.append(row_prob_list)
        return prob_list

    def vllm_params_init(self):
        # 这个对于chat 场景，原实现如果已经初始化就不会再初始化。这个在chat中需要每次请求都修改配置的，是无法接受的。
        if self.call_mode == CallMode.CHAT_API:
            self.vllm_sampling_params = self.create_sampling_params_for_vllm()
            logger.info(f"sampling_params:{self.vllm_sampling_params}")
        else:
            # 对于离线批量推理，这里保持原逻辑，已经初始化之后不再初始化。
            if not self.vllm_sampling_params:
                self.vllm_sampling_params = self.create_sampling_params_for_vllm()
        if self.model_args.adapter_name_or_path and not self.vllm_lora_request:
            self.vllm_lora_request = self.create_lora_request_for_vllm(self.model_args.adapter_name_or_path)

    def predict_batch_impl(self, samples):
        self.vllm_params_init()
        input_ids = samples['input_ids']
        llm_inputs = [{"prompt_token_ids": prompt_token_ids} for prompt_token_ids in input_ids]
        if "multi_modal_data" in samples:
            assert len(llm_inputs) == len(samples['multi_modal_data']), "the length of input_ids not same as multi_modal_data"
            for i in range(len(llm_inputs)):
                if samples['multi_modal_data'][i]:
                    if getattr(self.config, "model_type", None) in ("qwen3_vl", "qwen3_vl_moe") and self.infer_args.video:
                        # qwen3-vl process_vision_info后返回元组数据，经过dataloader处理转为list，list数据传入vllm会走入其他分支导致逻辑错误，所以此处恢复元组结构
                        llm_inputs[i]['multi_modal_data']= {'video': [tuple(item) for item in samples['multi_modal_data'][i]['video']]}
                        llm_inputs[i]['mm_processor_kwargs'] = samples['multi_modal_data'][i]['mm_processor_kwargs']
                    else:
                        llm_inputs[i]['multi_modal_data'] = samples['multi_modal_data'][i]

        vllm_outputs = self.model.generate(
            llm_inputs, sampling_params=self.vllm_sampling_params, lora_request=self.vllm_lora_request
        )
        outputs = self.vllm_output_post_processing(vllm_outputs)
        return outputs

    def vllm_output_post_processing(self, vllm_outputs, request_ids_for_outputs=None):
        generated_tokens_with_probs = self.parse_logprob_from_vllm_output(vllm_outputs=vllm_outputs) \
            if self.output_type == OutputType.TokensProbabilities else None
        # Print the outputs.
        outputs = [
            self.tokenizer.decode(
                output.token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for j, outputs in enumerate(vllm_outputs) for output in outputs.outputs
        ]
        output_finish_reasons = []
        output_lengths = []
        if self.call_mode == CallMode.CHAT_API:
            logger.info(vllm_outputs)
            vllm_texts = []
            for j, outputs in enumerate(vllm_outputs):
                for output in outputs.outputs:
                    output_lengths.append(len(output.token_ids))
                    output_finish_reasons.append(output.finish_reason)
                    vllm_texts.append(output.text)
            outputs = vllm_texts
        outputs = OutputDataAfterEngine(predict=outputs,
                                        generated_tokens_with_probs=generated_tokens_with_probs,
                                        predict_finish_reasons=output_finish_reasons,
                                        predict_response_lengths=output_lengths,
                                        request_ids=request_ids_for_outputs)
        return outputs


class VllmAsyncGenerateEngine(VllmGenerateEngine):
    def create_model(self, args):
        from vllm import AsyncLLMEngine,AsyncEngineArgs
        engine_args = AsyncEngineArgs(**args)
        self.id_creator = itertools.count(1)
        return AsyncLLMEngine.from_engine_args(engine_args)

    async def do_get_model_dtype(self, results_generator):
        config = await results_generator
        return config.dtype

    def get_model_dtype(self):
        return self.model.model_config

    async def do_generate_single(self, results_generator):
        output_list = {}
        async for request_output in results_generator:
            final_output = request_output
            for item in request_output.outputs:
                output_list[item.index] = item
        final_output.outputs = list(output_list.values())
        return final_output

    async def predict_batch_impl(self, samples):

        self.vllm_params_init()
        input_ids = samples['input_ids']

        llm_inputs = {"prompt_token_ids": input_ids[0]}
        mm_data = samples.get('multi_modal_data', None)
        if mm_data and samples['multi_modal_data'][0]:
            if getattr(self.config, "model_type", None) in ("qwen3_vl", "qwen3_vl_moe") and self.infer_args.video:
                # qwen3-vl process_vision_info后返回元组数据，经过dataloader处理转为list，list数据传入vllm会走入其他分支导致逻辑错误，所以此处恢复元组结构
                llm_inputs['multi_modal_data']= {'video': [tuple(item) for item in samples['multi_modal_data'][0]['video']]}
                llm_inputs['mm_processor_kwargs'] = samples['multi_modal_data'][0]['mm_processor_kwargs']
            else:
                llm_inputs['multi_modal_data'] = samples['multi_modal_data'][0]

        vllm_outputs_generator = self.model.generate(
            llm_inputs, 
            sampling_params=self.vllm_sampling_params, 
            lora_request=self.vllm_lora_request,
            request_id=str(next(self.id_creator)),
        )

        vllm_outputs = await self.do_generate_single(vllm_outputs_generator)

        outputs = self.vllm_output_post_processing([vllm_outputs])
        return outputs

    async def predict_batch(self, samples):
        samples = self.on_predict_batch_begin(samples)
        outputs_after_engine = await self.predict_batch_impl(samples)
        outputs_after_engine = self.on_predict_batch_finish(outputs_after_engine)
        return samples, outputs_after_engine


class VllmEnvSetup(object):
    def __init__(self, model_args, infer_args):
        self.model_args = model_args
        self.infer_args = infer_args

    def __call__(self, *args, **kwds):
        model_args = model_args_refine(self.model_args)
        tokenizer, _ = load_tokenizer(model_args)
        load_model_config(tokenizer=tokenizer, model_args=model_args, is_trainable=False)


def create_generate_engine(model_args, infer_args, finetuning_args, generating_args, call_mode=CallMode.OFFLINE_INFERENCE):
    if 'vllm' in infer_args.infer_mode:
        if infer_args.infer_mode == 'vllm-multi-node':
            gpu_cnt_in_group = infer_args.tensor_parallel_size * infer_args.pipeline_parallel_size
            gpu_cnt_in_device = torch.cuda.device_count()
            assert gpu_cnt_in_group > gpu_cnt_in_device, "tensor_parallel_size * pipeline_parallel_size has to be greater than torch.cuda.device_count() in vllm-multi-node mode"
            assert gpu_cnt_in_group % gpu_cnt_in_device == 0, "tensor_parallel_size * pipeline_parallel_size must be a multiple of  torch.cuda.device_count()"
            node_size = gpu_cnt_in_group // gpu_cnt_in_device
            vllm_env_setup = VllmEnvSetup(model_args=model_args, infer_args=infer_args)
            ray_cluster_init_remote(model_args.model_name_or_path, node_size, vllm_env_setup)
            _rank = int(os.getenv("RANK", "0"))
            _world_size = int(os.getenv("WORLD_SIZE", "1"))
            assert _rank % node_size == 0
            assert _world_size % node_size == 0 
            os.environ['RANK'] = str(_rank // node_size)
            os.environ['WORLD_SIZE'] = str(_world_size // node_size)
        else:
            ray_cluster_init_local(infer_args)

    if infer_args.infer_mode == 'vllm':
        return VllmGenerateEngine(model_args, infer_args, generating_args, call_mode)
    if infer_args.infer_mode in  ('vllm-async', 'vllm-multi-node'):
        assert call_mode == CallMode.OFFLINE_INFERENCE, "vllm-async or vllm-multi-node only support OFFLINE_INFERENCE"
        return VllmAsyncGenerateEngine(model_args, infer_args, generating_args, call_mode)
    else:
        return HuggingFaceGenerateEngine(model_args, infer_args, finetuning_args, generating_args, call_mode)
