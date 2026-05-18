"""
增强版Qwen3VL模型，支持动态视频token插入功能

🚀 简化版实现：通过模拟首次推理来处理动态视频插入

核心策略：
1. 检测</fps>token时插入视频token
2. 临时隐藏past_key_values，重置cache_position
3. 模拟首次推理让Qwen3VL标准处理视频特征
4. 获得完整KV状态后恢复增量推理模式
"""

import torch
import torch.nn as nn
import sys
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from transformers.generation.utils import GenerateDecoderOnlyOutput
import re
import os
import av
import numpy as np
from typing import Optional, List, Dict, Any, Union
from ..extras.logging import get_logger

logger = get_logger(__name__)


class Qwen3VLForConditionalGenerationWithDynamicVideo(Qwen3VLForConditionalGeneration):
    """
    增强版Qwen3VL，支持动态视频token插入

    功能：
    1. 检测</fps>token (151672)
    2. 解析<span>时间戳</span><fps>帧率</fps>参数
    3. 重新采样原始视频
    4. 插入新的视频token序列
    5. 继续生成
    """

    def __init__(self, config):
        super().__init__(config)

        # 特殊token定义
        self.fps_end_token_id = 151672      # </fps>
        self.span_start_token_id = 151669   # <span>
        self.span_end_token_id = 151670     # </span>
        self.fps_start_token_id = 151671    # <fps>
        self.vision_start_token_id = 151652 # <|vision_start|>
        self.vision_end_token_id = 151653   # <|vision_end|>
        self.video_pad_token_id = 151656    # <|video_pad|>

        # 动态视频上下文
        self.original_video_paths = None
        self.video_processor = None
        self.processor_args = None

        # 缓存管理
        self._video_features_cache = {}

        # 🎯 按顺序累积每次动态视频插入的特征
        # 解决问题：当同一个 span/fps 被生成多次时，cache hit 会返回 tokens 但不累积 features
        # 这个列表记录每次插入的 features（即使是 cache hit），确保 tokens 和 features 数量匹配
        self._dynamic_features_list = []

    def set_video_context(self, video_paths: List[str], processor, processor_args):
        """设置视频上下文信息"""
        self.original_video_paths = video_paths
        self.video_processor = processor.video_processor if hasattr(processor, 'video_processor') else processor
        self.processor_args = processor_args
        logger.info(f"设置视频上下文: {len(video_paths) if video_paths else 0}个视频")

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus: bool = False,
        streamer=None,
        logits_warper=None,
        **model_kwargs,
    ):
        """
        🚀 重写_sample方法，集成简化的动态视频token插入

        核心策略：
        1. 检测</fps>token时插入视频token
        2. 临时隐藏past_key_values，重置cache_position
        3. 模拟首次推理让Qwen3VL标准处理视频特征
        4. 获得完整KV状态后恢复增量推理模式
        """

        # 🎯 每次生成开始前清空动态特征列表
        # 确保不同 sample 之间的特征不会混淆
        self._dynamic_features_list = []

        # 获取生成配置参数
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        # max_length = generation_config._max_length_tensor
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # 初始化存储
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # 初始化生成状态
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids.shape[-1], input_ids.device, model_kwargs)

        this_peer_finished = False
        cur_len = input_ids.shape[-1]

        # 🚀 保留标准Transformers的所有性能优化
        model_forward = self.__call__

        # 1. 编译优化：判断是否可以编译模型前向传播
        compile_forward = False
        if hasattr(self, '_valid_auto_compile_criteria'):
            compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)

        if compile_forward:
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            # 处理Flash Attention 2的编译兼容性
            if hasattr(self.config, '_attn_implementation') and self.config._attn_implementation == "flash_attention_2":
                if hasattr(generation_config, 'compile_config') and generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                    logger.warning_once(
                        "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                        "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                    )
                    generation_config.compile_config.fullgraph = False
            if hasattr(self, 'get_compiled_call'):
                model_forward = self.get_compiled_call(generation_config.compile_config)

        # 2. 预填充优化：分块处理长序列
        is_prefill = True
        if hasattr(generation_config, 'prefill_chunk_size') and generation_config.prefill_chunk_size is not None:
            if hasattr(self, '_prefill_chunking'):
                model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
                is_prefill = False

        # 🔥🔥🔥 核心生成循环 - 动态视频插入的关键位置 🔥🔥🔥
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            try:
                # 1. 准备模型输入
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

                # 2. 模型前向传播 - 完全兼容标准Transformers的性能优化
                if is_prefill:
                    # 首次推理：使用普通forward，建立KV缓存
                    outputs = self(**model_inputs, return_dict=True)
                    is_prefill = False
                else:
                    # 后续推理：使用编译版本（如果启用），利用缓存加速
                    outputs = model_forward(**model_inputs, return_dict=True)

                # 3. 更新model_kwargs以准备下一轮
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                )

                if synced_gpus and this_peer_finished:
                    continue

                # 4. 处理logits并选择下一个token
                next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
                next_token_scores = logits_processor(input_ids, next_token_logits)

                # 5. 存储分数等信息
                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_logits:
                        raw_logits += (next_token_logits,)
                    if output_attentions:
                        decoder_attentions += (
                            (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                        )
                        if self.config.is_encoder_decoder:
                            cross_attentions += (outputs.cross_attentions,)
                    if output_hidden_states:
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,)
                        )

                # 6. Token采样
                if do_sample:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)

                # 7. 处理EOS
                if has_eos_stopping_criteria:
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                # 🎯🎯🎯 核心修改：简化的动态视频token插入 🎯🎯🎯
                if self.fps_end_token_id in next_tokens:
                    print(f"🚀🚀🚀 [DEBUG] 检测到</fps>token，开始动态视频处理 🚀🚀🚀", file=sys.stderr)
                    print(f"📊📊📊 [DEBUG] 当前序列长度: {input_ids.shape}, next_tokens: {next_tokens}", file=sys.stderr)
                    if 'past_key_values' in model_kwargs and model_kwargs['past_key_values'] is not None:
                        kv_shape = model_kwargs['past_key_values'][0][0].shape
                        print(f"📊📊📊 [DEBUG] 当前KV缓存形状: {kv_shape}", file=sys.stderr)
                    if 'cache_position' in model_kwargs:
                        print(f"📊📊📊 [DEBUG] 当前cache_position: {model_kwargs['cache_position']}", file=sys.stderr)

                    logger.info("🚀 检测到</fps>token，开始动态视频处理")

                    # Step 1: 插入视频token
                    input_ids, model_kwargs = self._handle_dynamic_video_insertion(
                        input_ids, next_tokens, model_kwargs
                    )

                    # Step 2: 检查是否需要模拟首次推理
                    if model_kwargs.get('_dynamic_video_insertion', {}).get('needs_first_inference_simulation', False):

                        # Step 3: 模拟首次推理模式
                        model_kwargs = self._simulate_first_inference_mode(input_ids, model_kwargs)

                        # Step 4: 🚀 关键：重新进行forward，按首次推理处理
                        print(f"🔄🔄🔄 [DEBUG] 重新forward，模拟首次推理 🔄🔄🔄", file=sys.stderr)
                        print(f"📊📊📊 [DEBUG] 模拟首次推理前 input_ids shape: {input_ids.shape}", file=sys.stderr)

                        logger.info("🔄 重新forward，模拟首次推理")
                        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

                        print(f"📊📊📊 [DEBUG] prepare_inputs_for_generation后的输入keys: {list(model_inputs.keys())}", file=sys.stderr)
                        if 'pixel_values_videos' in model_inputs:
                            print(f"📊📊📊 [DEBUG] pixel_values_videos存在，shape: {model_inputs['pixel_values_videos'].shape if model_inputs['pixel_values_videos'] is not None else None}", file=sys.stderr)
                        if 'cache_position' in model_inputs:
                            print(f"📊📊📊 [DEBUG] 模拟首次推理的cache_position: {model_inputs['cache_position']}", file=sys.stderr)

                        # 现在cache_position[0]==0，pixel_values_videos不会被清空！
                        outputs = self(**model_inputs, return_dict=True)

                        print(f"📊📊📊 [DEBUG] 模拟首次推理完成，outputs keys: {list(outputs.keys())}", file=sys.stderr)
                        if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                            new_kv_shape = outputs.past_key_values[0][0].shape
                            print(f"📊📊📊 [DEBUG] 新KV缓存形状: {new_kv_shape}", file=sys.stderr)

                        # forward会正确处理所有视频特征，生成完整的KV状态

                        # Step 5: 恢复增量推理模式
                        model_kwargs = self._restore_incremental_mode(outputs, model_kwargs)

                        print(f"✅✅✅ [DEBUG] 动态视频处理完成，恢复增量推理 ✅✅✅", file=sys.stderr)
                        logger.info("✅ 动态视频处理完成，恢复增量推理")

                    else:
                        # 注意：model_kwargs 已经在第158-160行更新过了
                        # 这里只需要确保 input_ids 已经在 _handle_dynamic_video_insertion 中正确更新
                        pass
                else:
                    # 🚀 标准路径：直接拼接token，与Transformers完全一致
                    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

                    # 注意：model_kwargs 已经在第158-160行更新过了，不需要再次更新
                    # 重复更新会导致 attention_mask 和 cache_position 错误

                # 8. 更新streamer
                if streamer is not None:
                    streamer.put(next_tokens.cpu())

                # 9. 检查停止条件
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
                this_peer_finished = unfinished_sequences.max() == 0
                cur_len += 1

                # 10. 内存清理
                del outputs

            except Exception as e:
                logger.error(f"生成过程中发生错误: {e}")
                import traceback
                traceback.print_exc()
                # 发生严重错误时，应该停止生成而不是继续
                # 继续生成会导致状态不一致，产生错误的结果
                break

        # 清理streamer
        if streamer is not None:
            streamer.end()

        # 🎯 关键修复：在清空缓存之前，保存动态视频特征到持久属性
        # 这样调用者可以在 generate() 返回后获取这些特征
        if self._video_features_cache:
            # 保存到 _last_dynamic_features，不会被 _clear_video_cache 清空
            self._last_dynamic_features = self.get_dynamic_video_features()
            print(f"🎥 [SAMPLE END] 保存 {len(self._last_dynamic_features.get('dynamic_pixel_values_videos', []))} 个动态视频特征")
        else:
            self._last_dynamic_features = {}

        # 清理缓存
        self._clear_video_cache()

        # 返回结果
        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
        else:
            return input_ids

    def _handle_dynamic_video_insertion(
        self,
        input_ids: torch.LongTensor,
        next_tokens: torch.LongTensor,
        model_kwargs: Dict[str, Any]
    ):
        """
        🚀 简化版动态视频token插入处理

        步骤：
        1. 拼接next_tokens
        2. 解析视频参数并生成视频token
        3. 插入视频token到序列中
        4. 设置模拟首次推理标志
        """

        # 1. 先进行正常的token拼接
        updated_input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        # 2. 检查每个batch是否包含fps结束token
        batch_size = input_ids.shape[0]
        video_inserted = False
        has_video_insertion = False

        for batch_idx in range(batch_size):
            if next_tokens[batch_idx].item() == self.fps_end_token_id:
                print(f"🔍🔍🔍 [DEBUG] 检测到</fps>token在batch {batch_idx} 🔍🔍🔍", file=sys.stderr)

                try:
                    # 3. 解析视频参数
                    sequence = updated_input_ids[batch_idx]
                    print(f"📝📝📝 [DEBUG] 准备解析序列，长度: {len(sequence)} 📝📝📝", file=sys.stderr)

                    video_params = self._parse_video_params_from_sequence(sequence)

                    if video_params is None:
                        print(f"❌❌❌ [DEBUG] 视频参数解析失败！video_params=None ❌❌❌", file=sys.stderr)
                        print(f"💡 可能原因：未找到<span>标签或解析格式错误", file=sys.stderr)
                    else:
                        print(f"✅✅✅ [DEBUG] 成功解析到视频参数: {video_params} ✅✅✅", file=sys.stderr)

                        can_insert = self._can_insert_video(batch_idx)
                        print(f"🔒🔒🔒 [DEBUG] _can_insert_video返回: {can_insert} 🔒🔒🔒", file=sys.stderr)

                        if not can_insert:
                            print(f"❌❌❌ [DEBUG] 不能插入视频！可能原因：已经插入过或不满足条件 ❌❌❌", file=sys.stderr)

                    if video_params and self._can_insert_video(batch_idx):
                        print(f"🎯🎯🎯 [DEBUG] 解析到视频参数: {video_params} 🎯🎯🎯", file=sys.stderr)

                        # 4. 生成新的视频token序列
                        video_token_sequence = self._generate_video_tokens(
                            batch_idx, video_params
                        )

                        if len(video_token_sequence) > 0:
                            # 5. 插入视频token到当前位置
                            updated_input_ids = self._insert_video_tokens_at_batch_position(
                                updated_input_ids, batch_idx, video_token_sequence
                            )

                            # 6. 更新model_kwargs中的视频特征
                            model_kwargs = self._update_model_kwargs_with_video_features(
                                model_kwargs, batch_idx, video_params
                            )

                            video_inserted = True
                            print(f"成功插入{len(video_token_sequence)}个视频token")

                except Exception as e:
                    print(f"处理batch {batch_idx}的视频插入时发生错误: {e}")
                    continue

        # 3. 设置模拟首次推理的标志
        if video_inserted:
            print(f"🎯🎯🎯 [DEBUG] 视频插入成功，设置模拟首次推理标志 🎯🎯🎯", file=sys.stderr)
            print(f"📊📊📊 [DEBUG] 插入后序列长度: {updated_input_ids.shape}", file=sys.stderr)

            # 初始化动态视频插入控制参数
            if '_dynamic_video_insertion' not in model_kwargs:
                model_kwargs['_dynamic_video_insertion'] = {}

            model_kwargs['_dynamic_video_insertion']['needs_first_inference_simulation'] = True
            print("🎯 设置模拟首次推理标志")

            # 更新attention_mask
            if 'attention_mask' in model_kwargs:
                old_mask_shape = model_kwargs['attention_mask'].shape
                model_kwargs['attention_mask'] = self._update_attention_mask(
                    model_kwargs['attention_mask'], updated_input_ids
                )
                new_mask_shape = model_kwargs['attention_mask'].shape
                print(f"📊📊📊 [DEBUG] attention_mask更新: {old_mask_shape} -> {new_mask_shape}", file=sys.stderr)
        else:
            print(f"⚠️⚠️⚠️ [DEBUG] 未插入视频，使用标准路径 ⚠️⚠️⚠️", file=sys.stderr)

        return updated_input_ids, model_kwargs

    def _parse_video_params_from_sequence(self, sequence: torch.Tensor) -> Optional[Dict[str, Any]]:
        """
        从token序列解析视频参数

        期望格式: <span>6.50 - 10.50</span><fps>2</fps>
        """
        try:
            sequence_list = sequence.tolist()

            # 查找</fps>位置 (从后往前找最近的一个)
            if self.fps_end_token_id not in sequence_list:
                print(f"❌ 序列中未找到</fps> token (ID={self.fps_end_token_id})", file=sys.stderr)
                return None

            fps_end_idx = len(sequence_list) - 1 - sequence_list[::-1].index(self.fps_end_token_id)
            print(f"📍 找到</fps>位置: idx={fps_end_idx}", file=sys.stderr)

            # 向前查找<span>位置 (最多向前查找50个token)
            span_start_idx = None
            search_start = max(0, fps_end_idx - 50)

            for i in range(fps_end_idx, search_start - 1, -1):
                if sequence_list[i] == self.span_start_token_id:
                    span_start_idx = i
                    break

            if span_start_idx is None:
                print(f"❌ 未找到<span>开始标签 (ID={self.span_start_token_id})，搜索范围: {search_start}~{fps_end_idx}", file=sys.stderr)
                logger.warning("未找到<span>开始标签")
                # 打印最后50个token用于调试
                debug_tokens = sequence_list[max(0, len(sequence_list)-50):]
                debug_text = self.get_tokenizer().decode(debug_tokens, skip_special_tokens=False)
                print(f"🔍 最后50个token解码: {debug_text}", file=sys.stderr)
                return None

            print(f"📍 找到<span>位置: idx={span_start_idx}", file=sys.stderr)

            # 提取相关token并解码
            relevant_tokens = sequence_list[span_start_idx:fps_end_idx + 1]
            decoded_text = self.get_tokenizer().decode(relevant_tokens, skip_special_tokens=False)

            print(f"📝 解码的参数文本: {decoded_text}", file=sys.stderr)
            logger.debug(f"解码的文本: {decoded_text}")

            # 解析文本参数
            result = self._parse_span_fps_text(decoded_text)
            if result:
                print(f"✅ 成功解析参数: {result}", file=sys.stderr)
            else:
                print(f"❌ 参数文本格式不匹配", file=sys.stderr)
            return result

        except Exception as e:
            logger.error(f"解析视频参数时发生错误: {e}")
            return None

    def _parse_span_fps_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        解析<span>6.50 - 10.50</span><fps>2</fps>格式的文本
        """
        try:
            # 匹配时间范围
            span_pattern = r'<span>\s*([\d.]+)\s*[-–—]\s*([\d.]+)\s*</span>'
            fps_pattern = r'<fps>\s*(\d+)\s*</fps>'

            span_match = re.search(span_pattern, text)
            fps_match = re.search(fps_pattern, text)

            if span_match and fps_match:
                start_time = float(span_match.group(1))
                end_time = float(span_match.group(2))
                fps = int(fps_match.group(1))

                # 验证参数合理性
                if start_time >= end_time:
                    logger.warning(f"无效的时间范围: {start_time} >= {end_time}")
                    return None

                if fps <= 0 or fps > 30:
                    logger.warning(f"无效的fps值: {fps}")
                    return None

                return {
                    'start_time': start_time,
                    'end_time': end_time,
                    'fps': fps,
                    'duration': end_time - start_time
                }
            else:
                logger.warning(f"无法解析文本格式: {text}")
                return None

        except Exception as e:
            logger.error(f"解析span/fps文本时发生错误: {e}")
            return None

    def _can_insert_video(self, batch_idx: int) -> bool:
        """检查是否可以为指定batch插入视频"""
        # 详细检查每个条件
        print(f"🔍 检查视频插入条件 (batch {batch_idx}):", file=sys.stderr)

        cond1 = self.original_video_paths is not None
        print(f"  条件1 - original_video_paths 不为 None: {cond1}", file=sys.stderr)
        if not cond1:
            print(f"    ❌ self.original_video_paths = {self.original_video_paths}", file=sys.stderr)
            return False

        cond2 = batch_idx < len(self.original_video_paths)
        print(f"  条件2 - batch_idx({batch_idx}) < len(paths)({len(self.original_video_paths)}): {cond2}", file=sys.stderr)
        if not cond2:
            return False

        cond3 = self.original_video_paths[batch_idx] is not None
        print(f"  条件3 - paths[{batch_idx}] 不为 None: {cond3}", file=sys.stderr)
        if not cond3:
            print(f"    ❌ self.original_video_paths[{batch_idx}] = {self.original_video_paths[batch_idx]}", file=sys.stderr)
            return False

        video_path = self.original_video_paths[batch_idx]
        cond4 = os.path.exists(video_path)
        print(f"  条件4 - 视频文件存在 '{video_path}': {cond4}", file=sys.stderr)
        if not cond4:
            print(f"    ❌ 文件不存在", file=sys.stderr)
            return False

        cond5 = self.video_processor is not None
        print(f"  条件5 - video_processor 不为 None: {cond5}", file=sys.stderr)
        if not cond5:
            print(f"    ❌ self.video_processor = {self.video_processor}", file=sys.stderr)
            return False

        print(f"  ✅ 所有条件满足，可以插入视频", file=sys.stderr)
        return True

    def _generate_video_tokens(
        self,
        batch_idx: int,
        video_params: Dict[str, Any]
    ) -> torch.Tensor:
        """
        根据视频参数生成新的视频token序列

        步骤：
        1. 重新采样视频
        2. 生成视频特征
        3. 构建token序列
        """
        try:
            video_path = self.original_video_paths[batch_idx]
            cache_key = f"{video_path}_{video_params['start_time']}_{video_params['end_time']}_{video_params['fps']}"

            # 检查缓存
            if cache_key in self._video_features_cache:
                logger.debug(f"使用缓存的视频特征: {cache_key}")
                cached_data = self._video_features_cache[cache_key]
                # 🎯 关键修复：即使 cache hit，也要把 features 加入列表
                # 这样当同一个 span/fps 被生成多次时，features 数量与 tokens 数量匹配
                self._dynamic_features_list.append(cached_data['features'])
                print(f"🔄 [CACHE HIT] 添加缓存的 features 到列表，当前列表长度: {len(self._dynamic_features_list)}", file=sys.stderr)
                return cached_data['tokens'].to(self.device)

            # 1. 重新采样视频
            logger.info(f"重新采样视频: {video_path}, {video_params['start_time']}-{video_params['end_time']}s, {video_params['fps']}fps")
            new_frames = self._resample_video_with_time_range(
                video_path,
                video_params['start_time'],
                video_params['end_time'],
                video_params['fps']
            )

            if not new_frames:
                logger.warning("视频采样失败，返回空token序列")
                return torch.tensor([], device=self.device, dtype=torch.long)

            logger.info(f"采样得到 {len(new_frames)} 帧")

            # 2. 生成视频特征
            # 🎯 创建正确的VideoMetadata对象，确保fps参数正确传递
            from transformers.video_utils import VideoMetadata

            # 创建正确的VideoMetadata
            video_metadata = VideoMetadata(
                total_num_frames=len(new_frames),
                fps=video_params['fps'],  # 使用我们指定的fps
                width=new_frames[0].width,
                height=new_frames[0].height,
                video_backend="pil",  # 使用PIL backend
                frames_indices=list(range(len(new_frames)))
            )

            # 调用视频处理器，传递正确的metadata作为参数
            video_features = self.video_processor(
                videos=[new_frames],
                return_tensors="pt",
                return_metadata=True,
                fps=video_params['fps'],
                video_metadata=[video_metadata]  # 🎯 关键修复：直接传递metadata参数
            )

            # 移动到正确设备
            video_features = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in video_features.items()}

            # 3. 构建视频token序列
            video_tokens = self._build_video_token_sequence(video_features, video_params)

            # 4. 缓存结果
            features_to_cache = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                                 for k, v in video_features.items()}
            self._video_features_cache[cache_key] = {
                'tokens': video_tokens.cpu(),
                'features': features_to_cache,
                'params': video_params
            }

            # 🎯 关键修复：cache miss 时也要把 features 加入列表
            self._dynamic_features_list.append(features_to_cache)
            print(f"🆕 [CACHE MISS] 添加新 features 到列表，当前列表长度: {len(self._dynamic_features_list)}", file=sys.stderr)

            return video_tokens

        except Exception as e:
            logger.error(f"生成视频token时发生错误: {e}")
            return torch.tensor([], device=self.device, dtype=torch.long)

    def _resample_video_with_time_range(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        fps: int
    ) -> List:
        """
        根据时间范围和fps重新采样视频
        """
        try:
            # 🎬🎬🎬 [DEBUG] 开始视频重采样过程 🎬🎬🎬
            print(f"🎬🎬🎬 [DEBUG] _resample_video_with_time_range()被调用! 🎬🎬🎬", file=sys.stderr)
            print(f"📁📁📁 [DEBUG] 视频路径: {video_path} 📁📁📁", file=sys.stderr)
            print(f"⏰⏰⏰ [DEBUG] 时间范围: {start_time}s - {end_time}s (时长: {end_time-start_time}s) ⏰⏰⏰", file=sys.stderr)
            print(f"🎯🎯🎯 [DEBUG] 目标FPS: {fps} 🎯🎯🎯", file=sys.stderr)

            with av.open(video_path, "r", metadata_errors="ignore") as container:
                video_stream = next((stream for stream in container.streams if stream.type == "video"), None)

                if video_stream is None:
                    print(f"❌❌❌ [ERROR] 视频文件中未找到视频流: {video_path} ❌❌❌", file=sys.stderr)
                    logger.error(f"视频文件中未找到视频流: {video_path}")
                    return []

                # 计算采样参数
                stream_fps = float(video_stream.average_rate)
                start_frame_idx = int(start_time * stream_fps)
                end_frame_idx = int(end_time * stream_fps)

                duration = end_time - start_time
                target_frame_count = int(duration * fps)

                print(f"📊📊📊 [DEBUG] 视频流信息: 原始FPS={stream_fps:.2f} 📊📊📊", file=sys.stderr)
                print(f"🔢🔢🔢 [DEBUG] 帧索引计算: 起始帧={start_frame_idx}, 结束帧={end_frame_idx} 🔢🔢🔢", file=sys.stderr)
                print(f"🎯🎯🎯 [DEBUG] 目标帧数计算: {duration}s × {fps}fps = {target_frame_count}帧 🎯🎯🎯", file=sys.stderr)

                if target_frame_count <= 0:
                    print(f"⚠️⚠️⚠️ [WARNING] 目标帧数无效: {target_frame_count} ⚠️⚠️⚠️", file=sys.stderr)
                    logger.warning(f"目标帧数无效: {target_frame_count}")
                    return []

                # 限制最大帧数
                max_frames = getattr(self.processor_args, 'video_maxlen', 128)
                original_target = target_frame_count
                target_frame_count = min(target_frame_count, max_frames)

                if target_frame_count < original_target:
                    print(f"⚠️⚠️⚠️ [WARNING] 帧数被限制: {original_target} -> {target_frame_count} (max_frames={max_frames}) ⚠️⚠️⚠️", file=sys.stderr)

                # 计算采样索引
                frame_indices = np.linspace(start_frame_idx, end_frame_idx, target_frame_count, dtype=int)
                frame_indices = np.unique(frame_indices)  # 去重

                print(f"📋📋📋 [DEBUG] 采样索引计算完成: {len(frame_indices)}个帧 📋📋📋", file=sys.stderr)
                print(f"📝📝📝 [DEBUG] 帧索引列表: {frame_indices.tolist()} 📝📝📝", file=sys.stderr)

                # 计算每帧对应的时间戳
                frame_timestamps = []
                for i, idx in enumerate(frame_indices):
                    timestamp = start_time + (idx - start_frame_idx) / stream_fps
                    frame_timestamps.append(timestamp)
                    print(f"⏱️⏱️⏱️ [DEBUG] 帧{i+1}: 索引{idx} -> 时间戳 {timestamp:.3f}s ⏱️⏱️⏱️", file=sys.stderr)

                # 提取帧
                frames = []
                extracted_count = 0

                # 🔧 修复：将Fraction转换为整数时间戳
                seek_timestamp = int(start_frame_idx * video_stream.time_base)
                container.seek(seek_timestamp)

                print(f"🔍🔍🔍 [DEBUG] 开始提取帧... 🔍🔍🔍", file=sys.stderr)

                for frame_idx, frame in enumerate(container.decode(video_stream)):
                    current_idx = start_frame_idx + frame_idx
                    current_timestamp = start_time + frame_idx / stream_fps

                    if current_idx > end_frame_idx:
                        break

                    if current_idx in frame_indices:
                        frames.append(frame.to_image())
                        extracted_count += 1
                        target_idx_pos = np.where(frame_indices == current_idx)[0][0]
                        target_timestamp = frame_timestamps[target_idx_pos]
                        print(f"✅✅✅ [DEBUG] 提取帧{extracted_count}: 索引{current_idx}, 实际时间={current_timestamp:.3f}s, 目标时间={target_timestamp:.3f}s ✅✅✅", file=sys.stderr)

                print(f"🎉🎉🎉 [DEBUG] 采样完成! 成功提取 {len(frames)} 帧 🎉🎉🎉", file=sys.stderr)

                if len(frames) > 0:
                    actual_start_time = frame_timestamps[0]
                    actual_end_time = frame_timestamps[-1]
                    actual_duration = actual_end_time - actual_start_time
                    actual_fps = (len(frames) - 1) / actual_duration if actual_duration > 0 else fps

                    print(f"📈📈📈 [DEBUG] 采样统计: 📈📈📈", file=sys.stderr)
                    print(f"   📍 目标时间范围: {start_time:.3f}s - {end_time:.3f}s", file=sys.stderr)
                    print(f"   📍 实际时间范围: {actual_start_time:.3f}s - {actual_end_time:.3f}s", file=sys.stderr)
                    print(f"   📍 实际采样率: {actual_fps:.2f} fps (目标: {fps} fps)", file=sys.stderr)

                logger.info(f"成功提取 {len(frames)} 帧")
                return frames

        except Exception as e:
            print(f"❌❌❌ [ERROR] 视频采样失败: {e} ❌❌❌", file=sys.stderr)
            logger.error(f"视频采样失败: {e}")
            import traceback
            print(f"🔍🔍🔍 [DEBUG] 详细错误堆栈: 🔍🔍🔍", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return []

    def _build_video_token_sequence(
        self,
        video_features: Dict[str, Any],
        video_params: Dict[str, Any]
    ) -> torch.Tensor:
        """
        构建视频token序列

        格式: <时间戳><|vision_start|><|video_pad|>...<|vision_end|><时间戳><|vision_start|>...
        """
        try:
            # 🎪🎪🎪 [DEBUG] 开始构建视频token序列 🎪🎪🎪
            print(f"🎪🎪🎪 [DEBUG] _build_video_token_sequence()被调用! 🎪🎪🎪", file=sys.stderr)
            print(f"🎬🎬🎬 [DEBUG] 视频参数: {video_params} 🎬🎬🎬", file=sys.stderr)

            video_grid_thw = video_features.get('video_grid_thw')
            if video_grid_thw is None or len(video_grid_thw) == 0:
                print(f"❌❌❌ [ERROR] video_grid_thw为空或不存在 ❌❌❌", file=sys.stderr)
                return torch.tensor([], device=self.device, dtype=torch.long)

            video_grid_thw = video_grid_thw[0]  # [T, H, W]
            num_frames = video_grid_thw[0].item()

            print(f"📊📊📊 [DEBUG] video_grid_thw: {video_grid_thw} 📊📊📊", file=sys.stderr)
            print(f"🎞️🎞️🎞️ [DEBUG] 处理帧数: {num_frames} 🎞️🎞️🎞️", file=sys.stderr)

            # 计算每帧的token数量
            merge_size = getattr(self.video_processor, 'merge_size', 2)
            tokens_per_frame = (video_grid_thw[1] * video_grid_thw[2]) // (merge_size ** 2)
            tokens_per_frame = tokens_per_frame.item()

            print(f"🔢🔢🔢 [DEBUG] merge_size: {merge_size}, 每帧token数: {tokens_per_frame} 🔢🔢🔢", file=sys.stderr)

            video_tokens = []
            tokenizer = self.get_tokenizer()

            print(f"🏗️🏗️🏗️ [DEBUG] 开始逐帧构建token序列... 🏗️🏗️🏗️", file=sys.stderr)

            for frame_idx in range(num_frames):
                # 计算当前帧的时间戳
                progress = frame_idx / max(1, num_frames - 1) if num_frames > 1 else 0.0
                timestamp = video_params['start_time'] + progress * video_params['duration']

                # 编码时间戳
                timestamp_text = f"<{timestamp:.1f} seconds>"
                timestamp_tokens = tokenizer.encode(timestamp_text, add_special_tokens=False)

                # 构建当前帧的token序列
                frame_tokens = []
                frame_tokens.extend(timestamp_tokens)
                frame_tokens.append(self.vision_start_token_id)
                frame_tokens.extend([self.video_pad_token_id] * tokens_per_frame)
                frame_tokens.append(self.vision_end_token_id)

                video_tokens.extend(frame_tokens)

                print(f"🎭🎭🎭 [DEBUG] 帧{frame_idx+1}: 时间戳='{timestamp_text}' ({len(timestamp_tokens)}个token), 视觉token={tokens_per_frame}个, 总计={len(frame_tokens)}个token 🎭🎭🎭", file=sys.stderr)

            result = torch.tensor(video_tokens, device=self.device, dtype=torch.long)

            print(f"🎉🎉🎉 [DEBUG] token序列构建完成! 🎉🎉🎉", file=sys.stderr)
            print(f"📊📊📊 [DEBUG] 构建统计: 📊📊📊", file=sys.stderr)
            print(f"   🎞️ 处理帧数: {num_frames}", file=sys.stderr)
            print(f"   🔢 每帧token数: {tokens_per_frame}", file=sys.stderr)
            print(f"   📝 总token数: {len(result)}", file=sys.stderr)
            print(f"   ⏰ 时间范围: {video_params['start_time']:.1f}s - {video_params['start_time'] + video_params['duration']:.1f}s", file=sys.stderr)

            # 🔍 输出前几个和后几个token用于验证
            if len(result) > 0:
                preview_len = min(20, len(result))
                preview_tokens = result[:preview_len].tolist()
                decoded_preview = tokenizer.decode(preview_tokens, skip_special_tokens=False)
                print(f"👀👀👀 [DEBUG] token序列预览(前{preview_len}个): {preview_tokens} 👀👀👀", file=sys.stderr)
                print(f"📖📖📖 [DEBUG] 预览解码: {repr(decoded_preview)} 📖📖📖", file=sys.stderr)

            logger.debug(f"构建完成，总计 {len(result)} 个token")

            return result

        except Exception as e:
            print(f"❌❌❌ [ERROR] 构建视频token序列失败: {e} ❌❌❌", file=sys.stderr)
            logger.error(f"构建视频token序列失败: {e}")
            import traceback
            print(f"🔍🔍🔍 [DEBUG] 详细错误堆栈: 🔍🔍🔍", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return torch.tensor([], device=self.device, dtype=torch.long)

    def _insert_video_tokens_at_batch_position(
        self,
        input_ids: torch.Tensor,
        batch_idx: int,
        video_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        在指定batch的当前位置插入视频token
        """
        if len(video_tokens) == 0:
            return input_ids

        try:
            batch_size, seq_len = input_ids.shape
            new_seq_len = seq_len + len(video_tokens)

            # 创建新的input_ids tensor
            pad_token_id = self.get_tokenizer().pad_token_id
            new_input_ids = torch.full(
                (batch_size, new_seq_len),
                pad_token_id,
                device=input_ids.device,
                dtype=input_ids.dtype
            )

            # 复制每个batch的数据
            for i in range(batch_size):
                if i == batch_idx:
                    # 当前batch: 原序列 + 视频tokens
                    old_seq = input_ids[i]
                    new_sequence = torch.cat([old_seq, video_tokens])
                    new_input_ids[i, :len(new_sequence)] = new_sequence
                else:
                    # 其他batch: 原序列 + padding
                    old_seq = input_ids[i]
                    new_input_ids[i, :len(old_seq)] = old_seq

            return new_input_ids

        except Exception as e:
            logger.error(f"插入视频token失败: {e}")
            return input_ids

    def _update_model_kwargs_with_video_features(
        self,
        model_kwargs: Dict[str, Any],
        batch_idx: int,
        video_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        🚀 更新model_kwargs中的视频特征，将新的视频特征添加到现有特征中
        """
        try:
            print(f"🔄🔄🔄 [DEBUG] 更新batch {batch_idx}的视频特征 🔄🔄🔄", file=sys.stderr)

            # 获取缓存的视频特征
            cache_key = f"{self.original_video_paths[batch_idx]}_{video_params['start_time']}_{video_params['end_time']}_{video_params['fps']}"

            if cache_key in self._video_features_cache:
                cached_data = self._video_features_cache[cache_key]
                new_video_features = cached_data['features']

                print(f"📊📊📊 [DEBUG] 获取缓存的视频特征", file=sys.stderr)

                # 🎯 关键修复：将新的视频特征添加到现有的pixel_values_videos中
                if 'pixel_values_videos' in model_kwargs and model_kwargs['pixel_values_videos'] is not None:
                    # 获取现有的视频特征
                    existing_video_features = model_kwargs['pixel_values_videos']
                    print(f"📊📊📊 [DEBUG] 现有视频特征shape: {existing_video_features.shape}", file=sys.stderr)

                    # 获取新的视频特征
                    if 'pixel_values_videos' in new_video_features:
                        new_pixel_values = new_video_features['pixel_values_videos'].to(existing_video_features.device)
                        print(f"📊📊📊 [DEBUG] 新视频特征shape: {new_pixel_values.shape}", file=sys.stderr)

                        # 拼接视频特征
                        combined_video_features = torch.cat([existing_video_features, new_pixel_values], dim=0)
                        model_kwargs['pixel_values_videos'] = combined_video_features
                        print(f"📊📊📊 [DEBUG] 合并后视频特征shape: {combined_video_features.shape}", file=sys.stderr)

                # 🎯 同样更新video_grid_thw
                if 'video_grid_thw' in model_kwargs and 'video_grid_thw' in new_video_features:
                    existing_grid_thw = model_kwargs['video_grid_thw']
                    new_grid_thw = new_video_features['video_grid_thw'].to(existing_grid_thw.device)

                    print(f"📊📊📊 [DEBUG] 现有video_grid_thw: {existing_grid_thw}", file=sys.stderr)
                    print(f"📊📊📊 [DEBUG] 新video_grid_thw: {new_grid_thw}", file=sys.stderr)

                    # 拼接grid_thw
                    combined_grid_thw = torch.cat([existing_grid_thw, new_grid_thw], dim=0)
                    model_kwargs['video_grid_thw'] = combined_grid_thw
                    print(f"📊📊📊 [DEBUG] 合并后video_grid_thw: {combined_grid_thw}", file=sys.stderr)

                print(f"✅✅✅ [DEBUG] 视频特征更新完成 ✅✅✅", file=sys.stderr)
            else:
                print(f"⚠️⚠️⚠️ [DEBUG] 未找到缓存的视频特征: {cache_key} ⚠️⚠️⚠️", file=sys.stderr)

            logger.debug(f"更新batch {batch_idx}的模型参数")
            return model_kwargs

        except Exception as e:
            print(f"❌❌❌ [DEBUG] 更新视频特征失败: {e} ❌❌❌", file=sys.stderr)
            logger.error(f"更新模型参数失败: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
            return model_kwargs

    def _update_attention_mask(
        self,
        attention_mask: torch.Tensor,
        new_input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        根据新的input_ids更新attention_mask
        """
        try:
            batch_size, new_seq_len = new_input_ids.shape
            pad_token_id = self.get_tokenizer().pad_token_id

            # 创建新的attention_mask
            new_attention_mask = (new_input_ids != pad_token_id).long()

            return new_attention_mask

        except Exception as e:
            logger.error(f"更新attention_mask失败: {e}")
            # 返回全1的mask作为备选
            return torch.ones_like(new_input_ids)

    def _simulate_first_inference_mode(self, input_ids: torch.Tensor, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        🚀 模拟首次推理模式：临时隐藏past_key_values，重置cache_position

        核心策略：
        1. 保存当前KV缓存到临时位置
        2. 清空past_key_values，让模型以为是首次推理
        3. 重置cache_position为0，确保prepare_inputs_for_generation不清空视频特征
        """
        try:
            print(f"🔄🔄🔄 [DEBUG] 开始模拟首次推理模式 🔄🔄🔄", file=sys.stderr)
            logger.info("🔄 开始模拟首次推理模式")

            # 1. 保存当前的KV缓存
            if 'past_key_values' in model_kwargs:
                print(f"💾💾💾 [DEBUG] 保存并隐藏past_key_values，原KV形状: {model_kwargs['past_key_values'][0][0].shape if model_kwargs['past_key_values'] else None}", file=sys.stderr)
                model_kwargs['_dynamic_video_insertion']['saved_past_key_values'] = model_kwargs['past_key_values']
                model_kwargs.pop('past_key_values')  # 临时移除，模拟首次推理
                logger.info("💾 已保存并隐藏past_key_values")

            # 2. 重置cache_position为0，模拟首次推理
            if 'cache_position' in model_kwargs:
                print(f"🔄🔄🔄 [DEBUG] 保存原cache_position: {model_kwargs['cache_position']}", file=sys.stderr)
                model_kwargs['_dynamic_video_insertion']['saved_cache_position'] = model_kwargs['cache_position']
                # 创建新的cache_position，从0开始到input_ids长度
                new_cache_position = torch.arange(0, input_ids.shape[-1], device=input_ids.device)
                model_kwargs['cache_position'] = new_cache_position
                print(f"🔄🔄🔄 [DEBUG] 重置cache_position: {new_cache_position} (0 到 {input_ids.shape[-1]-1})", file=sys.stderr)
                logger.info(f"🔄 重置cache_position: {new_cache_position.shape} (0 到 {input_ids.shape[-1]-1})")

            # 3. 确保像素值不会被清空
            print(f"✅✅✅ [DEBUG] 模拟首次推理模式设置完成，现在prepare_inputs_for_generation不会清空视频特征", file=sys.stderr)
            logger.info("✅ 模拟首次推理模式设置完成，现在prepare_inputs_for_generation不会清空视频特征")

            return model_kwargs

        except Exception as e:
            logger.error(f"❌ 模拟首次推理模式失败: {e}")
            return model_kwargs

    def _restore_incremental_mode(self, outputs, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        🔄 恢复增量推理模式：从outputs获取新的KV状态，恢复正常的cache_position

        核心策略：
        1. 从forward输出中获取完整的KV状态
        2. 恢复正常的cache_position，指向最后一个位置
        3. 清理临时标志
        """
        try:
            print(f"🔄🔄🔄 [DEBUG] 开始恢复增量推理模式 🔄🔄🔄", file=sys.stderr)
            logger.info("🔄 开始恢复增量推理模式")

            # 1. 从outputs中获取新的完整KV缓存
            if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                print(f"💾💾💾 [DEBUG] 从outputs恢复完整的past_key_values，新KV形状: {outputs.past_key_values[0][0].shape}", file=sys.stderr)
                model_kwargs['past_key_values'] = outputs.past_key_values
                logger.info("💾 从outputs恢复完整的past_key_values")

            # 2. 恢复正常的cache_position
            dynamic_info = model_kwargs.get('_dynamic_video_insertion', {})
            if 'saved_cache_position' in dynamic_info:
                original_cache_pos = dynamic_info['saved_cache_position']
                print(f"🔄🔄🔄 [DEBUG] 原始cache_position: {original_cache_pos}", file=sys.stderr)

                # 🎯 关键修复：cache_position应该指向下一个待生成的位置
                # 在transformers中，cache_position表示当前要写入KV缓存的位置
                # 由于我们刚刚完成了forward，KV缓存已经包含了完整序列的状态
                # 下一个token的cache_position应该是当前序列长度

                # 从past_key_values的形状推断当前序列长度
                if 'past_key_values' in model_kwargs and model_kwargs['past_key_values'] is not None:
                    # past_key_values的第二维是序列长度维度
                    current_seq_len = model_kwargs['past_key_values'][0][0].shape[-2]
                    new_cache_position = torch.tensor([current_seq_len], device=original_cache_pos.device)
                    model_kwargs['cache_position'] = new_cache_position
                    print(f"🔄🔄🔄 [DEBUG] 从KV缓存恢复cache_position到: {new_cache_position} (序列长度: {current_seq_len})", file=sys.stderr)
                    logger.info(f"🔄 从KV缓存恢复cache_position到: {new_cache_position} (序列长度: {current_seq_len})")
                else:
                    # 备用方案：基于原始cache_position计算
                    # 假设插入了N个视频token，cache_position需要相应增加
                    new_cache_position = torch.tensor([original_cache_pos[-1].item() + 1], device=original_cache_pos.device)
                    model_kwargs['cache_position'] = new_cache_position
                    print(f"🔄🔄🔄 [DEBUG] 基于原始位置恢复cache_position到: {new_cache_position}", file=sys.stderr)
                    logger.info(f"🔄 基于原始位置恢复cache_position到: {new_cache_position}")

            # 3. 清理动态视频插入的临时数据
            if '_dynamic_video_insertion' in model_kwargs:
                # 保留一些有用的信息，清理临时标志
                dynamic_info = model_kwargs['_dynamic_video_insertion']
                print(f"🧹🧹🧹 [DEBUG] 清理临时标志，当前dynamic_info keys: {list(dynamic_info.keys())}", file=sys.stderr)
                dynamic_info.pop('needs_first_inference_simulation', None)
                dynamic_info.pop('saved_past_key_values', None)
                dynamic_info.pop('saved_cache_position', None)

                if not dynamic_info:  # 如果字典为空，完全移除
                    model_kwargs.pop('_dynamic_video_insertion')
                    print(f"🧹🧹🧹 [DEBUG] 完全移除_dynamic_video_insertion", file=sys.stderr)

            print(f"✅✅✅ [DEBUG] 增量推理模式恢复完成 ✅✅✅", file=sys.stderr)
            logger.info("✅ 增量推理模式恢复完成")

            return model_kwargs

        except Exception as e:
            logger.error(f"❌ 恢复增量推理模式失败: {e}")
            return model_kwargs

    def _clear_video_cache(self):
        """清理视频特征缓存"""
        if self._video_features_cache:
            logger.debug(f"清理 {len(self._video_features_cache)} 个视频缓存")
            self._video_features_cache.clear()
        # 🎯 同时清空动态特征列表
        if self._dynamic_features_list:
            logger.debug(f"清理 {len(self._dynamic_features_list)} 个动态特征")
            self._dynamic_features_list.clear()

    def get_dynamic_video_features(self) -> Dict[str, Any]:
        """
        获取动态插入的视频特征（用于后续的 logps 计算）

        🎯 修复：从 _dynamic_features_list 获取（按插入顺序累积），而不是从 _video_features_cache（只有 unique keys）
        这样当同一个 span/fps 被生成多次时，能正确返回多份 features

        Returns:
            包含所有动态插入的视频特征的字典，格式：
            {
                'dynamic_pixel_values_videos': List[Tensor],  # 每个动态视频的 pixel values
                'dynamic_video_grid_thw': List[Tensor],       # 每个动态视频的 grid 信息
            }
        """
        # 🎯 从列表获取（包含重复的 features）
        if not self._dynamic_features_list:
            return {}

        all_pixel_values = []
        all_grid_thw = []

        for features in self._dynamic_features_list:
            if 'pixel_values_videos' in features:
                pv = features['pixel_values_videos']
                if isinstance(pv, torch.Tensor):
                    all_pixel_values.append(pv)
            if 'video_grid_thw' in features:
                gt = features['video_grid_thw']
                if isinstance(gt, torch.Tensor):
                    all_grid_thw.append(gt)

        result = {}
        if all_pixel_values:
            result['dynamic_pixel_values_videos'] = all_pixel_values
        if all_grid_thw:
            result['dynamic_video_grid_thw'] = all_grid_thw

        print(f"🎥 [GET_DYNAMIC_FEATURES] 从列表获取 {len(all_pixel_values)} 份 features", file=sys.stderr)

        return result

    def clear_and_get_dynamic_video_features(self) -> Dict[str, Any]:
        """
        获取动态视频特征后清理缓存

        这是推荐的方式：生成结束后调用此方法获取特征，然后缓存被清理
        """
        features = self.get_dynamic_video_features()
        self._clear_video_cache()
        return features

    def get_last_dynamic_features(self) -> Dict[str, Any]:
        """
        获取上一次 _sample 结束时保存的动态视频特征

        这个方法用于在 generate() 返回后获取动态视频特征。
        因为 _sample 在返回前会清空 _video_features_cache，所以我们在清空前
        把特征保存到 _last_dynamic_features 中。

        Returns:
            上一次生成时的动态视频特征，格式同 get_dynamic_video_features()
        """
        return getattr(self, '_last_dynamic_features', {})

    def get_tokenizer(self):
        """获取tokenizer的便利方法"""
        # 尝试从不同位置获取tokenizer
        if hasattr(self, 'tokenizer'):
            return self.tokenizer
        elif hasattr(self, '_tokenizer'):
            return self._tokenizer
        else:
            # 从全局注册中获取
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(self.config._name_or_path)