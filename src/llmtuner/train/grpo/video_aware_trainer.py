"""
Video-aware GRPO trainer for Qwen3VL VMCOT.

This module extends TRL's GRPOTrainer to handle video inputs during generation,
specifically customized for Qwen3VL's video multi-frame chain of thought.
"""

import os
import re
import sys
import torch
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
from trl import GRPOTrainer
from trl.trainer.utils import pad

from ...extras.logging import get_logger
from ...data.utils import Role

logger = get_logger(__name__)


def find_timestamp_token_positions(token_ids: List[int], tokenizer) -> List[int]:
    """
    识别系统自动插入的 timestamp tokens 的位置。

    Timestamp 格式: <X.X seconds> 或 <XX.X seconds> 或 <XXX.X seconds>
    例如: <26.0 seconds>, <126.5 seconds>

    这些 timestamps 是系统根据 <span> 和 <fps> 自动计算并插入的，
    不是模型的"选择"，因此不应该参与 loss 计算。

    Args:
        token_ids: completion 的 token IDs
        tokenizer: tokenizer 用于解码

    Returns:
        需要 mask 的 token 位置列表
    """
    # 将 token_ids 解码为文本
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    # 获取每个 token 对应的文本
    # 使用逐 token 解码来精确定位
    positions_to_mask = []

    # 方法：滑动窗口检测 timestamp 模式
    # Timestamp 通常由多个 tokens 组成，如 ["<", "26", ".", "0", " seconds", ">"]
    # 我们需要找到完整的 <X.X seconds> 模式

    # 先解码整个序列，找到 timestamp 的字符位置
    full_text = tokenizer.decode(token_ids, skip_special_tokens=False)

    # 匹配 timestamp 模式: <数字 seconds> 或 <数字.数字 seconds>
    # 支持: <26 seconds>, <26.0 seconds>, <126.5 seconds>
    timestamp_pattern = r'<\d+(?:\.\d+)?\s*seconds>'

    # 找到所有 timestamp 的字符位置
    timestamp_spans = []
    for match in re.finditer(timestamp_pattern, full_text):
        timestamp_spans.append((match.start(), match.end()))

    if not timestamp_spans:
        return []

    # 现在需要将字符位置映射回 token 位置
    # 逐 token 解码，累积字符位置
    char_pos = 0
    token_char_ranges = []  # [(start, end), ...] 每个 token 的字符范围

    for i, tid in enumerate(token_ids):
        token_text = tokenizer.decode([tid], skip_special_tokens=False)
        token_start = char_pos
        token_end = char_pos + len(token_text)
        token_char_ranges.append((token_start, token_end))
        char_pos = token_end

    # 找到与 timestamp spans 重叠的 tokens
    for ts_start, ts_end in timestamp_spans:
        for token_idx, (tok_start, tok_end) in enumerate(token_char_ranges):
            # 检查是否有重叠
            if tok_start < ts_end and tok_end > ts_start:
                positions_to_mask.append(token_idx)

    return list(set(positions_to_mask))  # 去重


class VideoAwareGRPOTrainer(GRPOTrainer):
    """Extended GRPO trainer that handles video inputs for Qwen3VL VMCOT."""

    def __init__(self, *args, **kwargs):
        # 🎯 提取 ref_model 参数（TRL 的 GRPOTrainer 不接受此参数）
        # 我们需要在 super().__init__() 之后手动设置
        custom_ref_model = kwargs.pop('ref_model', None)

        # 🎯 关键修复：如果提供了自定义 ref_model，临时设置 beta=0 来跳过 TRL 的自动创建
        # TRL 在 beta != 0 且非 PEFT 时会调用 create_model_from_path()，这会失败
        # 因为我们的自定义类 Qwen3VLForConditionalGenerationWithDynamicVideo 不在 transformers 中
        original_beta = None
        if custom_ref_model is not None and 'args' in kwargs:
            grpo_args = kwargs['args']
            if hasattr(grpo_args, 'beta') and grpo_args.beta != 0.0:
                original_beta = grpo_args.beta
                grpo_args.beta = 0.0  # 临时设为 0，跳过 ref_model 自动创建
                logger.info(f"🔧 临时设置 beta=0 以跳过 TRL 的 ref_model 自动创建 (原始 beta={original_beta})")

        super().__init__(*args, **kwargs)

        # 🎯 恢复原始 beta 值并设置自定义 ref_model
        if custom_ref_model is not None:
            if original_beta is not None:
                self.beta = original_beta  # 恢复原始 beta
                logger.info(f"🔧 恢复 beta={original_beta}")

            # 🎯 对 ref_model 进行 DeepSpeed/FSDP 准备（TRL 在 __init__ 中做这个，但我们跳过了）
            from trl.models.utils import prepare_deepspeed, prepare_fsdp
            if self.is_deepspeed_enabled:
                custom_ref_model = prepare_deepspeed(custom_ref_model, self.accelerator)
                logger.info("✅ ref_model 已进行 DeepSpeed 准备")
            elif self.is_fsdp_enabled:
                custom_ref_model = prepare_fsdp(custom_ref_model, self.accelerator)
                logger.info("✅ ref_model 已进行 FSDP 准备")
            else:
                custom_ref_model = self.accelerator.prepare_model(custom_ref_model, evaluation_mode=True)
                logger.info("✅ ref_model 已进行 accelerator 准备")

            self.ref_model = custom_ref_model
            logger.info("✅ 使用自定义 ref_model（与训练模型使用相同的模型类）")

        # Initialize processor for video handling
        self.processor = None
        self.template = None  # Will be set by trainer

    def _split_pixel_values_videos_by_grid(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        类似 TRL 的 split_pixel_values_by_grid，但用于视频特征。

        将 concatenated tensor 格式的视频特征切分为 list[Tensor] 格式，
        以便 TRL 的 shuffle_sequence_dict 和 split_tensor_dict 能正确处理。
        """
        if "video_grid_thw" not in batch or "pixel_values_videos" not in batch:
            return batch
        if "num_videos_per_sample" not in batch:
            return batch

        from itertools import accumulate

        pixel_values = batch["pixel_values_videos"]
        video_grid_thw = batch["video_grid_thw"]
        num_videos_per_sample = batch["num_videos_per_sample"]

        # 如果已经是 list 格式，直接返回
        if isinstance(pixel_values, list):
            return batch

        # 计算每个视频的 features 数量
        lengths = video_grid_thw.prod(-1).tolist()  # [num_videos]

        if sum(lengths) != pixel_values.size(0):
            print(f"⚠️ [SPLIT_VIDEO] Mismatch: sum(lengths)={sum(lengths)} != pixel_values.size(0)={pixel_values.size(0)}")
            return batch

        # 计算每个 sample 的边界
        boundaries = [0, *accumulate(num_videos_per_sample)]
        sections = [sum(lengths[boundaries[i]:boundaries[i + 1]]) for i in range(len(num_videos_per_sample))]

        # 切分 pixel_values 和 video_grid_thw
        split_pixel_values = list(torch.split(pixel_values, sections, dim=0))
        split_video_grid_thw = list(torch.split(video_grid_thw, num_videos_per_sample, dim=0))

        print(f"🔧 [SPLIT_VIDEO] 切分视频特征: {len(split_pixel_values)} samples")
        for i, (pv, gt) in enumerate(zip(split_pixel_values, split_video_grid_thw)):
            print(f"   Sample {i}: pv={pv.shape}, gt={gt.shape}")

        return {
            **batch,
            "pixel_values_videos": split_pixel_values,
            "video_grid_thw": split_video_grid_thw,
        }

    def _unsplit_pixel_values_videos_by_grid(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        类似 TRL 的 unsplit_pixel_values_by_grid，但用于视频特征。

        将 list[Tensor] 格式的视频特征合并回 concatenated tensor 格式。
        """
        pixel_values = batch.get("pixel_values_videos")
        if isinstance(pixel_values, list) and len(pixel_values) > 0:
            # 过滤掉 None
            valid_pv = [pv for pv in pixel_values if pv is not None]
            if valid_pv:
                merged = torch.cat(valid_pv, dim=0)
                batch = {**batch, "pixel_values_videos": merged}
            print(f"🔧 [UNSPLIT_VIDEO] 合并 pixel_values_videos: {len(pixel_values)} -> {merged.shape if valid_pv else 'empty'}")

        video_grid_thw = batch.get("video_grid_thw")
        if isinstance(video_grid_thw, list) and len(video_grid_thw) > 0:
            valid_gt = [gt for gt in video_grid_thw if gt is not None]
            if valid_gt:
                merged = torch.cat(valid_gt, dim=0)
                batch = {**batch, "video_grid_thw": merged}
            print(f"🔧 [UNSPLIT_VIDEO] 合并 video_grid_thw: {len(video_grid_thw)} -> {merged.shape if valid_gt else 'empty'}")

        return batch

    def _prepare_inputs(self, generation_batch):
        """
        重写 _prepare_inputs，在调用父类方法后处理视频特征。

        TRL 的 _prepare_inputs 流程：
        1. _generate_and_score_completions
        2. split_pixel_values_by_grid (只处理图片)
        3. shuffle_sequence_dict
        4. split_tensor_dict
        5. unsplit_pixel_values_by_grid

        我们在步骤 2 之后添加视频特征的 split，
        在步骤 5 之后添加视频特征的 unsplit。
        """
        from trl.trainer.utils import split_tensor_dict, shuffle_sequence_dict
        from trl.trainer.utils import split_pixel_values_by_grid, unsplit_pixel_values_by_grid

        mode = "train" if self.model.training else "eval"

        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # 调用我们重写的 _generate_and_score_completions
                generation_batch = self._generate_and_score_completions(generation_batch)

                # 🎯 TRL 的图片特征 split（保持兼容性）
                generation_batch = split_pixel_values_by_grid(generation_batch)

                # 🎯 对视频特征进行 split（类似 TRL 对图片的处理）
                generation_batch = self._split_pixel_values_videos_by_grid(generation_batch)

                # TRL 的标准流程：shuffle 和 split
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)

                # 🎯 对每个 mini-batch 进行 unsplit（先图片，再视频）
                self._buffered_inputs = []
                for batch in generation_batches:
                    batch = unsplit_pixel_values_by_grid(batch)
                    batch = self._unsplit_pixel_values_videos_by_grid(batch)
                    self._buffered_inputs.append(batch)

            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
        else:
            inputs = self._generate_and_score_completions(generation_batch)

        return inputs

    def _compute_loss(self, model, inputs):
        """
        🎯 重写 TRL 的 _compute_loss 方法以支持视频特征

        关键改动：
        - 检测 inputs 中是否有 pixel_values_videos 和 video_grid_thw
        - 如果有，使用 _get_per_token_logps_and_entropies_with_video 代替标准方法
        """
        from trl.trainer.grpo_trainer import nanmin, nanmax

        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        # 🎯 关键修复：使用 inputs 中的 attention_mask（视频 tokens = 1），而不是用 completion_mask 拼接
        # completion_mask 是 loss mask（视频 tokens = 0），不能用于 model forward
        attention_mask = inputs["attention_mask"]
        logits_to_keep = completion_ids.size(1)
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        # 🎯 检测是否有视频特征
        pixel_values_videos = inputs.get("pixel_values_videos")
        video_grid_thw = inputs.get("video_grid_thw")
        features_per_sample = inputs.get("features_per_sample")
        num_videos_per_sample = inputs.get("num_videos_per_sample")

        if pixel_values_videos is not None and video_grid_thw is not None:
            # 🎥 使用视频感知的 logps 计算方法
            print(f"🎥 [COMPUTE_LOSS] 使用视频感知的 logps 计算")

            device = input_ids.device
            batch_size = input_ids.size(0)
            VIDEO_PAD_TOKEN_ID = 151656
            merge_size = 2

            # 🔍 断点：验证 batch_size 和 features_per_sample 长度
            print(f"🔍 [BATCH_SIZE_CHECK] batch_size={batch_size}, "
                  f"features_per_sample_len={len(features_per_sample) if features_per_sample else 'None'}, "
                  f"num_videos_per_sample_len={len(num_videos_per_sample) if num_videos_per_sample else 'None'}")

            # 🎯 确保视频特征在正确的设备上
            if isinstance(pixel_values_videos, torch.Tensor):
                pixel_values_videos = pixel_values_videos.to(device)
            if isinstance(video_grid_thw, torch.Tensor):
                video_grid_thw = video_grid_thw.to(device)

            # 🎯 关键修复：使用 features_per_sample 元数据将 concatenated tensor 切分为 list
            # 这样每个 sample 可以正确获取自己的视频特征
            if features_per_sample is not None and isinstance(pixel_values_videos, torch.Tensor):
                print(f"   🔧 使用 features_per_sample 切分 concatenated tensor")
                print(f"   📊 pixel_values_videos shape: {pixel_values_videos.shape}")
                print(f"   📊 video_grid_thw shape: {video_grid_thw.shape}")
                print(f"   📊 features_per_sample: {features_per_sample}")
                print(f"   📊 num_videos_per_sample: {num_videos_per_sample}")
                print(f"   📊 batch_size: {batch_size}")

                # 🔍 验证：检查 batch_size 与 metadata 长度是否匹配
                if len(features_per_sample) != batch_size:
                    print(f"   ⚠️ WARNING: features_per_sample 长度 ({len(features_per_sample)}) != batch_size ({batch_size})")
                    print(f"   ⚠️ 这通常意味着 TRL 的数据处理已经切分了 batch，但 metadata 没有被正确切分")
                    # 尝试修复：如果 metadata 包含了所有 samples，我们需要推断当前 batch 的 offset
                    # 但这很难准确做到，所以这里先打印警告

                # 切分 pixel_values_videos
                split_pixel_values = []
                split_video_grid_thw = []
                num_videos = []

                pv_start_idx = 0
                gt_start_idx = 0

                for i in range(min(batch_size, len(features_per_sample))):
                    num_features = features_per_sample[i]
                    nv = num_videos_per_sample[i] if num_videos_per_sample else 1

                    # 切分 pixel_values
                    pv_end_idx = pv_start_idx + num_features
                    if pv_end_idx <= pixel_values_videos.shape[0]:
                        split_pixel_values.append(pixel_values_videos[pv_start_idx:pv_end_idx])
                    else:
                        print(f"   ⚠️ Sample {i}: pv_end_idx ({pv_end_idx}) > total ({pixel_values_videos.shape[0]})")
                        split_pixel_values.append(None)
                    pv_start_idx = pv_end_idx

                    # 切分 video_grid_thw
                    gt_end_idx = gt_start_idx + nv
                    if gt_end_idx <= video_grid_thw.shape[0]:
                        split_video_grid_thw.append(video_grid_thw[gt_start_idx:gt_end_idx])
                    else:
                        print(f"   ⚠️ Sample {i}: gt_end_idx ({gt_end_idx}) > total ({video_grid_thw.shape[0]})")
                        split_video_grid_thw.append(None)
                    gt_start_idx = gt_end_idx

                    num_videos.append(nv)

                # 🔍 验证切分结果
                print(f"   🔧 切分完成:")
                for i, (pv, gt) in enumerate(zip(split_pixel_values, split_video_grid_thw)):
                    # 计算 input_ids 中的 video_pad tokens 数量
                    sample_video_tokens = (input_ids[i] == VIDEO_PAD_TOKEN_ID).sum().item()
                    expected_features = sample_video_tokens * (merge_size ** 2)
                    actual_features = pv.shape[0] if pv is not None else 0
                    grid_features = gt.prod(dim=-1).sum().item() if gt is not None else 0

                    match = "✅" if (actual_features == expected_features == grid_features) else "❌"
                    print(f"   {match} Sample {i}: video_tokens={sample_video_tokens}, expected={expected_features}, pv={actual_features}, gt={grid_features}")

                # 使用切分后的 list 格式
                pixel_values_videos = split_pixel_values
                video_grid_thw = split_video_grid_thw

            else:
                # 没有 metadata，尝试推断 num_videos
                print(f"   📊 无 features_per_sample metadata，使用推断")
                if isinstance(video_grid_thw, torch.Tensor):
                    total_videos = video_grid_thw.shape[0]
                    videos_per_sample = total_videos // batch_size
                    num_videos = [videos_per_sample] * batch_size
                    print(f"   📊 pixel_values_videos: tensor shape {pixel_values_videos.shape}")
                    print(f"   📊 video_grid_thw: tensor shape {video_grid_thw.shape}")
                    print(f"   📊 推断 num_videos: {num_videos}")
                else:
                    num_videos = [1] * batch_size

            per_token_logps, entropies = self._get_per_token_logps_and_entropies_with_video(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                compute_entropy=True,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                num_videos=num_videos,
            )
        else:
            # 🖼️ 使用标准方法（可能有图片）
            per_token_logps, entropies = self._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                compute_entropy=True,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                num_images=inputs.get("num_images"),
                pixel_attention_mask=inputs.get("pixel_attention_mask"),
                image_sizes=inputs.get("image_sizes"),
                token_type_ids=inputs.get("token_type_ids"),
            )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the loss
        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        if self.off_policy_mask_threshold is not None:
            sampling_per_token_logps = inputs.get("sampling_per_token_logps", old_per_token_logps)
            off_policy_mask = self.get_off_policy_mask(
                advantages=advantages,
                per_token_logps=per_token_logps,
                sampling_per_token_logps=sampling_per_token_logps,
                mask=mask,
                off_policy_threshold=self.off_policy_mask_threshold,
            )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown importance sampling level: {self.importance_sampling_level}")

        coef_1 = torch.exp(log_importance_weights)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs.get("ref_per_token_logps")

            # 🎯 关键修改：如果 ref_per_token_logps 不存在，在这里计算
            # 这样可以复用 _compute_loss 中已有的视频特征处理逻辑
            if ref_per_token_logps is None:
                print(f"🔍 [COMPUTE_LOSS] ref_per_token_logps 不存在，在 _compute_loss 中计算")
                with torch.no_grad():
                    if self.ref_model is not None:
                        if pixel_values_videos is not None and video_grid_thw is not None:
                            ref_per_token_logps, _ = self._get_per_token_logps_and_entropies_with_video(
                                self.ref_model,
                                input_ids,
                                attention_mask,
                                logits_to_keep,
                                compute_entropy=False,
                                pixel_values_videos=pixel_values_videos,
                                video_grid_thw=video_grid_thw,
                                num_videos=num_videos,
                            )
                        else:
                            ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                                self.ref_model,
                                input_ids,
                                attention_mask,
                                logits_to_keep,
                                compute_entropy=False,
                                pixel_values=inputs.get("pixel_values"),
                                image_grid_thw=inputs.get("image_grid_thw"),
                                num_images=inputs.get("num_images"),
                            )
                    else:
                        # Use model with disabled adapter as reference
                        with self.accelerator.unwrap_model(model).disable_adapter():
                            if pixel_values_videos is not None and video_grid_thw is not None:
                                ref_per_token_logps, _ = self._get_per_token_logps_and_entropies_with_video(
                                    model,
                                    input_ids,
                                    attention_mask,
                                    logits_to_keep,
                                    compute_entropy=False,
                                    pixel_values_videos=pixel_values_videos,
                                    video_grid_thw=video_grid_thw,
                                    num_videos=num_videos,
                                )
                            else:
                                ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                                    model,
                                    input_ids,
                                    attention_mask,
                                    logits_to_keep,
                                    compute_entropy=False,
                                    pixel_values=inputs.get("pixel_values"),
                                    image_grid_thw=inputs.get("image_grid_thw"),
                                    num_images=inputs.get("num_images"),
                                )
                print(f"✅ [COMPUTE_LOSS] ref_per_token_logps 计算完成, shape={ref_per_token_logps.shape}")

            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)
            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "sapo":
            temperatures = torch.where(advantages > 0, self.args.sapo_temperature_pos, self.args.sapo_temperature_neg)
            soft_coef_1 = torch.sigmoid(temperatures * (coef_1 - 1)) * 4 / temperatures
            per_token_loss = -soft_coef_1 * advantages
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.off_policy_mask_threshold is not None:
            per_token_loss = per_token_loss * off_policy_mask

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        mode = "train" if self.model.training else "eval"
        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.loss_type in ["cispo", "dapo"]:
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * mask).sum() / normalizer
        elif self.loss_type == "luspo":
            loss = (per_token_loss * mask.sum(1, keepdim=True)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            else:
                return (x * mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather(cispo_clip_ratio)
            self._metrics[mode]["cispo_clip_ratio"].append(gathered_cispo_clip_ratio.nanmean().item())

        return loss

    def _get_per_token_logps_and_entropies_with_video(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values_videos=None,
        video_grid_thw=None,
        num_videos=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute log-probs and (optionally) entropies for each token, with video support.

        This extends TRL's _get_per_token_logps_and_entropies to handle video inputs
        (pixel_values_videos, video_grid_thw) in addition to image inputs.

        Args:
            model: The model to compute logps with
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            logits_to_keep: Number of completion tokens
            batch_size: Batch size for chunking (to reduce memory)
            compute_entropy: Whether to compute entropy
            pixel_values_videos: Video pixel values (list of tensors or concatenated tensor)
            video_grid_thw: Video grid dimensions [num_videos, 3] or list
            num_videos: Number of videos per sample (list of ints)

        Returns:
            Tuple of (per_token_logps, entropies or None)
        """
        from trl.trainer.grpo_trainer import selective_log_softmax, entropy_from_logits

        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []

        # Handle video features as list (one tensor per sample)
        if isinstance(pixel_values_videos, list):
            video_features_list = pixel_values_videos
            video_grid_list = video_grid_thw if isinstance(video_grid_thw, list) else [video_grid_thw[i:i+1] for i in range(len(pixel_values_videos))]
        else:
            video_features_list = None
            video_grid_list = None

        for start in range(0, input_ids.size(0), batch_size):
            end = min(start + batch_size, input_ids.size(0))
            input_ids_batch = input_ids[start:end]
            attention_mask_batch = attention_mask[start:end]

            # Build model inputs
            model_inputs = {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
            }

            # Add video features for this batch
            if video_features_list is not None:
                # Concatenate video features for this batch
                batch_video_features = video_features_list[start:end]
                if batch_video_features and batch_video_features[0] is not None:
                    model_inputs["pixel_values_videos"] = torch.cat(batch_video_features, dim=0)

                    # Handle video_grid_thw
                    batch_video_grids = video_grid_list[start:end]
                    if batch_video_grids and batch_video_grids[0] is not None:
                        if isinstance(batch_video_grids[0], torch.Tensor):
                            model_inputs["video_grid_thw"] = torch.cat(batch_video_grids, dim=0)
                        else:
                            model_inputs["video_grid_thw"] = torch.tensor(batch_video_grids)

            elif pixel_values_videos is not None and video_grid_thw is not None:
                # Handle concatenated tensor format (similar to TRL's image handling)
                if num_videos is not None:
                    # video_grid_thw = [T, H, W]，每个视频的 patches 数 = T * H * W
                    # pixel_values_videos 是展开的，每个 patch 一行
                    rows_per_video = video_grid_thw.prod(dim=-1)  # T * H * W

                    # 🎯 关键：计算每个 sample 的总特征行数
                    # num_videos[i] 表示 sample i 有多少个视频
                    # 我们需要把 rows_per_video 按 num_videos 分组，然后对每组求和
                    rows_per_sample_list = []
                    vid_idx = 0
                    for nv in num_videos:
                        sample_rows = rows_per_video[vid_idx:vid_idx + nv].sum()
                        rows_per_sample_list.append(sample_rows)
                        vid_idx += nv
                    rows_per_sample = torch.stack(rows_per_sample_list)

                    # 计算累积行索引
                    cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                    row_start, row_end = cum_rows[start].item(), cum_rows[end].item()
                    model_inputs["pixel_values_videos"] = pixel_values_videos[row_start:row_end]

                    # 计算 video_grid_thw 的切片索引
                    cum_vids = torch.tensor([0] + list(num_videos)).cumsum(0)
                    vid_start, vid_end = cum_vids[start].item(), cum_vids[end].item()
                    model_inputs["video_grid_thw"] = video_grid_thw[vid_start:vid_end]

                    # 🔍 验证 video_pad tokens 与 pixel_values 匹配
                    # 如果不匹配，则不传递视频特征（避免 get_rope_index 错误）
                    video_pad_token_id = 151656
                    batch_video_tokens = (input_ids_batch == video_pad_token_id).sum().item()
                    batch_expected_tokens = model_inputs["video_grid_thw"].prod(dim=-1).sum().item() // 4
                    print(f"🔍 [LOGPS DEBUG] batch [{start}:{end}] - video_pad tokens in ids: {batch_video_tokens}, expected from grid: {batch_expected_tokens}")

                    if batch_video_tokens != batch_expected_tokens:
                        print(f"⚠️ [LOGPS DEBUG] MISMATCH! Removing video features to avoid get_rope_index error")
                        print(f"⚠️ [LOGPS DEBUG] video_grid_thw: {model_inputs['video_grid_thw']}")
                        print(f"⚠️ [LOGPS DEBUG] num_videos: {num_videos}")
                        # 清除视频特征，让模型不处理视频
                        model_inputs.pop("pixel_values_videos", None)
                        model_inputs.pop("video_grid_thw", None)
                else:
                    # Simple slicing if num_videos not provided
                    model_inputs["pixel_values_videos"] = pixel_values_videos
                    model_inputs["video_grid_thw"] = video_grid_thw

            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False

            logits = model(**model_inputs).logits
            logits = logits[:, :-1, :]  # (B, L-1, H)
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            logits = logits / self.temperature
            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None

        return logps, entropies

    def _format_prompt_with_chat_template(self, prompt: str) -> List[int]:
        """
        Apply chat template to format prompt correctly for generation.

        This is crucial for the model to know where to start generating.
        Without the chat template, the model doesn't see the <|im_start|>assistant\n
        prefix and may generate unexpected tokens (like a leading space).

        Args:
            prompt: Raw prompt text (e.g., "<video>\nSelect the best answer...")

        Returns:
            List of token IDs with proper chat template applied
        """
        if not hasattr(self, 'template') or self.template is None:
            # Fallback: direct tokenization (may cause leading space issue)
            logger.warning("Template not available, using direct tokenization")
            return self.processing_class(prompt, add_special_tokens=True)['input_ids']

        # Build messages in the expected format
        messages = [
            {"role": Role.USER.value, "content": prompt},
            {"role": Role.ASSISTANT.value, "content": ""}  # Empty assistant for generation
        ]

        # Use template's encode_oneturn to get properly formatted input_ids
        # This adds the <|im_start|>assistant\n prefix automatically
        input_ids, _ = self.template.encode_oneturn(
            self.processing_class,
            messages=messages,
            system=None,
            tools=None
        )

        return input_ids

    def _set_signature_columns_if_needed(self):
        """Override to include video column in signature columns."""
        if self._signature_columns is None:
            # Include video and answer columns in addition to the default ones
            self._signature_columns = ["prompt", "image", "images", "video", "answer", "video_configs"]

    def _preprocess_videos(
        self,
        prompts: List[str],
        videos: List[List[str]],
    ) -> Dict[str, Any]:
        """
        预处理视频，返回视频特征。

        这个方法只处理唯一的视频，用于后续的特征复用。

        Args:
            prompts: 原始 prompts（不是扩展后的）
            videos: 视频路径列表

        Returns:
            包含 pixel_values_videos, video_grid_thw 等特征的字典
        """
        # Replace <video>\n with vision tokens in prompts
        video_placeholder = "<|vision_start|><|video_pad|><|vision_end|>\n"
        formatted_prompts = [prompt.replace("<video>\n", video_placeholder) for prompt in prompts]

        logger.info(f"[VIDEO PREPROCESS] Processing {len(videos)} unique videos")

        if hasattr(self, 'template') and self.template and hasattr(self.template, 'plugin') and self.template.plugin:
            # Process each prompt with its video using mm_plugin
            all_processed_inputs = []
            for idx, (prompt, video_list) in enumerate(zip(formatted_prompts, videos)):
                # 🔧 FIX: Use chat template to format prompt correctly
                # This ensures the model sees <|im_start|>assistant\n before generating
                # Without this, the model generates a leading space before <think>
                input_ids = self._format_prompt_with_chat_template(prompt)

                # Process with mm_plugin (like inference does)
                processed_ids, mm_features = self.template.plugin.process_single_mm_input(
                    input_ids=input_ids,
                    videos=[video_list[0]] if video_list else None,
                    failover=True
                )

                if processed_ids is None:
                    logger.error(f"Failed to process video for prompt {idx}")
                    continue

                all_processed_inputs.append({
                    'input_ids': processed_ids['input_ids'],
                    'attention_mask': processed_ids.get('attention_mask', [1] * len(processed_ids['input_ids'])),
                    **mm_features
                })

            # Separate text features from video features
            text_features = []
            video_features = {}

            for item in all_processed_inputs:
                text_item = {
                    'input_ids': item['input_ids'],
                    'attention_mask': item.get('attention_mask', [1] * len(item['input_ids']))
                }
                text_features.append(text_item)

                # Collect video features separately
                for k, v in item.items():
                    if k not in ['input_ids', 'attention_mask']:
                        if k not in video_features:
                            video_features[k] = []
                        video_features[k].append(v)

            # Pad text features
            model_inputs = self.processing_class.pad(
                text_features,
                padding=True,
                return_tensors="pt"
            )

            # Process video features
            processed_video_features = {}
            for k, v_list in video_features.items():
                if len(v_list) > 0:
                    if k == 'pixel_values_videos':
                        processed_video_features[k] = v_list  # Keep as list for per-sample access
                    elif k == 'video_grid_thw':
                        processed_video_features[k] = v_list  # Keep as list for per-sample access
                    else:
                        processed_video_features[k] = v_list

            # Return both text and video features
            return {
                'text_features': text_features,
                'model_inputs': model_inputs,
                'video_features': processed_video_features,
                'formatted_prompts': formatted_prompts,
            }
        else:
            raise RuntimeError("Template or mm_plugin not available for video preprocessing")

    def _generate(
        self,
        prompts: List[str],
        videos: Optional[List[List[str]]] = None,
        preprocessed_video_features: Optional[Dict[str, Any]] = None,
    ) -> Tuple[
        List[List[int]],  # prompt_ids_list
        List[List[int]],  # completion_ids_list
        List[List[int]],  # tool_mask_list
        List[str],        # completions
        int,              # num_items_in_batch
        Optional[List[List[float]]],  # sampling_per_token_logps_list
        Dict[str, Any],   # extra_fields
    ]:
        """
        Generate completions for prompts with video support for Qwen3VL VMCOT.

        This method extends the parent's _generate to handle video inputs.

        Args:
            prompts: List of prompts to generate completions for
            videos: Optional list of video paths (used for dynamic video insertion)
            preprocessed_video_features: Optional pre-computed video features (for optimization)
        """
        device = self.accelerator.device

        print(f"[DEBUG] _generate called with {len(prompts)} prompts")
        if preprocessed_video_features is not None:
            print(f"🎥 [DEBUG] Using preprocessed video features (OPTIMIZED - no re-encoding)")
        elif videos:
            print(f"[DEBUG] Videos provided: {len(videos)} videos (will process each)")
            for i, v in enumerate(videos[:3]):
                print(f"[DEBUG] Video {i}: {v}")
            if len(videos) > 3:
                print(f"[DEBUG] ... and {len(videos) - 3} more videos")

        # For Qwen3VL VMCOT, replace <video>\n with vision tokens in prompts
        formatted_prompts = []
        if videos is not None:
            video_placeholder = "<|vision_start|><|video_pad|><|vision_end|>\n"
            for prompt in prompts:
                formatted_prompts.append(prompt.replace("<video>\n", video_placeholder))
        else:
            formatted_prompts = prompts

        print(f"[DEBUG] Formatted prompts: {formatted_prompts[:2]}...")

        # 🎯 为动态视频插入设置上下文（始终需要，即使使用预处理特征）
        if videos is not None and hasattr(self.model, 'set_video_context'):
            video_paths = [video_list[0] if video_list else None for video_list in videos]
            print(f"🎥🎥🎥 [DEBUG] 设置视频上下文，视频数量: {len(video_paths)} 🎥🎥🎥", file=sys.stderr)
            self.model.set_video_context(
                video_paths=video_paths,
                processor=self.processor,
                processor_args={}
            )

        # ⭐ 优化路径：使用预处理的视频特征
        if preprocessed_video_features is not None:
            print(f"🎥 [VIDEO OPTIMIZATION] Using preprocessed features - skipping video encoding!")

            text_features = preprocessed_video_features['text_features']
            video_features = preprocessed_video_features['video_features']

            # Pad text features
            model_inputs = self.processing_class.pad(
                text_features,
                padding=True,
                return_tensors="pt"
            )

            # Add video features
            for k, v_list in video_features.items():
                if len(v_list) > 0:
                    if k == 'pixel_values_videos':
                        model_inputs[k] = torch.cat(v_list, dim=0) if isinstance(v_list[0], torch.Tensor) else torch.concat(v_list)
                    elif k == 'video_grid_thw':
                        if isinstance(v_list[0], torch.Tensor):
                            stacked = torch.stack(v_list)
                            if stacked.dim() == 3 and stacked.shape[1] == 1:
                                model_inputs[k] = stacked.squeeze(1)
                            else:
                                model_inputs[k] = stacked
                        else:
                            model_inputs[k] = torch.tensor(v_list)
                    else:
                        model_inputs[k] = v_list[0] if len(v_list) == 1 else v_list

            print(f"[DEBUG] Preprocessed video features loaded: {list(model_inputs.keys())}")

        # 原始路径：需要处理视频
        elif videos is not None and self.processor is not None:
            logger.info(f"Processing {len(videos)} video inputs for generation")

            if hasattr(self, 'template') and self.template and hasattr(self.template, 'plugin') and self.template.plugin:
                logger.info("Using template's mm_plugin for video processing (same as inference)")

                all_processed_inputs = []
                for idx, (prompt, video_list) in enumerate(zip(formatted_prompts, videos)):
                    # 🔧 FIX: Use chat template to format prompt correctly
                    # This ensures the model sees <|im_start|>assistant\n before generating
                    # Without this, the model generates a leading space before <think>
                    input_ids = self._format_prompt_with_chat_template(prompt)

                    processed_ids, mm_features = self.template.plugin.process_single_mm_input(
                        input_ids=input_ids,
                        videos=[video_list[0]] if video_list else None,
                        failover=True
                    )

                    if processed_ids is None:
                        logger.error(f"Failed to process video for prompt {idx}")
                        continue

                    all_processed_inputs.append({
                        'input_ids': processed_ids['input_ids'],
                        'attention_mask': processed_ids.get('attention_mask', [1] * len(processed_ids['input_ids'])),
                        **mm_features
                    })

                text_features = []
                video_features = {}

                for item in all_processed_inputs:
                    text_item = {
                        'input_ids': item['input_ids'],
                        'attention_mask': item.get('attention_mask', [1] * len(item['input_ids']))
                    }
                    text_features.append(text_item)

                    for k, v in item.items():
                        if k not in ['input_ids', 'attention_mask']:
                            if k not in video_features:
                                video_features[k] = []
                            video_features[k].append(v)

                model_inputs = self.processing_class.pad(
                    text_features,
                    padding=True,
                    return_tensors="pt"
                )

                for k, v_list in video_features.items():
                    if len(v_list) > 0:
                        if k == 'pixel_values_videos':
                            model_inputs[k] = torch.cat(v_list, dim=0) if isinstance(v_list[0], torch.Tensor) else torch.concat(v_list)
                        elif k == 'video_grid_thw':
                            if isinstance(v_list[0], torch.Tensor):
                                stacked = torch.stack(v_list)
                                if stacked.dim() == 3 and stacked.shape[1] == 1:
                                    model_inputs[k] = stacked.squeeze(1)
                                else:
                                    model_inputs[k] = stacked
                            else:
                                model_inputs[k] = torch.tensor(v_list)
                        else:
                            model_inputs[k] = v_list[0] if len(v_list) == 1 else v_list
            else:
                logger.warning("Template or mm_plugin not available, falling back to direct processor")
                model_inputs = self.processor(
                    text=formatted_prompts,
                    videos=videos,
                    return_tensors="pt",
                    padding=True
                )

            print(f"[DEBUG] Processor output keys: {list(model_inputs.keys())}")
            for k, v in model_inputs.items():
                if hasattr(v, 'shape'):
                    print(f"[DEBUG] {k}: shape={v.shape}")
        else:
            # No videos, use chat template to format prompts correctly
            # 🔧 FIX: Apply chat template to ensure model sees <|im_start|>assistant\n
            all_input_ids = []
            for prompt in formatted_prompts:
                input_ids = self._format_prompt_with_chat_template(prompt)
                all_input_ids.append({'input_ids': input_ids, 'attention_mask': [1] * len(input_ids)})

            model_inputs = self.processing_class.pad(
                all_input_ids,
                padding=True,
                return_tensors="pt"
            )

        # Move to device
        model_inputs = {k: v.to(device) if hasattr(v, 'to') else v
                       for k, v in model_inputs.items()}

        # Generate completions
        print(f"[DEBUG] Starting generation with model_inputs keys: {list(model_inputs.keys())}")

        # For Qwen3VL with dynamic video, we need to handle generation carefully
        # Generate one by one to avoid batch processing issues
        all_sequences = []
        batch_size = model_inputs["input_ids"].shape[0]

        # 🎯 初始化 per-sample 视频特征存储
        # 在生成过程中，每个 sample 的完整视频特征（初始 + 动态）会被保存
        # 这样在 logps 计算时可以直接复用，避免重新合并导致的不匹配问题
        self._per_sample_video_features = []

        # 🔧 Fix for gradient_checkpointing compatibility with Qwen3VL
        # When gradient_checkpointing is enabled, we need to disable it during generation
        # to avoid attention_mask/input_ids shape mismatch in get_rope_index
        # TRL's _unwrap_model_for_generation does this automatically, but our custom _generate
        # bypasses that, so we need to handle it manually here
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        is_gradient_checkpointing = getattr(unwrapped_model, 'is_gradient_checkpointing', False)
        if is_gradient_checkpointing:
            print(f"[DEBUG] Disabling gradient_checkpointing for generation")
            unwrapped_model.gradient_checkpointing_disable()

        # In GRPO, num_generations samples share the same video
        # Get num_generations from config
        num_generations = getattr(self.generation_config, 'num_generations',
                                 getattr(self.args, 'num_generations', 1))

        print(f"[DEBUG] batch_size={batch_size}, num_generations={num_generations}")

        # Calculate features offset for each video
        video_grid_thw = model_inputs.get("video_grid_thw")
        if video_grid_thw is not None:
            # Each sample has its own video features, calculate for all samples
            features_per_video = []
            for idx in range(batch_size):
                grid = video_grid_thw[idx]  # Should be [T, H, W] or just [3]
                print(f"[DEBUG] Sample {idx} grid shape: {grid.shape}")
                # Handle different grid shapes
                if grid.dim() == 1 and grid.shape[0] == 3:
                    # grid is [T, H, W] as expected
                    num_features = grid[0] * grid[1] * grid[2]
                elif grid.dim() == 2 and grid.shape[0] == 1 and grid.shape[1] == 3:
                    # grid is [1, 3], need to squeeze
                    grid = grid.squeeze(0)
                    num_features = grid[0] * grid[1] * grid[2]
                else:
                    raise ValueError(f"Unexpected grid shape: {grid.shape}")
                features_per_video.append(num_features.item())

            # Calculate start indices for each video's features
            feature_starts = [0]
            for i in range(len(features_per_video) - 1):
                feature_starts.append(feature_starts[-1] + features_per_video[i])

            print(f"[DEBUG] Features per video: {features_per_video}")
            print(f"[DEBUG] Feature starts: {feature_starts}")
            print(f"[DEBUG] Total features expected: {sum(features_per_video)}")
            print(f"[DEBUG] Total features actual: {model_inputs['pixel_values_videos'].shape[0]}")

        try:
            with torch.no_grad():
                for i in range(batch_size):
                    # Extract single sample inputs
                    single_inputs = {}
                    for k, v in model_inputs.items():
                        if hasattr(v, '__getitem__'):
                            if k == "pixel_values_videos" and video_grid_thw is not None:
                                # Each sample has its own video features
                                start_idx = feature_starts[i]
                                end_idx = start_idx + features_per_video[i]
                                single_inputs[k] = v[start_idx:end_idx]
                                print(f"[DEBUG] Sample {i}: features {start_idx}:{end_idx}")
                            elif k == "video_grid_thw":
                                single_inputs[k] = v[i:i+1]
                            else:
                                # Keep batch dimension for other tensors
                                single_inputs[k] = v[i:i+1]
                        else:
                            single_inputs[k] = v

                    print(f"[DEBUG] Generating for sample {i+1}/{batch_size}")
                    # Debug inputs
                    for k, v in single_inputs.items():
                        if hasattr(v, 'shape'):
                            print(f"[DEBUG] single_inputs[{k}]: shape={v.shape}")

                    # 🎯 Debug generation parameters
                    if self.temperature == 0.0:
                        print(f"[DEBUG] Generation params: GREEDY DECODING (do_sample=False), max_new_tokens={self.args.max_completion_length}, repetition_penalty={self.repetition_penalty}")
                    else:
                        print(f"[DEBUG] Generation params: SAMPLING (do_sample=True), max_new_tokens={self.args.max_completion_length}, temperature={self.temperature}, top_p={self.top_p}, top_k={self.top_k}, repetition_penalty={self.repetition_penalty}")

                    try:
                        # 🎯 Explicitly pass all generation parameters to override checkpoint's generation_config.json
                        # This ensures our training parameters (temperature, top_k, etc.) are actually used

                        # Special handling for greedy decoding (temperature=0)
                        # Transformers requires do_sample=False when using greedy decoding
                        if self.temperature == 0.0:
                            # Greedy decoding: do_sample=False, no temperature/top_p/top_k
                            generation_kwargs = {
                                'max_new_tokens': self.args.max_completion_length,
                                'min_new_tokens': 10,
                                'do_sample': False,  # Greedy decoding
                                'repetition_penalty': self.repetition_penalty,
                                'pad_token_id': self.processing_class.pad_token_id,
                                'eos_token_id': self.processing_class.eos_token_id,
                                'return_dict_in_generate': True,
                                'output_scores': True,
                            }
                        else:
                            # Sampling with temperature > 0
                            generation_kwargs = {
                                'max_new_tokens': self.args.max_completion_length,
                                'min_new_tokens': 10,
                                'do_sample': True,  # Enable sampling
                                'temperature': self.temperature,
                                'top_p': self.top_p,
                                'top_k': self.top_k if self.top_k is not None else -1,
                                'repetition_penalty': self.repetition_penalty,
                                'pad_token_id': self.processing_class.pad_token_id,
                                'eos_token_id': self.processing_class.eos_token_id,
                                'return_dict_in_generate': True,
                                'output_scores': True,
                            }

                        # Use unwrapped_model for generation to avoid DeepSpeed wrapper issues
                        single_output = unwrapped_model.generate(
                            **single_inputs,
                            **generation_kwargs,
                        )
                        # Debug: decode and print what was generated
                        input_length = single_inputs['input_ids'].shape[1]
                        generated_tokens = single_output.sequences[0, input_length:]
                        generated_text = self.processing_class.decode(generated_tokens, skip_special_tokens=True)
                        print(f"[DEBUG] Sample {i} input_length={input_length}, output_length={single_output.sequences.shape[1]}, generated {single_output.sequences.shape[1] - input_length} tokens")
                        print(f"[DEBUG] Sample {i} generated text: '{generated_text}'")
                        print(f"[DEBUG] Sample {i} generated token IDs: {generated_tokens.tolist()}")
                        all_sequences.append(single_output.sequences)

                        # 🎯 关键改进：保存每个 sample 生成时使用的完整视频特征
                        # 这样计算 logps 时可以直接使用，避免重新合并导致的不匹配问题
                        if not hasattr(self, '_per_sample_video_features'):
                            self._per_sample_video_features = []

                        # 获取动态视频特征（如果有的话）
                        dynamic_pv = None
                        dynamic_gt = None

                        # 🔍 先检查 completion 中是否有动态插入的 video tokens
                        VIDEO_PAD_TOKEN_ID = 151656
                        completion_video_tokens = (generated_tokens == VIDEO_PAD_TOKEN_ID).sum().item()
                        initial_video_tokens = (single_inputs['input_ids'] == VIDEO_PAD_TOKEN_ID).sum().item()
                        dynamic_video_tokens = completion_video_tokens  # completion 中的 video tokens 都是动态的
                        print(f"🔍 [VIDEO TOKENS] Sample {i}: initial={initial_video_tokens}, dynamic_in_completion={dynamic_video_tokens}")

                        if hasattr(unwrapped_model, 'get_last_dynamic_features'):
                            sample_dynamic_features = unwrapped_model.get_last_dynamic_features()
                            if sample_dynamic_features:
                                dpv_list = sample_dynamic_features.get('dynamic_pixel_values_videos', [])
                                dgt_list = sample_dynamic_features.get('dynamic_video_grid_thw', [])
                                if dpv_list:
                                    # 合并这个 sample 的所有动态视频特征
                                    dynamic_pv = torch.cat(dpv_list, dim=0) if len(dpv_list) > 1 else dpv_list[0]
                                    dynamic_gt = torch.cat(dgt_list, dim=0) if len(dgt_list) > 1 else dgt_list[0]
                                    print(f"🎥 [DYNAMIC] Sample {i}: 捕获动态视频特征 shape={dynamic_pv.shape}")

                                    # 🔍 验证动态特征与 completion 中的 video tokens 是否匹配
                                    expected_patches = dynamic_video_tokens * 4  # merge_size=2, so 4 patches per token
                                    actual_patches = dynamic_pv.shape[0]
                                    if actual_patches != expected_patches:
                                        print(f"⚠️ [MISMATCH] Sample {i}: dynamic_video_tokens={dynamic_video_tokens}, "
                                              f"expected_patches={expected_patches}, actual_patches={actual_patches}")
                                else:
                                    if dynamic_video_tokens > 0:
                                        print(f"⚠️ [MISSING] Sample {i}: completion 有 {dynamic_video_tokens} 个 video tokens，但 dpv_list 为空！")
                            else:
                                if dynamic_video_tokens > 0:
                                    print(f"⚠️ [MISSING] Sample {i}: completion 有 {dynamic_video_tokens} 个 video tokens，但 sample_dynamic_features 为空！")

                        # 获取这个 sample 的初始视频特征
                        initial_pv = single_inputs.get('pixel_values_videos')
                        initial_gt = single_inputs.get('video_grid_thw')

                        # 合并初始 + 动态视频特征（如果有动态的话）
                        # 🎯 同时记录是否需要清理 completion 中的动态 video tokens
                        need_clean_completion = False

                        if dynamic_pv is not None and initial_pv is not None:
                            combined_pv = torch.cat([initial_pv, dynamic_pv.to(initial_pv.device)], dim=0)
                            # video_grid_thw 需要处理维度
                            if initial_gt.dim() == 2:  # [1, 3]
                                initial_gt_for_cat = initial_gt
                            else:  # [3]
                                initial_gt_for_cat = initial_gt.unsqueeze(0)
                            if dynamic_gt.dim() == 1:  # [3]
                                dynamic_gt_for_cat = dynamic_gt.unsqueeze(0)
                            else:
                                dynamic_gt_for_cat = dynamic_gt
                            combined_gt = torch.cat([initial_gt_for_cat, dynamic_gt_for_cat.to(initial_gt.device)], dim=0)
                            num_videos_for_sample = combined_gt.shape[0]
                            print(f"🎥 [COMBINED] Sample {i}: 合并视频特征 pv={combined_pv.shape}, gt={combined_gt.shape}, num_videos={num_videos_for_sample}")
                        else:
                            combined_pv = initial_pv
                            combined_gt = initial_gt
                            num_videos_for_sample = 1 if initial_gt is not None else 0
                            if dynamic_pv is None and initial_pv is not None:
                                print(f"🎥 [INITIAL ONLY] Sample {i}: 只有初始视频特征")
                                # 🎯 关键：如果 completion 中有 video tokens 但没有对应特征，需要清理
                                if dynamic_video_tokens > 0:
                                    need_clean_completion = True
                                    print(f"⚠️ [CLEAN NEEDED] Sample {i}: 需要清理 completion 中的 {dynamic_video_tokens} 个孤立 video tokens")

                        # 🔍 验证 pixel_values 和 video_grid_thw 是否一致
                        if combined_pv is not None and combined_gt is not None:
                            pv_rows = combined_pv.shape[0]
                            # 处理 combined_gt 的维度
                            if combined_gt.dim() == 1:
                                gt_expected = combined_gt[0] * combined_gt[1] * combined_gt[2]
                            else:
                                gt_expected = combined_gt.prod(dim=-1).sum()
                            gt_expected = gt_expected.item()

                            if pv_rows != gt_expected:
                                print(f"❌ [SAVE ERROR] Sample {i}: pv_rows={pv_rows} != gt_expected={gt_expected}")
                                print(f"   combined_gt: {combined_gt}")
                                # 修复：如果不一致，只保存初始视频特征
                                combined_pv = initial_pv
                                combined_gt = initial_gt
                                num_videos_for_sample = 1
                                print(f"   🔧 已回退到初始视频特征")
                            else:
                                print(f"✅ [SAVE] Sample {i}: pv_rows={pv_rows} == gt_expected={gt_expected}")

                        # 保存这个 sample 的完整视频特征
                        # 🎯 不转到 CPU，保持在 GPU 上避免后续 forward 时的设备不匹配
                        self._per_sample_video_features.append({
                            'pixel_values_videos': combined_pv if combined_pv is not None else None,
                            'video_grid_thw': combined_gt if combined_gt is not None else None,
                            'num_videos': num_videos_for_sample,
                        })

                    except Exception as e:
                        print(f"[ERROR] Generation failed for sample {i}: {str(e)}")
                        print(f"[ERROR] Traceback: {traceback.format_exc()}")
                        # Create a dummy output to continue
                        dummy_seq = torch.full((1, 1), self.processing_class.eos_token_id, device=device)
                        all_sequences.append(dummy_seq)

                        # 🎯 即使生成失败，也要保存初始视频特征（保持列表长度一致）
                        initial_pv = single_inputs.get('pixel_values_videos')
                        initial_gt = single_inputs.get('video_grid_thw')
                        self._per_sample_video_features.append({
                            'pixel_values_videos': initial_pv if initial_pv is not None else None,
                            'video_grid_thw': initial_gt if initial_gt is not None else None,
                            'num_videos': 1 if initial_gt is not None else 0,
                        })
        finally:
            # 🔧 Re-enable gradient_checkpointing after generation if it was enabled before
            # Use finally to ensure it's always re-enabled even if an error occurred
            if is_gradient_checkpointing:
                print(f"[DEBUG] Re-enabling gradient_checkpointing after generation")
                unwrapped_model.gradient_checkpointing_enable()

        # ⭐ Extract prompt and completion IDs directly from all_sequences (no need to pad+cat first)
        # This avoids introducing pad tokens into completion_ids_list
        print(f"[DEBUG] Sequence lengths: {[seq.shape[1] for seq in all_sequences]}")
        print(f"[DEBUG] Generation completed for {len(all_sequences)} samples")

        prompt_ids_list = [ids.tolist() for ids in model_inputs["input_ids"]]

        # Extract generated token IDs (excluding prompt) directly from each sequence
        # NOTE: We keep video/vision tokens as they are part of the model's learned format
        # for fine-grained video understanding (dynamic video with timestamps)
        completion_ids_list = []
        VIDEO_TOKENS = {151652, 151653, 151654, 151655, 151656}  # For counting only
        for i, seq in enumerate(all_sequences):
            generated_ids = seq.squeeze(0)  # Remove batch dimension: [1, seq_len] -> [seq_len]
            prompt_length = len(prompt_ids_list[i])
            completion_ids = generated_ids[prompt_length:].tolist()

            # Count video tokens (for debugging, but keep them in the output)
            video_token_count = sum(1 for tid in completion_ids if tid in VIDEO_TOKENS)
            if video_token_count > 0:
                print(f"✅ [KEEP] Sample {i}: contains {video_token_count} video tokens (preserved for fine-grained understanding)")

            completion_ids_list.append(completion_ids)

        # Decode completions
        completions = self.processing_class.batch_decode(
            [torch.tensor(ids) for ids in completion_ids_list],
            skip_special_tokens=True
        )

        # 📝 Print decoded completions for visualization
        print("\n" + "="*80)
        print("📝 DECODED COMPLETIONS")
        print("="*80)

        # 🔍 检查 completions 的多样性（重要：同一 prompt 的多个 generations 应该不同）
        unique_completions = set(completions)
        print(f"🎲 [DIVERSITY CHECK] {len(completions)} completions, {len(unique_completions)} unique")
        if len(unique_completions) == 1 and len(completions) > 1:
            print("⚠️  WARNING: All completions are identical! Check if sampling is working correctly.")
            print(f"    temperature={self.temperature}, do_sample should be True if temp > 0")
        elif len(unique_completions) == len(completions):
            print("✅ All completions are unique (good diversity)")
        else:
            print(f"ℹ️  Some completions are duplicated ({len(completions) - len(unique_completions)} duplicates)")

        for i, completion in enumerate(completions):
            print(f"\n🔍 Sample {i}:")
            print("-" * 80)
            # Show length info - calculate actual tokens (excluding pad tokens)
            actual_tokens = sum(1 for tid in completion_ids_list[i] if tid != self.processing_class.pad_token_id)
            print(f"Length: {len(completion)} chars, {actual_tokens} tokens (total with pad: {len(completion_ids_list[i])})")
            # Check for tags
            has_think_open = '<think>' in completion
            has_think_close = '</think>' in completion
            has_answer_open = '<answer>' in completion
            has_answer_close = '</answer>' in completion
            print(f"Tags: <think>={has_think_open}, </think>={has_think_close}, <answer>={has_answer_open}, </answer>={has_answer_close}")
            # Show the actual text (truncated to 500 chars for readability)
            content_preview = completion[:500] + "..." if len(completion) > 500 else completion
            print(f"\nContent (first 500 chars):\n{content_preview}")
            print("-" * 80)
        print("="*80 + "\n")

        # For now, we don't handle tools or per-token logps
        tool_mask_list = None
        sampling_per_token_logps_list = None

        # Calculate number of items
        num_items_in_batch = len(prompts)

        # Extra fields - include video features for use in _generate_and_score_completions
        extra_fields = {}
        if videos is not None:
            # 🎯 关键改进：直接使用生成过程中保存的 per-sample 视频特征
            # 这些特征在生成时就已经验证过匹配，不需要重新合并
            if hasattr(self, '_per_sample_video_features') and self._per_sample_video_features:
                extra_fields["per_sample_video_features"] = self._per_sample_video_features
                print(f"🎥 [GENERATE] 保存 {len(self._per_sample_video_features)} 个 sample 的视频特征到 extra_fields")
                for i, feat in enumerate(self._per_sample_video_features):
                    pv_shape = feat['pixel_values_videos'].shape if feat['pixel_values_videos'] is not None else None
                    gt_shape = feat['video_grid_thw'].shape if feat['video_grid_thw'] is not None else None
                    print(f"   Sample {i}: pv={pv_shape}, gt={gt_shape}, num_videos={feat['num_videos']}")

                # 清理（避免影响下一个 batch）
                self._per_sample_video_features = []
            else:
                # Fallback: 使用初始视频特征（没有动态视频的情况）
                initial_pv = model_inputs.get("pixel_values_videos")
                initial_gt = model_inputs.get("video_grid_thw")
                extra_fields["pixel_values_videos"] = initial_pv
                extra_fields["video_grid_thw"] = initial_gt
                print(f"[DEBUG] Fallback: 使用初始视频特征")
                print(f"[DEBUG]   initial pixel_values shape: {initial_pv.shape if initial_pv is not None else None}")
                print(f"[DEBUG]   initial video_grid_thw: {initial_gt}")

        return (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions,
            num_items_in_batch,
            sampling_per_token_logps_list,
            extra_fields,
        )

    def _generate_and_score_completions(
        self, inputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate completions and score them, handling video inputs for Qwen3VL VMCOT.

        This method is called by TRL's _prepare_inputs method.

        ⚠️ 重要：TRL 的 RepeatSampler 已经在数据层面做了 prompt 的重复！
        传入的 inputs 已经包含了重复的 prompts（每个 prompt 重复 num_generations 次）。
        例如：inputs = [P0, P0, P0, P0, P1, P1, P1, P1] （num_generations=4）

        我们不需要再扩展 num_generations 倍，只需要：
        1. 识别 unique videos
        2. 每个 unique video 只编码一次
        3. 复制特征给同一 prompt 的多个 generations
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        print(f"[DEBUG] _generate_and_score_completions called with {len(inputs)} inputs")
        print(f"[DEBUG] First input keys: {list(inputs[0].keys()) if inputs else 'empty'}")

        # 🔍 打印每个 input 的 prompt 前 50 字符，确认是否是同一个 prompt
        print(f"[DEBUG] Input prompts (first 50 chars each):")
        for i, inp in enumerate(inputs):
            prompt_preview = inp.get('prompt', '')[:50].replace('\n', ' ')
            video_path = inp.get('video', 'N/A')
            if isinstance(video_path, list):
                video_path = video_path[0] if video_path else 'N/A'
            # 只显示视频文件名
            video_name = video_path.split('/')[-1] if isinstance(video_path, str) else 'N/A'
            print(f"  Input {i}: prompt='{prompt_preview}...' | video={video_name}")

        # 获取 num_generations（用于后续的 reward 计算）
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        print(f"[DEBUG] num_generations = {num_generations}, mode = {mode}")

        # ====================================================================
        # ⭐ 视频特征复用优化（修复版）
        # TRL 的 RepeatSampler 已经重复了 prompts，所以 inputs 中已经有重复
        # 我们需要：
        # 1. 识别 unique videos（通过视频路径去重）
        # 2. 每个 unique video 只编码一次
        # 3. 根据 input 的 video path 映射到对应的特征
        # ====================================================================

        # 提取所有 prompts（不扩展，直接使用 inputs）
        prompts = [x["prompt"] for x in inputs]

        # 提取所有视频路径
        all_videos = None
        if "video" in inputs[0]:
            all_videos = []
            for x in inputs:
                video = x.get("video")
                if isinstance(video, str):
                    all_videos.append([video])
                elif isinstance(video, list):
                    all_videos.append(video)
                else:
                    all_videos.append(None)

        # ⭐ 关键：识别 unique videos 并建立映射
        unique_videos = []  # 唯一的视频列表
        unique_prompts = []  # 对应的 prompts（用于预处理）
        video_to_unique_idx = {}  # 视频路径 -> unique index 的映射
        input_to_unique_idx = []  # 每个 input 对应的 unique video index

        if all_videos is not None:
            for i, video_list in enumerate(all_videos):
                if video_list is None:
                    input_to_unique_idx.append(None)
                    continue

                # 使用第一个视频路径作为 key（假设每个 input 只有一个视频）
                video_key = video_list[0] if video_list else None

                if video_key not in video_to_unique_idx:
                    # 新的 unique video
                    video_to_unique_idx[video_key] = len(unique_videos)
                    unique_videos.append(video_list)
                    unique_prompts.append(prompts[i])

                input_to_unique_idx.append(video_to_unique_idx[video_key])

            print(f"🎥 [VIDEO DEDUP] Found {len(unique_videos)} unique videos from {len(all_videos)} inputs")
            print(f"🎥 [VIDEO DEDUP] Video mapping: {input_to_unique_idx}")

        # ⭐ 只预处理 unique videos（每个 unique video 只编码一次！）
        preprocessed_video_features = None
        if unique_videos and self.processor is not None:
            print(f"🎥 [VIDEO OPTIMIZATION] Preprocessing {len(unique_videos)} unique videos (saved {len(all_videos) - len(unique_videos)} encodings)")
            preprocessed_video_features = self._preprocess_videos(unique_prompts, unique_videos)
            print(f"🎥 [VIDEO OPTIMIZATION] Video preprocessing complete")

        # ⭐ 根据映射，为每个 input 复制对应的视频特征
        expanded_video_features = None
        if preprocessed_video_features is not None and input_to_unique_idx:
            expanded_text_features = []
            expanded_pixel_values = []
            expanded_video_grid_thw = []
            expanded_formatted_prompts = []

            text_features = preprocessed_video_features['text_features']
            video_features = preprocessed_video_features['video_features']
            formatted_prompts_list = preprocessed_video_features['formatted_prompts']

            for unique_idx in input_to_unique_idx:
                if unique_idx is None:
                    continue
                # 复制 unique video 的特征
                expanded_text_features.append(text_features[unique_idx])
                expanded_formatted_prompts.append(formatted_prompts_list[unique_idx])
                if 'pixel_values_videos' in video_features:
                    expanded_pixel_values.append(video_features['pixel_values_videos'][unique_idx])
                if 'video_grid_thw' in video_features:
                    expanded_video_grid_thw.append(video_features['video_grid_thw'][unique_idx])

            expanded_video_features = {
                'text_features': expanded_text_features,
                'formatted_prompts': expanded_formatted_prompts,
                'video_features': {}
            }
            if expanded_pixel_values:
                expanded_video_features['video_features']['pixel_values_videos'] = expanded_pixel_values
            if expanded_video_grid_thw:
                expanded_video_features['video_features']['video_grid_thw'] = expanded_video_grid_thw

            print(f"🎥 [VIDEO OPTIMIZATION] Expanded features: {len(unique_videos)} unique -> {len(expanded_text_features)} total")

        # Extract ground truths for reward calculation if present
        # ⚠️ 注意：现在直接使用 inputs，不再使用 expanded_inputs
        ground_truths = None
        if "answer" in inputs[0]:
            ground_truths = [x.get("answer", "") for x in inputs]

        # Handle images (from parent implementation)
        images = None
        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]

        # For Qwen3VL VMCOT, if we have videos, handle them
        if all_videos is not None:
            # Generate with video inputs using our custom _generate
            # 使用预处理的视频特征（如果有的话）
            (
                prompt_ids_list,
                completion_ids_list,
                tool_mask_list,
                completions,
                num_items_in_batch,
                sampling_per_token_logps_list,
                extra_fields,
            ) = self._generate(prompts, videos=all_videos, preprocessed_video_features=expanded_video_features)
        else:
            # No videos, use parent's _generate
            (
                prompt_ids_list,
                completion_ids_list,
                tool_mask_list,
                completions,
                num_items_in_batch,
                sampling_per_token_logps_list,
                extra_fields,
            ) = self._generate(prompts)

        # Convert to tensors and pad
        # IMPORTANT: Use dtype=torch.long for token IDs (not float)
        prompt_ids = [torch.tensor(ids, dtype=torch.long, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")

        completion_ids = [torch.tensor(ids, dtype=torch.long, device=device) for ids in completion_ids_list]

        # ⭐ 创建两种 mask:
        # 1. completion_attention_mask: 用于 model forward，只 mask pad tokens
        # 2. completion_loss_mask: 用于 loss 计算，只 mask video_pad 和 pad tokens
        #
        # 🎯 GRPO 中动态插入的视频 token 结构:
        #   <X.X seconds><|vision_start|><|video_pad|>×N<|vision_end|>
        #
        # 🎯 GRPO mask 策略（与 SFT Baseline 保持一致）：
        #   - 只 mask video_pad tokens（视觉特征占位符）
        #   - 保留时间戳、vision_start、vision_end 的学习
        #   - 这样模型可以学习预测视频边界和时间信息
        #
        # [备选策略] 全部mask：如果认为这些token都是代码插入的，理论上不需要学习
        # VIDEO_TOKENS_TO_MASK = {151652, 151653, 151656}  # vision_start, vision_end, video_pad
        VIDEO_TOKENS_TO_MASK = {151656}  # 只 mask video_pad，保留 vision_start(151652) 和 vision_end(151653)

        completion_loss_mask = []      # 用于 loss 计算
        completion_attention_mask = [] # 用于 model forward
        total_tokens = 0
        total_video_masked = 0
        total_timestamp_masked = 0
        total_pad_masked = 0

        for idx, ids in enumerate(completion_ids):
            loss_mask = torch.ones_like(ids)
            attn_mask = torch.ones_like(ids)  # 🎯 新增: attention mask

            # 1. Mask video tokens (vision_start, vision_end, video_pad)
            # 只在 loss_mask 中 mask，attention_mask 保持为 1
            for video_token in VIDEO_TOKENS_TO_MASK:
                token_mask = (ids == video_token)
                total_video_masked += token_mask.sum().item()
                loss_mask[token_mask] = 0
                # attn_mask[token_mask] = 1  # 保持为 1，不需要改

            # 2. 时间戳 tokens (e.g., <26.0 seconds>) - 保留学习，与 SFT Baseline 保持一致
            # [备选策略] 如果需要 mask 时间戳，取消下面的注释：
            # timestamp_positions = find_timestamp_token_positions(ids, self.processing_class)
            # if timestamp_positions:
            #     for pos in timestamp_positions:
            #         if pos < len(loss_mask):
            #             loss_mask[pos] = 0
            #             total_timestamp_masked += 1

            # 3. Mask pad tokens - 在两个 mask 中都设为 0
            pad_mask = (ids == self.pad_token_id)
            total_pad_masked += pad_mask.sum().item()
            loss_mask[pad_mask] = 0
            attn_mask[pad_mask] = 0  # pad tokens 在 attention 中也要 mask

            total_tokens += len(ids)
            completion_loss_mask.append(loss_mask)
            completion_attention_mask.append(attn_mask)

        # Calculate effective tokens (for DAPO loss normalization)
        total_masked = total_video_masked + total_timestamp_masked + total_pad_masked
        effective_tokens = total_tokens - total_masked
        if total_masked > 0:
            print(f"🎭 [TOKEN MASK] Masked {total_video_masked} video + {total_timestamp_masked} timestamp + {total_pad_masked} pad = {total_masked} total, effective: {effective_tokens}/{total_tokens} ({100*effective_tokens/total_tokens:.1f}%)")

        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_loss_mask = pad(completion_loss_mask, padding_value=0, padding_side="right")
        completion_attention_mask = pad(completion_attention_mask, padding_value=0, padding_side="right")

        # Handle per-token logps
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        # Handle tools
        if tool_mask_list is not None and self.tools:
            tool_mask = [torch.tensor(mask, device=device) for mask in tool_mask_list]
            tool_mask = pad(tool_mask, padding_value=1, padding_side="right")
        else:
            tool_mask = None

        # Mask truncated completions
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            # 🎯 只影响 loss_mask，不影响 attention_mask
            completion_loss_mask = completion_loss_mask * (~is_truncated).unsqueeze(1).int()

        # Prepare for forward pass
        # 🎯 关键区分:
        # - attention_mask: 用于 model forward，视频 tokens = 1
        # - completion_loss_mask (下面保存为 completion_mask): 用于 loss，视频 tokens = 0
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_attention_mask], dim=1)  # 🎯 用 attention_mask
        completion_mask = completion_loss_mask  # 🎯 loss mask 保存为 completion_mask（供后续使用）

        # Keep only completion logits
        logits_to_keep = completion_ids.shape[1]

        # Calculate batch size
        batch_size = prompt_ids.shape[0]

        # ⭐ Compute old_per_token_logps and ref_per_token_logps (following TRL's logic)
        # 🎯 关键修复：直接使用生成过程中保存的 per-sample 视频特征
        # 这些特征在生成时就已经验证过匹配，不需要重新合并！
        from trl.models.utils import disable_gradient_checkpointing

        # Prepare video features for logps computation
        combined_pixel_values_videos = None
        combined_video_grid_thw = None
        num_videos = None

        if all_videos is not None:
            # 🎯 优先使用 per_sample_video_features（生成时保存的完整特征）
            per_sample_features = extra_fields.get("per_sample_video_features")

            if per_sample_features:
                print(f"🎥 [LOGPS] 使用生成时保存的 per-sample 视频特征 (共 {len(per_sample_features)} 个 sample)")

                # 🎯 关键修复：保持视频特征为 list 格式（每个 sample 一个 tensor）
                # 这样 _get_per_token_logps_and_entropies_with_video 会使用正确的 list slicing 逻辑
                # 而不是有问题的 concatenated tensor + num_videos slicing 逻辑
                pixel_values_videos_list = []  # list[Tensor], 每个 sample 的 pixel_values
                video_grid_thw_list = []       # list[Tensor], 每个 sample 的 grid_thw
                num_videos = []

                for i, feat in enumerate(per_sample_features):
                    pv = feat['pixel_values_videos']
                    gt = feat['video_grid_thw']
                    nv = feat['num_videos']

                    if pv is not None:
                        pixel_values_videos_list.append(pv.to(device))
                    else:
                        pixel_values_videos_list.append(None)

                    if gt is not None:
                        # 处理 video_grid_thw 的维度，确保是 [num_videos, 3]
                        if gt.dim() == 1:
                            video_grid_thw_list.append(gt.unsqueeze(0).to(device))
                        else:
                            video_grid_thw_list.append(gt.to(device))
                    else:
                        video_grid_thw_list.append(None)

                    num_videos.append(nv)

                    print(f"   Sample {i}: pv={pv.shape if pv is not None else None}, gt={gt.shape if gt is not None else None}, num_videos={nv}")

                # 🎯 使用 list 格式传递给 _get_per_token_logps_and_entropies_with_video
                # 这样会走 lines 192-204 的正确 slicing 逻辑
                combined_pixel_values_videos = pixel_values_videos_list
                combined_video_grid_thw = video_grid_thw_list

                # 🔍 验证匹配（使用 list 格式）
                video_pad_token_id = 151656
                merge_size = 2

                # 计算每个 sample 的 video_tokens 和 features
                all_match = True
                for sample_idx in range(prompt_completion_ids.shape[0]):
                    sample_video_tokens = (prompt_completion_ids[sample_idx] == video_pad_token_id).sum().item()
                    sample_expected_features = sample_video_tokens * (merge_size ** 2)
                    sample_pv = pixel_values_videos_list[sample_idx]
                    sample_gt = video_grid_thw_list[sample_idx]
                    sample_actual_features = sample_pv.shape[0] if sample_pv is not None else 0
                    sample_grid_features = sample_gt.prod(dim=-1).sum().item() if sample_gt is not None else 0

                    match_status = "✅" if (sample_actual_features == sample_grid_features == sample_expected_features) else "⚠️"
                    print(f"   {match_status} Sample {sample_idx}: video_tokens={sample_video_tokens}, "
                          f"expected_features={sample_expected_features}, actual_features={sample_actual_features}, "
                          f"grid_features={sample_grid_features}")

                    if sample_actual_features != sample_grid_features:
                        all_match = False
                        print(f"      ❌ pv.shape[0] != grid.prod().sum()! 这是一个严重错误")

                if all_match:
                    print(f"✅ [LOGPS] 所有 sample 的视频特征验证通过")
                    self._skip_kl_due_to_video_mismatch = False
                else:
                    print(f"⚠️ [LOGPS] 部分 sample 的视频特征不匹配")
                    self._skip_kl_due_to_video_mismatch = False  # 仍然尝试计算，让 list slicing 自然工作
                self._prompt_completion_ids_for_logps = None

            else:
                # Fallback: 使用旧的合并逻辑（没有 per_sample_features 时）
                print(f"⚠️ [LOGPS] per_sample_video_features 不可用，使用 fallback 逻辑")
                initial_pixel_values = extra_fields.get("pixel_values_videos")
                initial_video_grid_thw = extra_fields.get("video_grid_thw")

                if initial_pixel_values is not None:
                    # 🎯 转换为 list 格式以保持一致性
                    # 假设每个 sample 有一个初始视频，需要根据 video_grid_thw 切分 pixel_values
                    if isinstance(initial_pixel_values, list):
                        # 已经是 list 格式
                        combined_pixel_values_videos = initial_pixel_values
                        combined_video_grid_thw = initial_video_grid_thw if isinstance(initial_video_grid_thw, list) else [initial_video_grid_thw[i:i+1] for i in range(len(initial_pixel_values))]
                    else:
                        # 需要切分为 list 格式
                        pixel_values_list = []
                        grid_thw_list = []
                        if initial_video_grid_thw is not None and initial_video_grid_thw.shape[0] == batch_size:
                            # 每个 sample 有一个视频
                            start_idx = 0
                            for i in range(batch_size):
                                grid = initial_video_grid_thw[i]
                                if grid.dim() == 1:
                                    num_features = grid[0] * grid[1] * grid[2]
                                else:
                                    num_features = grid.squeeze(0).prod()
                                num_features = num_features.item()
                                end_idx = start_idx + num_features
                                pixel_values_list.append(initial_pixel_values[start_idx:end_idx])
                                grid_thw_list.append(grid.unsqueeze(0) if grid.dim() == 1 else grid)
                                start_idx = end_idx
                            combined_pixel_values_videos = pixel_values_list
                            combined_video_grid_thw = grid_thw_list
                        else:
                            # 无法切分，使用原始 tensor（会走旧逻辑）
                            combined_pixel_values_videos = initial_pixel_values
                            combined_video_grid_thw = initial_video_grid_thw

                    num_videos = [1] * batch_size
                    print(f"🎥 [LOGPS] 使用初始视频特征 (list格式): {len(combined_pixel_values_videos) if isinstance(combined_pixel_values_videos, list) else combined_pixel_values_videos.shape}")
                    self._skip_kl_due_to_video_mismatch = False
                    self._prompt_completion_ids_for_logps = None

        # Prepare forward kwargs for images (if any)
        forward_kwargs = {}
        num_images = None
        if images is not None:
            num_images = [len(img_list) if img_list else 0 for img_list in images]

        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            # 🎯 当视频 token/特征不匹配时，使用替换后的 ids（视频 token 已替换为 pad token）
            ids_for_logps = getattr(self, '_prompt_completion_ids_for_logps', None)
            if ids_for_logps is None:
                ids_for_logps = prompt_completion_ids

            # Compute old_per_token_logps for importance sampling if needed
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and getattr(self, 'vllm_importance_sampling_correction', False)
            ):
                # Use video-aware method if we have video features
                if combined_pixel_values_videos is not None:
                    # Debug: check if using list format (correct) or tensor format (problematic)
                    if isinstance(combined_pixel_values_videos, list):
                        print(f"🔍 [OLD LOGPS] Using list format (correct) - {len(combined_pixel_values_videos)} samples")
                    else:
                        print(f"🔍 [OLD LOGPS] Using tensor format - shape: {combined_pixel_values_videos.shape}")

                    old_per_token_logps, _ = self._get_per_token_logps_and_entropies_with_video(
                        self.model,
                        ids_for_logps,
                        attention_mask,
                        logits_to_keep,
                        batch_size,
                        pixel_values_videos=combined_pixel_values_videos,
                        video_grid_thw=combined_video_grid_thw,
                        num_videos=num_videos,
                    )
                else:
                    old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.model,
                        ids_for_logps,
                        attention_mask,
                        logits_to_keep,
                        batch_size,
                        num_images=num_images,
                        **forward_kwargs,
                    )
            else:
                old_per_token_logps = None

            # 🎯 ref_per_token_logps 不再在这里计算，改为在 _compute_loss 中计算
            # 这样可以复用 _compute_loss 中已有的视频特征处理逻辑，避免 token/feature 不匹配问题
            ref_per_token_logps = None
            print(f"🔍 [DEBUG] ref_per_token_logps 将在 _compute_loss 中计算 (beta={self.beta})")

        print(f"[DEBUG] old_per_token_logps: {'computed' if old_per_token_logps is not None else 'None'}")

        # ⭐ Calculate rewards and advantages (following TRL's logic)
        print("\n" + "="*80)
        print("🎁 REWARD CALCULATION")
        print("="*80)

        # Call parent's _calculate_rewards method
        # ⚠️ 现在直接使用 inputs（TRL 已经做了 prompt 重复）
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        print(f"\n[DEBUG] rewards_per_func shape: {rewards_per_func.shape}")
        print(f"[DEBUG] rewards_per_func values:\n{rewards_per_func}")

        # Compute grouped-wise rewards
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        # ⭐ 支持 multi_objective_aggregation 参数 (TRL 0.28.0+)
        # - "sum_then_normalize": 先加权求和，再归一化（默认）
        # - "normalize_then_sum": 先各自归一化，再加权求和（GDPO 论文推荐）
        from trl.trainer.utils import nanstd
        multi_objective_aggregation = getattr(self, 'multi_objective_aggregation', 'sum_then_normalize')
        print(f"\n🎯 [MULTI-OBJECTIVE] aggregation mode: {multi_objective_aggregation}")

        if multi_objective_aggregation == "sum_then_normalize":
            # 方式1: 先加权求和，再归一化
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            print(f"\n💰 [TOTAL REWARDS] After weighting:")
            for i, reward in enumerate(rewards):
                print(f"  Sample {i}: total_reward = {reward.item():.3f}")

            mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)

            if self.scale_rewards in ["group", "none"]:
                if num_generations > 1:
                    std_rewards = rewards.view(-1, num_generations).std(dim=1)
                    std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
                else:
                    std_rewards = torch.zeros_like(rewards)
            elif self.scale_rewards == "batch":
                if rewards.numel() > 1:
                    std_rewards = rewards.std().expand_as(rewards)
                else:
                    std_rewards = torch.zeros_like(rewards)
            else:
                raise ValueError(f"Invalid scale_rewards: {self.scale_rewards}")

            advantages = rewards - mean_grouped_rewards
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)

        elif multi_objective_aggregation == "normalize_then_sum":
            # 方式2: 先各自归一化，再加权求和（GDPO 论文推荐）
            # 每个 reward function 在 group 内归一化，然后再加权求和
            grouped = rewards_per_func.view(-1, num_generations, len(self.reward_funcs))
            mean_k = torch.nanmean(grouped, dim=1, keepdim=True)
            std_k = nanstd(grouped, dim=1, keepdim=True) if num_generations > 1 else torch.zeros_like(mean_k)
            reward_k = (grouped - mean_k) / (std_k + 1e-4)
            reward_k = reward_k.view(-1, len(self.reward_funcs))
            rewards = (reward_k * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

            print(f"\n💰 [TOTAL REWARDS] After normalize_then_sum:")
            for i, reward in enumerate(rewards):
                print(f"  Sample {i}: total_reward = {reward.item():.3f}")

            # ✅ 修复 Bug 1: 使用 scale_rewards 参数（与 sum_then_normalize 分支一致）
            if self.scale_rewards == "batch":
                mean_grouped_rewards = rewards.mean().expand_as(rewards)
                std_rewards = rewards.std().expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
            elif self.scale_rewards == "group":
                mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
                mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
                if num_generations > 1:
                    std_rewards = rewards.view(-1, num_generations).std(dim=1)
                    std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
                else:
                    std_rewards = torch.zeros_like(rewards)
            elif self.scale_rewards == "none":
                mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
                mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
                std_rewards = torch.ones_like(rewards)
            else:
                raise ValueError(f"Invalid scale_rewards: {self.scale_rewards}")

            advantages = (rewards - mean_grouped_rewards)
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)

        else:
            raise ValueError(f"Invalid multi_objective_aggregation: {multi_objective_aggregation}")

        print(f"\n📊 [GROUP STATISTICS]")
        print(f"  num_generations = {num_generations}")
        print(f"  aggregation = {multi_objective_aggregation}")

        print(f"\n⚖️  [ADVANTAGES]:")
        for i, adv in enumerate(advantages):
            sign = "📈" if adv > 0 else "📉" if adv < 0 else "➡️"
            print(f"  Sample {i}: advantage = {adv.item():+.3f} {sign}")
        print(f"[DEBUG] advantages shape: {advantages.shape}")

        print("="*80 + "\n")

        # ⭐ Log reward metrics to self._metrics (for wandb/tensorboard)
        from accelerate.utils import gather_object

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        all_process_advantages = advantages.clone()  # keep for logging before slicing

        # Log per-function rewards
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)

        # Compute original weighted sum reward (not normalized) for logging
        # This shows the actual model performance in 0-1 range
        original_weighted_reward = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        original_reward_mean = original_weighted_reward.mean().item()

        # Log aggregate reward metrics
        # "reward" now shows the original weighted sum (interpretable, 0-1 range)
        # "reward_normalized" shows the normalized value used for training
        self._metrics[mode]["reward"].append(original_reward_mean)
        self._metrics[mode]["reward_normalized"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # ⭐ 新增：组内 reward std 监控（关键诊断指标）
        rewards_grouped = original_weighted_reward.view(-1, num_generations)
        per_group_std = rewards_grouped.std(dim=-1)
        self._metrics[mode]["group_reward_std_mean"].append(per_group_std.mean().item())
        self._metrics[mode]["group_reward_std_min"].append(per_group_std.min().item())

        # 低方差组比例（std < 0.02 的组）
        low_var_ratio = (per_group_std < 0.02).float().mean().item()
        self._metrics[mode]["low_variance_group_ratio"].append(low_var_ratio)

        # Advantage 统计
        self._metrics[mode]["advantage_abs_mean"].append(advantages.abs().mean().item())
        self._metrics[mode]["advantage_std"].append(advantages.std().item())

        # 正负 advantage 比例（用于判断学习方向）
        pos_adv_ratio = (advantages > 0).float().mean().item()
        self._metrics[mode]["positive_advantage_ratio"].append(pos_adv_ratio)

        # Log prompt and completion texts for visualization
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        print(f"📊 [METRICS LOGGED] reward={original_reward_mean:.4f} (原始加权), reward_normalized={mean_grouped_rewards.mean().item():.4f}, reward_std={std_rewards.mean().item():.4f}")

        # Slice to keep only the local part (for distributed training)
        process_slice = slice(
            self.accelerator.process_index * num_items_in_batch,
            (self.accelerator.process_index + 1) * num_items_in_batch,
        )
        advantages = advantages[process_slice]
        print(f"[DEBUG] advantages (after slicing) shape: {advantages.shape}")

        # Prepare result - ONLY include Tensors and lists (indexable)
        # TRL's shuffle_sequence_dict will try to index ALL fields with tensor indices
        result = {
            "prompt_ids": prompt_ids,                      # Tensor
            "prompt_mask": prompt_mask,                    # Tensor
            "completion_ids": completion_ids,              # Tensor
            "completion_mask": completion_mask,            # Tensor
            "prompt_completion_ids": prompt_completion_ids, # Tensor
            "attention_mask": attention_mask,              # Tensor
            "completions": completions,                    # list[str]
            "advantages": advantages,                      # Tensor - REQUIRED for GRPO!
            # ⭐ CRITICAL FIX: Use effective token count for DAPO loss normalization
            # DAPO loss formula: loss = (per_token_loss * mask).sum() / normalizer
            # After masking video tokens, we must update normalizer to effective_tokens
            # Otherwise: numerator decreases but denominator stays large → extreme loss values
            "num_items_in_batch": torch.tensor(effective_tokens if total_masked > 0 else total_tokens, device=device),
        }

        # Add optional Tensor/list fields
        if sampling_per_token_logps is not None:
            result["sampling_per_token_logps"] = sampling_per_token_logps

        if tool_mask is not None:
            result["tool_mask"] = tool_mask

        if ground_truths is not None:
            result["ground_truths"] = ground_truths

        # ⭐ Add logps for importance sampling and KL divergence (REQUIRED for TRL)
        if old_per_token_logps is not None:
            result["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            result["ref_per_token_logps"] = ref_per_token_logps

        # Add video features for forward pass (following TRL's pattern for images)
        # 🎯 关键修复：使用 concatenated tensor + features_per_sample 格式
        # TRL 的 shuffle_sequence_dict 不能正确处理 list[Tensor]，会导致索引错误
        # 改为使用 TRL 处理图片的方式：concatenated tensor + 边界元数据
        if all_videos is not None:
            per_sample_features = extra_fields.get("per_sample_video_features")

            if per_sample_features:
                # 🎯 收集所有 sample 的视频特征
                all_pixel_values = []
                all_video_grid_thw = []
                features_per_sample = []  # 每个 sample 有多少行 pixel_values
                num_videos_per_sample = []  # 每个 sample 有多少个视频

                for i, feat in enumerate(per_sample_features):
                    pv = feat['pixel_values_videos']
                    gt = feat['video_grid_thw']
                    nv = feat['num_videos']

                    if pv is not None:
                        all_pixel_values.append(pv.to(device))
                        features_per_sample.append(pv.shape[0])
                    else:
                        features_per_sample.append(0)

                    if gt is not None:
                        # 确保 gt 是 [num_videos, 3] 格式
                        if gt.dim() == 1:
                            all_video_grid_thw.append(gt.unsqueeze(0).to(device))
                        else:
                            all_video_grid_thw.append(gt.to(device))

                    num_videos_per_sample.append(nv)

                # 🎯 合并为 concatenated tensor
                if all_pixel_values:
                    concatenated_pv = torch.cat(all_pixel_values, dim=0)
                    concatenated_gt = torch.cat(all_video_grid_thw, dim=0)

                    # 存储为 tensor 格式，加上边界元数据
                    result["pixel_values_videos"] = concatenated_pv
                    result["video_grid_thw"] = concatenated_gt
                    result["features_per_sample"] = features_per_sample  # list[int]
                    result["num_videos_per_sample"] = num_videos_per_sample  # list[int]

                    print(f"[DEBUG] 使用 concatenated tensor 格式:")
                    print(f"[DEBUG]   concatenated_pv shape: {concatenated_pv.shape}")
                    print(f"[DEBUG]   concatenated_gt shape: {concatenated_gt.shape}")
                    print(f"[DEBUG]   features_per_sample: {features_per_sample}")
                    print(f"[DEBUG]   num_videos_per_sample: {num_videos_per_sample}")
                    for i, (fps, nvps) in enumerate(zip(features_per_sample, num_videos_per_sample)):
                        print(f"[DEBUG]   Sample {i}: features={fps}, num_videos={nvps}")
            else:
                # Fallback: per_sample_features 不存在时，使用初始视频特征
                video_grid_thw_from_gen = extra_fields.get("video_grid_thw")
                pixel_values_videos_from_gen = extra_fields.get("pixel_values_videos")

                if video_grid_thw_from_gen is not None and pixel_values_videos_from_gen is not None:
                    # 🎯 同样使用 concatenated tensor 格式
                    result["pixel_values_videos"] = pixel_values_videos_from_gen.to(device)
                    result["video_grid_thw"] = video_grid_thw_from_gen.to(device)

                    # 计算每个 sample 的 features 数量
                    features_per_sample = []
                    num_videos_per_sample = []
                    for idx in range(batch_size):
                        grid = video_grid_thw_from_gen[idx]
                        if grid.dim() == 1 and grid.shape[0] == 3:
                            num_features = (grid[0] * grid[1] * grid[2]).item()
                            nv = 1
                        elif grid.dim() == 2:
                            num_features = grid.prod(dim=-1).sum().item()
                            nv = grid.shape[0]
                        else:
                            raise ValueError(f"Unexpected grid shape: {grid.shape}")
                        features_per_sample.append(num_features)
                        num_videos_per_sample.append(nv)

                    result["features_per_sample"] = features_per_sample
                    result["num_videos_per_sample"] = num_videos_per_sample

                    print(f"[DEBUG] Fallback: 使用初始视频特征 (concatenated tensor)")
                    print(f"[DEBUG]   pixel_values shape: {pixel_values_videos_from_gen.shape}")
                    print(f"[DEBUG]   features_per_sample: {features_per_sample}")
                else:
                    print("[WARNING] Video features not found in extra_fields")

        # Add other extra fields from generation (only if indexable)
        # Skip video features as they've been handled above
        skip_keys = {"pixel_values_videos", "video_grid_thw", "per_sample_video_features"}
        for k, v in extra_fields.items():
            if k in skip_keys:
                continue  # Already handled above
            if isinstance(v, (list, tuple, torch.Tensor)):
                result[k] = v
            else:
                print(f"[WARNING] Skipping non-indexable extra field: {k} (type={type(v).__name__})")

        print("\n🔍 [DEBUG] Result structure before shuffle:")
        for key, val in result.items():
            val_type = type(val).__name__
            is_seq = isinstance(val, (list, tuple))
            status = "✅" if is_seq else "❌"

            if hasattr(val, '__len__') and not isinstance(val, str):
                try:
                    print(f"{status} {key:25s}: {val_type:10s} | len={len(val)}")
                except:
                    print(f"{status} {key:25s}: {val_type:10s} | (no len)")
            else:
                print(f"{status} {key:25s}: {val_type:10s} | NOT INDEXABLE!")
        print()

        return result