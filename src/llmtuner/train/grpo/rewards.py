"""
Custom reward functions for GRPO training of Qwen3VL VMCOT models.

This module implements reward functions to evaluate:
1. Format compliance: Presence of <think> and <answer> tags
2. Answer accuracy: Exact match with ground truth for VQA multiple choice
"""

import re
from typing import List, Dict, Any, Optional
from ...extras.logging import get_logger

logger = get_logger(__name__)


def format_reward(
    prompts: List[str],
    completions: List[str],
    **kwargs
) -> List[float]:
    """
    Check for proper format with fine-grained scoring:

    Scoring breakdown (total 1.0):
    - 0.4 points: Complete <think>...</think> block AND <think> is the FIRST token (no leading spaces)
    - 0.4 points: Complete <answer>...</answer> block
    - 0.2 points: <span>/<fps> tags are properly paired (each <span> has </span>, each <fps> has </fps>)
                  If any unpaired tags exist, this score is 0

    Args:
        prompts: List of input prompts
        completions: List of generated completions
        **kwargs: Additional arguments (not used)

    Returns:
        List of rewards (0.0 to 1.0 based on format compliance)
    """
    rewards = []
    for i, completion in enumerate(completions):
        reward = 0.0

        # ========== Think score (0.4 points) ==========
        # Requirement 1: <think> must be the FIRST token (no leading spaces allowed!)
        starts_with_think = completion.startswith('<think>')

        # Requirement 2: Must have complete <think>...</think> block (both open and close tags)
        has_think_open = '<think>' in completion
        has_think_close = '</think>' in completion
        has_complete_think = has_think_open and has_think_close

        think_score = 0.0
        if starts_with_think and has_complete_think:
            think_score = 0.4

        # ========== Answer score (0.4 points) ==========
        # Requirement: Has complete <answer>...</answer> block
        has_answer_open = '<answer>' in completion
        has_answer_close = '</answer>' in completion
        has_complete_answer = has_answer_open and has_answer_close

        answer_score = 0.0
        if has_complete_answer:
            answer_score = 0.4

        # ========== Span/FPS pairing score (0.2 points) ==========
        # Check that <span> and </span> are properly paired (same count)
        # Check that <fps> and </fps> are properly paired (same count)
        # If any tag is unpaired, score is 0
        span_open_count = completion.count('<span>')
        span_close_count = completion.count('</span>')
        fps_open_count = completion.count('<fps>')
        fps_close_count = completion.count('</fps>')

        span_paired = (span_open_count == span_close_count)
        fps_paired = (fps_open_count == fps_close_count)

        # Additional check: if <span> exists, <fps> should also exist (they come in pairs)
        # <span>X-Y</span><fps>Z</fps> pattern
        span_fps_consistent = True
        if span_open_count > 0 or fps_open_count > 0:
            # If one exists, the other should exist with same count
            span_fps_consistent = (span_open_count == fps_open_count)

        pairing_score = 0.0
        if span_paired and fps_paired and span_fps_consistent:
            pairing_score = 0.2

        # Total reward
        reward = think_score + answer_score + pairing_score
        rewards.append(reward)

        # Always print detailed format check results
        print(f"📋 [FORMAT] Sample {i}: reward={reward:.2f} | think={think_score:.1f} | answer={answer_score:.1f} | pairing={pairing_score:.1f}")
        print(f"   📊 Tags count: <span>={span_open_count}, </span>={span_close_count}, <fps>={fps_open_count}, </fps>={fps_close_count}")

        if reward < 1.0:
            # Show first 50 chars to see what the first token is
            first_chars = completion[:50].replace('\n', '\\n')
            print(f"  ⚠️  Partial format. First 50 chars: '{first_chars}'")
            if think_score == 0.0:
                if not starts_with_think:
                    print(f"  ❌ <think> is not the first token! First char: '{completion[0] if completion else 'EMPTY'}' (ord={ord(completion[0]) if completion else -1})")
                if not has_complete_think:
                    print(f"  ❌ Incomplete think block: <think>={has_think_open}, </think>={has_think_close}")
            if answer_score == 0.0:
                print(f"  ❌ Incomplete answer block: <answer>={has_answer_open}, </answer>={has_answer_close}")
            if pairing_score == 0.0:
                if not span_paired:
                    print(f"  ❌ Unpaired <span> tags: open={span_open_count}, close={span_close_count}")
                if not fps_paired:
                    print(f"  ❌ Unpaired <fps> tags: open={fps_open_count}, close={fps_close_count}")
                if not span_fps_consistent:
                    print(f"  ❌ <span>/<fps> count mismatch: span={span_open_count}, fps={fps_open_count}")

    return rewards


def _parse_time_span(text: str) -> tuple:
    """
    解析时间段字符串，返回 (start, end)
    支持格式: "3.00 - 8.20", "3.0-8.2", "3 - 8"
    """
    match = re.match(r'^\s*([\d.]+)\s*-\s*([\d.]+)\s*$', str(text).strip())
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def _compute_time_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """
    计算两个时间段的 IoU（Intersection over Union）
    """
    # 计算交集
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0, inter_end - inter_start)

    # 计算并集
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def _get_difficulty_params(gt_duration: float) -> tuple:
    """
    根据 GT 时间段长度动态选择难度参数 alpha 和 beta。

    时间段越长，定位越难，所以降低 alpha/beta 阈值，让模型更容易获得奖励。

    Difficulty-aware scaling (论文 Algorithm 1):
    - alpha: IoU 低于此值时 reward=0
    - beta: IoU 高于此值时 reward=1
    - alpha < IoU < beta 时线性插值

    更新后的阈值（更严格，增加学习难度）：
    - 短时间段 (< 5s): alpha=0.6, beta=0.95 (要求 IoU>0.6 才有分，>0.95 才满分)
    - 中等时间段 (5-15s): alpha=0.5, beta=0.9
    - 长时间段 (15-30s): alpha=0.4, beta=0.85
    - 超长时间段 (> 30s): alpha=0.3, beta=0.8

    Args:
        gt_duration: GT 时间段长度（秒）

    Returns:
        (alpha, beta) 难度参数
    """
    if gt_duration < 5:
        # 短时间段：容易定位，高阈值（更严格）
        return 0.6, 0.95
    elif gt_duration < 15:
        # 中等时间段
        return 0.5, 0.9
    elif gt_duration < 30:
        # 长时间段：较难定位
        return 0.4, 0.85
    else:
        # 超长时间段：最难定位
        return 0.3, 0.8


def accuracy_reward(
    prompts: List[str],
    completions: List[str],
    ground_truths: List[str] = None,
    use_difficulty_aware: bool = True,
    default_alpha: float = 0.1,
    default_beta: float = 0.6,
    **kwargs
) -> List[float]:
    """
    Extract answer from <answer> tags and compare with ground truth.

    自动识别任务类型：
    - 选择题: answer 字段是 A/B/C/D/E，使用精确匹配
    - Time Grounding: gt_span_start/gt_span_end 存在，使用 Difficulty-aware IoU 计算

    Difficulty-aware IoU 缩放 (DGRPO 论文 Algorithm 1):
    - 根据 GT 时间段长度动态选择 alpha/beta
    - 时间段越长（更难定位），alpha/beta 越低，让模型更容易获得奖励
    - scaled_iou = clamp((IoU - alpha) / (beta - alpha), 0, 1)

    Args:
        prompts: List of input prompts
        completions: List of generated completions
        ground_truths: List of correct answers (from kwargs or dataset)
        use_difficulty_aware: 是否使用 difficulty-aware 缩放 (默认 True)
        default_alpha: 默认 IoU 下界阈值 (当不使用 difficulty-aware 时)
        default_beta: 默认 IoU 上界阈值 (当不使用 difficulty-aware 时)
        **kwargs: Additional arguments (may contain 'answer', 'gt_span_start', 'gt_span_end' fields)

    Returns:
        List of rewards (0.0 to 1.0)
    """
    # Get ground truths from kwargs if not provided directly
    if ground_truths is None:
        ground_truths = kwargs.get('answer', [None] * len(completions))

    # Get time grounding ground truths
    gt_span_starts = kwargs.get('gt_span_start', [None] * len(completions))
    gt_span_ends = kwargs.get('gt_span_end', [None] * len(completions))

    # 选择题答案的有效字母集合
    VALID_MC_ANSWERS = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'}

    rewards = []
    for i, (completion, ground_truth) in enumerate(zip(completions, ground_truths)):
        gt_start = gt_span_starts[i] if i < len(gt_span_starts) else None
        gt_end = gt_span_ends[i] if i < len(gt_span_ends) else None

        # 判断任务类型：
        # 1. 如果 answer 是单个字母 (A-H)，则是选择题任务
        # 2. 否则，如果 gt_span_start 和 gt_span_end 都存在，则是 time grounding 任务
        # 这样可以正确处理同时有 answer 和 gt_span 的数据
        answer_str = str(ground_truth).strip().upper() if ground_truth else ""
        is_multiple_choice = answer_str in VALID_MC_ANSWERS
        is_time_grounding = (not is_multiple_choice) and (gt_start is not None and gt_end is not None)

        # Extract answer from tags
        match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
        if match:
            extracted_answer = match.group(1).strip()

            if is_time_grounding:
                # ===== Time Grounding 任务：使用 Difficulty-aware IoU 计算 =====
                pred_start, pred_end = _parse_time_span(extracted_answer)

                if pred_start is not None and pred_end is not None:
                    # 计算 IoU
                    gt_start = float(gt_start)
                    gt_end = float(gt_end)
                    iou = _compute_time_iou(pred_start, pred_end, gt_start, gt_end)

                    # 根据 GT 时间段长度选择 alpha/beta (Difficulty-aware)
                    gt_duration = gt_end - gt_start
                    if use_difficulty_aware:
                        alpha, beta = _get_difficulty_params(gt_duration)
                    else:
                        alpha, beta = default_alpha, default_beta

                    # 使用 alpha/beta 缩放 IoU
                    # scaled_iou = (iou - alpha) / (beta - alpha), 然后 clamp 到 [0, 1]
                    if beta > alpha:
                        scaled_iou = (iou - alpha) / (beta - alpha)
                        reward = max(0.0, min(1.0, scaled_iou))
                    else:
                        reward = 1.0 if iou >= alpha else 0.0

                    status = "✅" if reward >= 0.8 else ("⚠️" if reward > 0 else "❌")
                    print(f"🎯 [ACCURACY-TIME] Sample {i}: reward={reward:.2f} (scaled) | IoU={iou:.2f} | gt_dur={gt_duration:.1f}s α={alpha} β={beta} | pred=[{pred_start:.1f}-{pred_end:.1f}] vs gt=[{gt_start:.1f}-{gt_end:.1f}] {status}")
                else:
                    # 预测格式不正确
                    reward = 0.0
                    print(f"🎯 [ACCURACY-TIME] Sample {i}: reward=0.0 | ❌ Invalid time format: '{extracted_answer}' (expected: 'X.XX - Y.YY', gt=[{gt_start}-{gt_end}])")
            else:
                # ===== 选择题任务：使用精确匹配 =====
                if ground_truth is None:
                    logger.warning(f"No ground truth provided for completion {i}, returning 0.0")
                    rewards.append(0.0)
                    continue

                extracted_answer = extracted_answer.upper().strip()
                normalized_ground_truth = str(ground_truth).upper().strip()

                # Check for exact match
                reward = 1.0 if extracted_answer == normalized_ground_truth else 0.0

                status = "✅" if reward == 1.0 else "❌"
                print(f"🎯 [ACCURACY-MC] Sample {i}: reward={reward:.1f} | extracted='{extracted_answer}' vs ground_truth='{normalized_ground_truth}' {status}")
        else:
            reward = 0.0
            # Show where we expected to find the answer tag
            answer_region = completion[-200:] if len(completion) > 200 else completion
            answer_region = answer_region.replace('\n', '\\n')
            task_type = "TIME" if is_time_grounding else "MC"
            print(f"🎯 [ACCURACY-{task_type}] Sample {i}: reward=0.0 | ❌ No <answer> tag found")
            print(f"  Last 200 chars: ...{answer_region}")

        rewards.append(reward)

    return rewards


def combined_reward(
    prompts: List[str],
    completions: List[str],
    ground_truths: List[str] = None,
    format_weight: float = 0.3,
    accuracy_weight: float = 0.7,
    **kwargs
) -> List[float]:
    """
    Combine format and accuracy rewards with configurable weights.

    Args:
        prompts: List of input prompts
        completions: List of generated completions
        ground_truths: List of correct answers
        format_weight: Weight for format compliance (default: 0.3)
        accuracy_weight: Weight for answer accuracy (default: 0.7)
        **kwargs: Additional arguments passed to sub-functions

    Returns:
        List of combined rewards
    """
    # Get weights from kwargs if provided (for configuration flexibility)
    format_weight = kwargs.get('format_weight', format_weight)
    accuracy_weight = kwargs.get('accuracy_weight', accuracy_weight)

    # Normalize weights to sum to 1.0
    total_weight = format_weight + accuracy_weight
    if total_weight > 0:
        format_weight = format_weight / total_weight
        accuracy_weight = accuracy_weight / total_weight
    else:
        format_weight = 0.5
        accuracy_weight = 0.5

    # Calculate individual rewards
    format_rewards = format_reward(prompts, completions, **kwargs)
    accuracy_rewards = accuracy_reward(prompts, completions, ground_truths, **kwargs)

    # Combine rewards
    combined = []
    for i, (f_reward, a_reward) in enumerate(zip(format_rewards, accuracy_rewards)):
        combined_value = format_weight * f_reward + accuracy_weight * a_reward
        combined.append(combined_value)

        logger.debug(f"Combined reward for completion {i}: format={f_reward}, accuracy={a_reward}, combined={combined_value}")

    return combined


def create_reward_function(finetuning_args):
    """
    Create a reward function with configuration from finetuning arguments.

    Args:
        finetuning_args: Finetuning arguments containing reward weights

    Returns:
        Reward function configured with specified weights
    """
    format_weight = getattr(finetuning_args, 'grpo_format_weight', 0.3)
    accuracy_weight = getattr(finetuning_args, 'grpo_accuracy_weight', 0.7)

    def configured_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        return combined_reward(
            prompts,
            completions,
            format_weight=format_weight,
            accuracy_weight=accuracy_weight,
            **kwargs
        )

    logger.info(f"Created reward function with format_weight={format_weight}, accuracy_weight={accuracy_weight}")

    return configured_reward


def create_reward_functions(finetuning_args):
    """
    Create separate reward functions based on finetuning arguments.

    Dynamically includes reward functions based on their weights:
    - format_reward: always included if weight > 0
    - accuracy_reward: always included if weight > 0
    - fps_reward: included if grpo_fps_weight > 0
    - time_iou_reward_with_difficulty: included if grpo_time_iou_weight > 0

    Args:
        finetuning_args: Finetuning arguments with reward configuration

    Returns:
        A list of reward functions
    """
    reward_funcs = []
    func_names = []

    format_weight = getattr(finetuning_args, 'grpo_format_weight', 0.3)
    accuracy_weight = getattr(finetuning_args, 'grpo_accuracy_weight', 0.7)
    fps_weight = getattr(finetuning_args, 'grpo_fps_weight', 0.0)
    time_iou_weight = getattr(finetuning_args, 'grpo_time_iou_weight', 0.0)

    if format_weight > 0:
        reward_funcs.append(format_reward)
        func_names.append("format_reward")

    if accuracy_weight > 0:
        reward_funcs.append(accuracy_reward)
        func_names.append("accuracy_reward")

    if fps_weight > 0:
        reward_funcs.append(fps_reward)
        func_names.append("fps_reward")

    if time_iou_weight > 0:
        reward_funcs.append(time_iou_reward_with_difficulty)
        func_names.append("time_iou_reward")

    logger.info(f"Created {len(reward_funcs)} reward functions: {func_names}")

    return reward_funcs


def get_reward_weights(finetuning_args):
    """
    Get reward weights from finetuning arguments.

    Only returns weights for enabled reward functions (weight > 0).

    Args:
        finetuning_args: Finetuning arguments with reward configuration

    Returns:
        A list of reward weights
    """
    weights = []
    weight_names = []

    format_weight = getattr(finetuning_args, 'grpo_format_weight', 0.3)
    accuracy_weight = getattr(finetuning_args, 'grpo_accuracy_weight', 0.7)
    fps_weight = getattr(finetuning_args, 'grpo_fps_weight', 0.0)
    time_iou_weight = getattr(finetuning_args, 'grpo_time_iou_weight', 0.0)

    if format_weight > 0:
        weights.append(format_weight)
        weight_names.append(f"format={format_weight}")

    if accuracy_weight > 0:
        weights.append(accuracy_weight)
        weight_names.append(f"accuracy={accuracy_weight}")

    if fps_weight > 0:
        weights.append(fps_weight)
        weight_names.append(f"fps={fps_weight}")

    if time_iou_weight > 0:
        weights.append(time_iou_weight)
        weight_names.append(f"time_iou={time_iou_weight}")

    logger.info(f"Reward weights: {', '.join(weight_names)}")

    return weights


def fps_reward(
    prompts: List[str],
    completions: List[str],
    ground_truth_fps: List[float] = None,
    max_fps: float = 5.0,
    **kwargs
) -> List[float]:
    """
    FPS 匹配奖励：评估预测的 fps 与 ground truth fps 的匹配程度。

    使用平滑的误差函数，而非二元匹配，使得接近正确答案也能获得部分奖励。

    Args:
        prompts: 输入 prompts
        completions: 生成的回答
        ground_truth_fps: ground truth fps 列表（从 kwargs['gemini_suggested_fps'] 获取）
        max_fps: 最大 fps 值，用于归一化误差（默认 5.0）
        **kwargs: 可能包含 'gemini_suggested_fps' 字段

    Returns:
        List[float]: 奖励列表，范围 [0, 1]
    """
    # 从 kwargs 获取 ground truth fps
    if ground_truth_fps is None:
        ground_truth_fps = kwargs.get('fps', [None] * len(completions))

    rewards = []
    for i, (completion, gt_fps) in enumerate(zip(completions, ground_truth_fps)):
        if gt_fps is None:
            logger.warning(f"No ground truth fps for sample {i}, returning 0.0")
            rewards.append(0.0)
            continue

        # 从回答中提取预测的 fps
        fps_match = re.search(r'<fps>([\d.]+)</fps>', completion)

        if fps_match:
            pred_fps = float(fps_match.group(1))
            gt_fps = float(gt_fps)

            # 计算误差并转换为奖励
            # reward = max(0, 1 - |pred - gt| / max_fps)
            error = abs(pred_fps - gt_fps)
            reward = max(0.0, 1.0 - error / max_fps)

            status = "✅" if reward >= 0.9 else ("⚠️" if reward > 0 else "❌")
            print(f"🎚️ [FPS] Sample {i}: reward={reward:.2f} | pred_fps={pred_fps}, gt_fps={gt_fps}, error={error:.1f} {status}")
        else:
            reward = 0.0
            print(f"🎚️ [FPS] Sample {i}: reward=0.0 | ❌ No <fps> tag found")

        rewards.append(reward)

    return rewards


def time_iou_reward_with_difficulty(
    prompts: List[str],
    completions: List[str],
    ground_truth_spans: List[tuple] = None,
    alpha: float = 0.1,
    beta: float = 0.6,
    **kwargs
) -> List[float]:
    """
    带难度缩放的时间段 IoU 奖励。

    将原始 IoU 通过难度参数缩放到 [0, 1] 范围：
    - alpha: 下界阈值，IoU <= alpha 时 reward = 0
    - beta: 上界阈值，IoU >= beta 时 reward = 1
    - alpha < IoU < beta 时，线性插值

    scaled_iou = (iou - alpha) / (beta - alpha)
    reward = clamp(scaled_iou, 0, 1)

    Args:
        prompts: 输入 prompts
        completions: 生成的回答
        ground_truth_spans: ground truth 时间段列表 [(start, end), ...]
        alpha: 下界阈值（默认 0.1）
        beta: 上界阈值（默认 0.6）
        **kwargs: 可能包含 'gt_span_start', 'gt_span_end' 字段

    Returns:
        List[float]: 缩放后的 IoU 奖励列表，范围 [0, 1]
    """
    # 从 kwargs 获取 ground truth spans
    if ground_truth_spans is None:
        gt_starts = kwargs.get('gt_span_start', [None] * len(completions))
        gt_ends = kwargs.get('gt_span_end', [None] * len(completions))
        ground_truth_spans = list(zip(gt_starts, gt_ends))

    rewards = []
    for i, (completion, gt_span) in enumerate(zip(completions, ground_truth_spans)):
        gt_start, gt_end = gt_span

        if gt_start is None or gt_end is None:
            logger.warning(f"No ground truth span for sample {i}, returning 0.0")
            rewards.append(0.0)
            continue

        # 从回答中提取预测的时间段
        span_match = re.search(r'<span>([\d.]+)\s*-\s*([\d.]+)</span>', completion)

        if span_match:
            pred_start = float(span_match.group(1))
            pred_end = float(span_match.group(2))
            gt_start = float(gt_start)
            gt_end = float(gt_end)

            # 计算 IoU
            intersection_start = max(pred_start, gt_start)
            intersection_end = min(pred_end, gt_end)
            intersection = max(0.0, intersection_end - intersection_start)

            union_start = min(pred_start, gt_start)
            union_end = max(pred_end, gt_end)
            union = union_end - union_start

            if union > 0:
                iou = intersection / union
            else:
                iou = 0.0

            # 难度缩放
            if beta > alpha:
                scaled_iou = (iou - alpha) / (beta - alpha)
                reward = max(0.0, min(1.0, scaled_iou))
            else:
                reward = 1.0 if iou >= alpha else 0.0

            status = "✅" if reward >= 0.8 else ("⚠️" if reward > 0.2 else "❌")
            print(f"⏱️ [TIME_IOU] Sample {i}: reward={reward:.2f} | pred=[{pred_start:.1f}-{pred_end:.1f}], gt=[{gt_start:.1f}-{gt_end:.1f}], raw_iou={iou:.2f}, scaled={reward:.2f} (α={alpha}, β={beta}) {status}")
        else:
            reward = 0.0
            print(f"⏱️ [TIME_IOU] Sample {i}: reward=0.0 | ❌ No <span> tag found")

        rewards.append(reward)

    return rewards


def combined_reward_with_fps_iou(
    prompts: List[str],
    completions: List[str],
    ground_truths: List[str] = None,
    format_weight: float = 0.2,
    accuracy_weight: float = 0.4,
    fps_weight: float = 0.2,
    time_iou_weight: float = 0.2,
    **kwargs
) -> List[float]:
    """
    组合 reward：包含格式、准确性、FPS 和时间 IoU 四个维度。

    Args:
        prompts: 输入 prompts
        completions: 生成的回答
        ground_truths: 正确答案
        format_weight: 格式权重（默认 0.2）
        accuracy_weight: 准确性权重（默认 0.4）
        fps_weight: FPS 匹配权重（默认 0.2）
        time_iou_weight: 时间 IoU 权重（默认 0.2）
        **kwargs: 包含 ground truth 信息

    Returns:
        List[float]: 组合奖励列表，范围 [0, 1]
    """
    # 归一化权重
    total_weight = format_weight + accuracy_weight + fps_weight + time_iou_weight
    if total_weight > 0:
        format_weight /= total_weight
        accuracy_weight /= total_weight
        fps_weight /= total_weight
        time_iou_weight /= total_weight

    # 计算各维度 reward
    format_rewards = format_reward(prompts, completions, **kwargs)
    accuracy_rewards = accuracy_reward(prompts, completions, ground_truths, **kwargs)
    fps_rewards = fps_reward(prompts, completions, **kwargs)
    time_iou_rewards = time_iou_reward_with_difficulty(prompts, completions, **kwargs)

    # 组合
    combined = []
    for i, (f_r, a_r, fps_r, iou_r) in enumerate(zip(format_rewards, accuracy_rewards, fps_rewards, time_iou_rewards)):
        combined_value = (
            format_weight * f_r +
            accuracy_weight * a_r +
            fps_weight * fps_r +
            time_iou_weight * iou_r
        )
        combined.append(combined_value)

        print(f"🎯 [COMBINED] Sample {i}: total={combined_value:.2f} | format={f_r:.2f}×{format_weight:.2f} + accuracy={a_r:.2f}×{accuracy_weight:.2f} + fps={fps_r:.2f}×{fps_weight:.2f} + iou={iou_r:.2f}×{time_iou_weight:.2f}")

    return combined