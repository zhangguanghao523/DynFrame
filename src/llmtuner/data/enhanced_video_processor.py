"""
Enhanced video processor that combines qwen_vl_utils robust video loading
with transformers efficient batch processing.
"""

import sys
from typing import List, Union, Optional, Any, Dict, Callable
import torch
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
from transformers.video_utils import VideoInput, VideoMetadata, make_batched_videos, make_batched_metadata

from .aligned_video_loader import load_video_aligned, load_video_aligned_with_segment_config


class EnhancedQwen3VLVideoProcessor(Qwen3VLVideoProcessor):
    """
    Enhanced Qwen3VL video processor that uses qwen_vl_utils for video loading
    while maintaining transformers' efficient batch processing capabilities.

    Key improvements:
    - Multi-backend video loading (decord/torchcodec/torchvision) via qwen_vl_utils
    - Robust error handling for video files
    - Maintains compatibility with existing transformers pipeline
    - Preserves efficient batch processing via group_videos_by_shape()

    Architecture:
    - Override _decode_and_sample_videos for video I/O only
    - Keep all other transformers optimizations (grouping, patching, etc.)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # print(f"🚀🚀🚀 [DEBUG] EnhancedQwen3VLVideoProcessor初始化完成 🚀🚀🚀", file=sys.stderr)

    def _decode_and_sample_videos(
        self,
        videos: VideoInput,
        video_metadata: Union[VideoMetadata, dict],
        do_sample_frames: Optional[bool] = None,
        sample_indices_fn: Optional[Callable] = None,
        video_configs: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> tuple[list[torch.Tensor], list[VideoMetadata]]:
        """
        Enhanced video decoding using qwen_vl_utils backends for robust video loading.

        Only handles string video paths with multi-backend support.
        Falls back to parent implementation for all other input types.

        Args:
            video_configs: 视频配置映射，支持时间段采样
                格式: {video_index: {'type': 'segment'|'simple', 'video_start': float, 'video_end': float, 'fps': float}}
        """
        # 🎯 从实例变量获取video_configs（高效解决方案）
        if video_configs is None:
            video_configs = getattr(self, '_current_video_configs', None)

        # print(f"🎬🎬🎬 [DEBUG] Enhanced._decode_and_sample_videos()被调用! 配置数量: {len(video_configs) if video_configs else 0} 🎬🎬🎬", file=sys.stderr)

        # Convert to batched format
        videos = make_batched_videos(videos)
        video_metadata = make_batched_metadata(videos, video_metadata=video_metadata)

        processed_videos = []
        processed_metadata = []

        for i, (video, metadata) in enumerate(zip(videos, video_metadata)):
            # print(f"📹📹📹 [DEBUG] 处理第{i+1}个视频: {type(video)} 📹📹📹", file=sys.stderr)

            if isinstance(video, str):
                # Use enhanced loader for file paths/URLs with smart parameter adjustment
                try:
                    # 获取 rank 信息用于调试
                    try:
                        import torch.distributed as dist
                        rank = dist.get_rank() if dist.is_initialized() else -1
                    except Exception:
                        rank = -1

                    # 获取该视频的配置 - 🎯 修复：使用string key
                    video_config = video_configs.get(str(i), {}) if video_configs else {}

                    import time as _time
                    _load_start = _time.time()

                    if video_config.get('type') == 'segment':
                        # 时间段采样 - 使用qwen_vl_utils的现有功能！
                        print(f"⏱️⏱️⏱️ [DEBUG] [Rank {rank}] 时间段采样: {video_config['video_start']}-{video_config['video_end']}s @ {video_config['fps']}fps | 视频: {video} ⏱️⏱️⏱️", file=sys.stderr)

                        video_tensor, video_meta = load_video_aligned_with_segment_config(
                            video_path=video,
                            video_start=video_config['video_start'],
                            video_end=video_config['video_end'],
                            fps=video_config['fps']
                        )
                    else:
                        # 全视频采样 - 使用现有逻辑
                        print(f"🎥🎥🎥 [DEBUG] [Rank {rank}] 全视频采样 @ {self.fps}fps, max_frames={self.max_frames} | 视频: {video} 🎥🎥🎥", file=sys.stderr)
                        video_tensor, video_meta = load_video_aligned(
                            video_path=video,
                            fps=self.fps,
                            max_frames=self.max_frames,
                        )

                    _load_time = _time.time() - _load_start
                    if _load_time > 10:  # 超过10秒打印警告
                        print(f"🐢🐢🐢 [SLOW] [Rank {rank}] 视频加载耗时 {_load_time:.1f}s | 视频: {video} 🐢🐢🐢", file=sys.stderr)
                    processed_videos.append(video_tensor)
                    processed_metadata.append(video_meta)
                    print(f"✅✅✅ [DEBUG] [Rank {rank}] 增强加载器成功处理视频{i+1}, shape={video_tensor.shape} ✅✅✅", file=sys.stderr)

                except Exception as e:
                    print(f"⚠️⚠️⚠️ [WARNING] 增强加载器失败: {e}，使用父类实现 ⚠️⚠️⚠️", file=sys.stderr)
                    # Fallback to parent implementation
                    fallback_videos, fallback_metadata = super()._decode_and_sample_videos(
                        [video], [metadata], do_sample_frames, sample_indices_fn
                    )
                    processed_videos.extend(fallback_videos)
                    processed_metadata.extend(fallback_metadata)
            else:
                # Use parent implementation for tensors, lists, and other types
                fallback_videos, fallback_metadata = super()._decode_and_sample_videos(
                    [video], [metadata], do_sample_frames, sample_indices_fn
                )
                processed_videos.extend(fallback_videos)
                processed_metadata.extend(fallback_metadata)

        return processed_videos, processed_metadata


def create_enhanced_video_processor_from_original(original_processor) -> EnhancedQwen3VLVideoProcessor:
    """
    Factory function to create enhanced video processor with the same configuration
    as an existing processor.

    Args:
        original_processor: Original Qwen3VLVideoProcessor instance

    Returns:
        EnhancedQwen3VLVideoProcessor: Enhanced processor instance with same config
    """
    print(f"🏭🏭🏭 [DEBUG] 从现有processor创建EnhancedQwen3VLVideoProcessor 🏭🏭🏭", file=sys.stderr)

    # Copy all attributes from original processor
    config_dict = original_processor.to_dict()
    print("original_processor:", original_processor)

    # Remove non-constructor parameters
    config_dict.pop("video_processor_type", None)

    return EnhancedQwen3VLVideoProcessor(**config_dict)


def create_enhanced_video_processor(**kwargs) -> EnhancedQwen3VLVideoProcessor:
    """
    Factory function to create enhanced video processor with custom configuration.

    Args:
        **kwargs: Configuration parameters for the processor

    Returns:
        EnhancedQwen3VLVideoProcessor: Enhanced processor instance
    """
    print(f"🏭🏭🏭 [DEBUG] 创建新的EnhancedQwen3VLVideoProcessor实例 🏭🏭🏭", file=sys.stderr)

    # Set reasonable defaults
    default_config = {
        "fps": 2.0,
        "max_frames": 768,
        "size": {"longest_edge": 32 * 32 * 768, "shortest_edge": 128 * 32 * 32},
        "merge_size": 2,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "do_resize": True,
        "do_rescale": True,
        "do_normalize": True,
        "do_convert_rgb": True,
        "do_sample_frames": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }

    # Update with user-provided config
    default_config.update(kwargs)

    return EnhancedQwen3VLVideoProcessor(**default_config)