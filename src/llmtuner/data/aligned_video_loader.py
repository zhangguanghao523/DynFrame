"""
对齐版视频加载器
提供与transformers video_utils完全一致的输出格式
基于qwen_vl_utils实现但输出格式与transformers标准一致

transformers标准输出格式：
- decord: numpy array [num_frames, height, width, 3], dtype=uint8, RGB
- torchvision: torch tensor [num_frames, 3, height, width], TCHW格式
- torchcodec: torch tensor [num_frames, height, width, 3], THWC格式
"""

import sys
import os
import time
import warnings
from typing import Union, List, Optional, Dict, Any, Callable
import numpy as np
import torch
from transformers.video_utils import VideoMetadata
from transformers.utils import is_decord_available, is_torchvision_available, is_torchcodec_available

# 导入qwen_vl_utils的后端实现
from .third_party_utils.qwen_vl_utils import (
    _read_video_decord,
    _read_video_torchvision,
    _read_video_torchcodec,
    VIDEO_READER_BACKENDS
)


def read_video_decord_aligned(video_config: Dict[str, Any]) -> tuple[np.ndarray, VideoMetadata]:
    """
    调用qwen_vl_utils的decord实现，输出格式对齐transformers

    Returns:
        tuple[np.ndarray, VideoMetadata]:
            - numpy array [num_frames, height, width, 3], dtype=uint8, RGB格式
            - VideoMetadata对象
    """
    # print(f"🎬📹🎬 [DEBUG] read_video_decord_aligned()被调用! 🎬📹🎬", file=sys.stderr)

    # 调用qwen_vl_utils的decord实现
    video_tensor, video_metadata, sample_fps = _read_video_decord(video_config)

    # qwen_vl_utils输出: torch.Tensor [T, C, H, W], 需要转换为 numpy [T, H, W, C]
    if isinstance(video_tensor, torch.Tensor):
        video_numpy = video_tensor.permute(0, 2, 3, 1).numpy()  # [T, C, H, W] -> [T, H, W, C]
    else:
        video_numpy = video_tensor
        if video_numpy.ndim == 4 and video_numpy.shape[1] == 3:
            video_numpy = np.transpose(video_numpy, (0, 2, 3, 1))  # [T, C, H, W] -> [T, H, W, C]

    # 确保数据类型为uint8
    if video_numpy.dtype != np.uint8:
        if video_numpy.max() <= 1.0:
            video_numpy = (video_numpy * 255).astype(np.uint8)
        else:
            video_numpy = video_numpy.astype(np.uint8)

    # 转换为transformers VideoMetadata格式
    metadata = VideoMetadata(
        total_num_frames=video_metadata.get("total_num_frames", video_numpy.shape[0]),
        fps=video_metadata.get("fps", sample_fps),
        width=video_numpy.shape[2],
        height=video_numpy.shape[1],
        video_backend="decord",
        frames_indices=video_metadata.get("frames_indices", list(range(video_numpy.shape[0]))),
    )

    # print(f"✅✅✅ [DEBUG] decord对齐输出: shape={video_numpy.shape}, dtype={video_numpy.dtype} ✅✅✅", file=sys.stderr)
    return video_numpy, metadata


def read_video_torchvision_aligned(video_config: Dict[str, Any]) -> tuple[torch.Tensor, VideoMetadata]:
    """
    调用qwen_vl_utils的torchvision实现，输出格式对齐transformers

    Returns:
        tuple[torch.Tensor, VideoMetadata]:
            - torch tensor [num_frames, 3, height, width], TCHW格式
            - VideoMetadata对象
    """
    # print(f"📺📺📺 [DEBUG] read_video_torchvision_aligned()被调用! 📺📺📺", file=sys.stderr)

    # 调用qwen_vl_utils的torchvision实现
    video_tensor, video_metadata, sample_fps = _read_video_torchvision(video_config)

    # qwen_vl_utils输出: torch.Tensor [T, C, H, W]
    # transformers的torchvision后端也输出 [T, C, H, W]，所以不需要转换格式！
    # 保持原有的TCHW格式

    # 转换为transformers VideoMetadata格式
    metadata = VideoMetadata(
        total_num_frames=video_metadata.get("total_num_frames", video_tensor.shape[0]),
        fps=video_metadata.get("fps", sample_fps),
        width=video_tensor.shape[3],   # TCHW格式中W在位置3
        height=video_tensor.shape[2],  # TCHW格式中H在位置2
        video_backend="torchvision",
        frames_indices=video_metadata.get("frames_indices", list(range(video_tensor.shape[0]))),
    )

    # print(f"✅✅✅ [DEBUG] torchvision对齐输出: shape={video_tensor.shape}, dtype={video_tensor.dtype} ✅✅✅", file=sys.stderr)
    return video_tensor, metadata


def read_video_torchcodec_aligned(video_config: Dict[str, Any]) -> tuple[torch.Tensor, VideoMetadata]:
    """
    调用qwen_vl_utils的torchcodec实现，输出格式对齐transformers

    Returns:
        tuple[torch.Tensor, VideoMetadata]:
            - torch tensor [num_frames, height, width, 3], THWC格式
            - VideoMetadata对象
    """
    # print(f"🎞️🎞️🎞️ [DEBUG] read_video_torchcodec_aligned()被调用! 🎞️🎞️🎞️", file=sys.stderr)

    # 调用qwen_vl_utils的torchcodec实现
    video_tensor, video_metadata, sample_fps = _read_video_torchcodec(video_config)

    # qwen_vl_utils输出: torch.Tensor [T, C, H, W], 需要转换为 [T, H, W, C]
    if video_tensor.dim() == 4 and video_tensor.shape[1] == 3:
        video_tensor = video_tensor.permute(0, 2, 3, 1)  # [T, C, H, W] -> [T, H, W, C]

    # 转换为transformers VideoMetadata格式
    metadata = VideoMetadata(
        total_num_frames=video_metadata.get("total_num_frames", video_tensor.shape[0]),
        fps=video_metadata.get("fps", sample_fps),
        width=video_tensor.shape[2],
        height=video_tensor.shape[1],
        video_backend="torchcodec",
        frames_indices=video_metadata.get("frames_indices", list(range(video_tensor.shape[0]))),
    )

    # print(f"✅✅✅ [DEBUG] torchcodec对齐输出: shape={video_tensor.shape}, dtype={video_tensor.dtype} ✅✅✅", file=sys.stderr)
    return video_tensor, metadata


# 对齐版后端字典
ALIGNED_VIDEO_BACKENDS = {
    "decord": read_video_decord_aligned,
    "torchvision": read_video_torchvision_aligned,
    "torchcodec": read_video_torchcodec_aligned,
}


def get_aligned_video_backend() -> str:
    """获取对齐版视频后端"""
    FORCE_BACKEND = os.getenv("FORCE_QWENVL_VIDEO_READER", None)

    # print(f"🎯🎯🎯 [DEBUG] get_aligned_video_backend()被调用! FORCE_BACKEND={FORCE_BACKEND} 🎯🎯🎯", file=sys.stderr)

    if FORCE_BACKEND is not None and FORCE_BACKEND in ALIGNED_VIDEO_BACKENDS:
        backend = FORCE_BACKEND
        # print(f"🔧🔧🔧 [DEBUG] 使用强制指定的对齐后端: {backend} 🔧🔧🔧", file=sys.stderr)
    elif is_torchcodec_available():
        backend = "torchcodec"
        # print(f"🎞️🎞️🎞️ [DEBUG] 自动选择torchcodec后端 🎞️🎞️🎞️", file=sys.stderr)
    elif is_decord_available():
        backend = "decord"
        # print(f"🎬🎬🎬 [DEBUG] 自动选择decord后端 🎬🎬🎬", file=sys.stderr)
    elif is_torchvision_available():
        backend = "torchvision"
        # print(f"📺📺📺 [DEBUG] 回退到torchvision后端 📺📺📺", file=sys.stderr)
    else:
        raise RuntimeError("No video backend available. Please install at least one of: torchcodec, decord, torchvision")

    # print(f"✅✅✅ [DEBUG] get_aligned_video_backend()返回: {backend} ✅✅✅", file=sys.stderr)
    return backend


def load_video_aligned(
    video_path: str,
    fps: Optional[float] = None,
    backend: Optional[str] = None,
    **kwargs,
) -> tuple[Union[np.ndarray, torch.Tensor], VideoMetadata]:
    """
    对齐版load_video，直接调用qwen_vl_utils后端并转换输出格式

    Args:
        video_path: 视频路径
        num_frames: 帧数
        fps: 帧率
        backend: 后端名称，如果不指定则自动选择

    Returns:
        tuple[Union[np.ndarray, torch.Tensor], VideoMetadata]:
            - decord返回numpy array [T,H,W,C]
            - torchvision/torchcodec返回torch tensor [T,H,W,C]
            - VideoMetadata对象
    """
    # print(f"🎥🎥🎥 [DEBUG] load_video_aligned()被调用! 视频路径: {video_path} 🎥🎥🎥", file=sys.stderr)

    # 选择后端
    if backend is None:
        backend = get_aligned_video_backend()
    elif backend not in ALIGNED_VIDEO_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}. Available backends: {list(ALIGNED_VIDEO_BACKENDS.keys())}")

    # 构建video_config
    video_config = {
        "video": video_path,
        "fps": fps
    }

    # 添加其他参数
    video_config.update(kwargs)

    # print(f"📊📊📊 [DEBUG] 使用后端 {backend} 处理视频配置: {video_config} 📊📊📊", file=sys.stderr)

    # 调用对应的对齐版后端
    aligned_backend_fn = ALIGNED_VIDEO_BACKENDS[backend]
    video, metadata = aligned_backend_fn(video_config)

    # print(f"✅✅✅ [DEBUG] load_video_aligned()完成! 后端={backend}, 输出shape={video.shape} ✅✅✅", file=sys.stderr)
    return video, metadata



def load_video_aligned_with_segment_config(
    video_path: str,
    video_start: float,
    video_end: float,
    fps: float,
    backend: Optional[str] = None,
    max_segment_frames: int = 256,
    **kwargs,
) -> tuple[Union[np.ndarray, torch.Tensor], VideoMetadata]:
    """
    使用时间段配置加载视频，直接利用qwen_vl_utils的现有功能

    Args:
        video_path: 视频路径
        video_start: 开始时间（秒）
        video_end: 结束时间（秒）
        fps: 采样帧率
        backend: 后端名称，如果不指定则自动选择
        max_segment_frames: segment视频的最大帧数限制，默认256

    Returns:
        tuple[Union[np.ndarray, torch.Tensor], VideoMetadata]:
            视频数据和元数据
    """
    # 计算预期帧数，如果超过max_segment_frames则降低fps
    span_duration = video_end - video_start
    expected_frames = fps * span_duration

    if expected_frames > max_segment_frames and span_duration > 0:
        # 自动降低fps以控制帧数
        adjusted_fps = max_segment_frames / span_duration
        print(f"⚠️⚠️⚠️ [WARNING] segment帧数过多: fps={fps} × span={span_duration:.1f}s = {expected_frames:.0f}帧 > {max_segment_frames}, 自动降低fps: {fps} -> {adjusted_fps:.2f} ⚠️⚠️⚠️", file=sys.stderr)
        fps = adjusted_fps

    # 选择后端
    if backend is None:
        backend = get_aligned_video_backend()
    elif backend not in ALIGNED_VIDEO_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}. Available backends: {list(ALIGNED_VIDEO_BACKENDS.keys())}")

    # 构建包含时间段信息的video_config - 直接使用qwen_vl_utils的现有功能！
    video_config = {
        "video": video_path,
        "video_start": video_start,  # qwen_vl_utils已经支持这个参数！
        "video_end": video_end,      # qwen_vl_utils已经支持这个参数！
        "fps": fps
    }

    # 添加其他参数
    video_config.update(kwargs)

    # print(f"📊📊📊 [DEBUG] 使用后端 {backend} 处理时间段视频配置: {video_config} 📊📊📊", file=sys.stderr)

    # 直接调用现有的对齐版后端 - 零额外开销！
    aligned_backend_fn = ALIGNED_VIDEO_BACKENDS[backend]
    video, metadata = aligned_backend_fn(video_config)

    # print(f"✅✅✅ [DEBUG] load_video_aligned_with_segment_config()完成! 后端={backend}, 输出shape={video.shape} ✅✅✅", file=sys.stderr)
    return video, metadata


def test_alignment():
    """测试对齐版加载器与transformers的兼容性"""

    print("🧪 测试对齐版视频加载器...")

    test_video_path = "/mnt/workspace/user/code/LLaMA-Factory/2976913210.mp4"

    try:
        # 测试单个视频加载
        video, metadata = fetch_video_aligned(test_video_path, fps=2.0)

        print(f"✅ 对齐版加载成功:")
        print(f"  - shape: {video.shape}")
        print(f"  - dtype: {video.dtype}")
        print(f"  - pixel range: [{video.min()}, {video.max()}]")
        print(f"  - metadata: fps={metadata.fps}, total_frames={metadata.total_num_frames}")

        # 与原版transformers对比
        from transformers.video_utils import load_video

        try:
            # 使用decord后端，与对齐版实现保持一致
            original_video, original_metadata = load_video(test_video_path, backend="decord", fps=2.0)

            print(f"📊 原版transformers (decord后端):")
            print(f"  - shape: {original_video.shape}")
            print(f"  - dtype: {original_video.dtype}")
            print(f"  - pixel range: [{original_video.min()}, {original_video.max()}]")

            # 检查格式一致性
            shape_match = video.shape == original_video.shape
            dtype_match = video.dtype == original_video.dtype

            print(f"🔍 格式对比:")
            print(f"  - shape匹配: {shape_match}")
            print(f"  - dtype匹配: {dtype_match}")

            if shape_match and dtype_match:
                print("🎉 对齐版与transformers格式完全一致!")
            else:
                print("⚠️  存在格式差异，需要调整")

        except Exception as e:
            print(f"⚠️  原版transformers加载失败: {e}")

    except Exception as e:
        print(f"❌ 对齐版加载器测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_alignment()