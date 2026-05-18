"""
Enhanced Video Token Parser

支持解析两种视频token格式：
1. 普通格式: <video>
2. 时间段格式: <span>start_time - end_time</span><fps>fps_value</fps><video>

保留原始文本不变，只提取视频token的元数据信息。
"""

import re
import sys
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any


@dataclass
class VideoTokenInfo:
    """视频token信息"""
    video_index: int           # 在文本中的实际出现顺序（0, 1, 2, ...）
    token_type: str           # 'simple' 或 'segment'
    start_pos: int            # 在文本中的起始位置
    end_pos: int             # 在文本中的结束位置

    # 时间段视频专有字段
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    fps: Optional[float] = None


def extract_all_video_tokens_info(text: str) -> Tuple[str, List[VideoTokenInfo]]:
    """
    提取文本中所有视频token的信息，按实际出现顺序编号

    Args:
        text: 包含视频token的文本

    Returns:
        tuple: (原始文本, 按出现顺序排列的视频token信息列表)
    """
    video_tokens = []

    # 模式1: 时间段视频 <span>start - end</span><fps>fps_value</fps><video>
    segment_pattern = r'<span>([\d.]+)\s*-\s*([\d.]+)</span><fps>([\d.]+)</fps><video>'

    # 模式2: 普通视频 <video> (但不能是时间段视频的一部分)
    # 使用负向后查找，确保<video>前面没有</fps>
    simple_pattern = r'(?<!fps>)<video>'

    # 收集所有匹配
    all_matches = []

    # 找到所有时间段视频
    for match in re.finditer(segment_pattern, text):
        all_matches.append({
            'type': 'segment',
            'start_pos': match.start(),
            'end_pos': match.end(),
            'match_obj': match
        })

    # 找到所有普通视频
    for match in re.finditer(simple_pattern, text):
        # 确保这个<video>不是时间段视频的一部分
        is_part_of_segment = False
        for segment_match in re.finditer(segment_pattern, text):
            if segment_match.start() <= match.start() <= segment_match.end():
                is_part_of_segment = True
                break

        if not is_part_of_segment:
            all_matches.append({
                'type': 'simple',
                'start_pos': match.start(),
                'end_pos': match.end(),
                'match_obj': match
            })

    # 按在文本中的出现位置排序
    all_matches.sort(key=lambda x: x['start_pos'])

    # 构建VideoTokenInfo列表
    for video_index, match_info in enumerate(all_matches):
        if match_info['type'] == 'segment':
            match = match_info['match_obj']
            video_tokens.append(VideoTokenInfo(
                video_index=video_index,
                token_type='segment',
                start_pos=match_info['start_pos'],
                end_pos=match_info['end_pos'],
                start_time=float(match.group(1)),
                end_time=float(match.group(2)),
                fps=float(match.group(3))
            ))
        else:  # simple
            video_tokens.append(VideoTokenInfo(
                video_index=video_index,
                token_type='simple',
                start_pos=match_info['start_pos'],
                end_pos=match_info['end_pos']
            ))

    return text, video_tokens  # 保留原始文本


def build_video_config_map(conversations: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    构建视频配置映射表，按照视频在整个对话中的出现顺序编号

    Args:
        conversations: 对话列表

    Returns:
        Dict[global_video_index, video_config]: 全局视频索引到配置的映射
    """
    video_config_map = {}
    global_video_index = 0

    for conv in conversations:
        if conv.get('from') in ['human', 'gpt'] and 'value' in conv:
            text = conv['value']

            # 提取该段对话中的所有视频token信息
            _, video_tokens = extract_all_video_tokens_info(text)

            # 为每个视频token创建配置
            for token_info in video_tokens:
                if token_info.token_type == 'segment':
                    # 时间段视频配置 - 🎯 修复：使用string key
                    video_config_map[str(global_video_index)] = {
                        'video_start': token_info.start_time,
                        'video_end': token_info.end_time,
                        'fps': token_info.fps,
                        'type': 'segment'
                    }
                    # print(f"🎯 [DEBUG] 全局视频索引{global_video_index}: 时间段{token_info.start_time}-{token_info.end_time}s @ {token_info.fps}fps", file=sys.stderr)
                else:
                    # 普通视频配置 - 🎯 修复：使用string key
                    video_config_map[str(global_video_index)] = {
                        'type': 'simple'
                    }
                    # print(f"🎯 [DEBUG] 全局视频索引{global_video_index}: 普通视频", file=sys.stderr)

                global_video_index += 1

    return video_config_map


def test_video_token_parsing():
    """测试视频token解析功能"""

    print("🧪 开始测试视频token解析功能...")

    # 测试用例1: 混合的视频token
    test_text1 = """First, let's look at the overall video: <video>

Now focus on this specific segment: <span>6.50 - 10.50</span><fps>2</fps><video>

And another normal video: <video>

Finally, another segment: <span>12.0 - 15.0</span><fps>1.5</fps><video>"""

    print("\n🧪 测试用例1 - 混合视频token:")
    _, tokens = extract_all_video_tokens_info(test_text1)
    for i, token in enumerate(tokens):
        if token.token_type == 'segment':
            print(f"  视频{i}: 时间段 {token.start_time}-{token.end_time}s @ {token.fps}fps (位置: {token.start_pos}-{token.end_pos})")
        else:
            print(f"  视频{i}: 普通视频 (位置: {token.start_pos}-{token.end_pos})")

    # 测试用例2: 只有普通视频
    test_text2 = "Look at <video> and then <video>"
    print("\n🧪 测试用例2 - 只有普通视频:")
    _, tokens = extract_all_video_tokens_info(test_text2)
    for i, token in enumerate(tokens):
        print(f"  视频{i}: {token.token_type} (位置: {token.start_pos}-{token.end_pos})")

    # 测试用例3: 只有时间段视频
    test_text3 = "<span>1.0 - 5.0</span><fps>3</fps><video> and <span>8.0 - 12.0</span><fps>1</fps><video>"
    print("\n🧪 测试用例3 - 只有时间段视频:")
    _, tokens = extract_all_video_tokens_info(test_text3)
    for i, token in enumerate(tokens):
        if token.token_type == 'segment':
            print(f"  视频{i}: 时间段 {token.start_time}-{token.end_time}s @ {token.fps}fps (位置: {token.start_pos}-{token.end_pos})")

    # 测试用例4: 实际数据格式
    real_data_text = """<think>Let me look at the sequence of images. The man is riding an ATV through a muddy, watery area. <span>6.50 - 10.50</span><fps>2</fps><video> Around 6.5 seconds, the ATV appears to be stuck, as the wheels are spinning and the vehicle is not making much forward progress. The man continues to operate the ATV, trying to get it unstuck by accelerating and maneuvering, but he does not stop, touch his head, point, prepare to swing, or look at the floor in a way that suggests giving up or changing tactics. He is persistently trying to get the ATV moving. This matches option (C) "kept trying".</think>
<answer>C</answer>"""

    print("\n🧪 测试用例4 - 真实数据格式:")
    _, tokens = extract_all_video_tokens_info(real_data_text)
    for i, token in enumerate(tokens):
        if token.token_type == 'segment':
            print(f"  视频{i}: 时间段 {token.start_time}-{token.end_time}s @ {token.fps}fps")

    # 测试用例5: 测试build_video_config_map
    test_conversations = [
        {
            'from': 'human',
            'value': 'Based on the content of the video, answer the following question: <video>'
        },
        {
            'from': 'gpt',
            'value': real_data_text
        }
    ]

    print("\n🧪 测试用例5 - 构建视频配置映射:")
    config_map = build_video_config_map(test_conversations)
    for idx, config in config_map.items():
        print(f"  全局索引{idx}: {config}")

    print("\n✅ 视频token解析功能测试完成!")


if __name__ == "__main__":
    test_video_token_parsing()