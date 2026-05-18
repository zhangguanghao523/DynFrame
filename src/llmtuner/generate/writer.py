import copy
import json
import os
from datetime import datetime
from typing import Any, Dict, Iterator, List

from ..extras.logging import get_logger
from .constant import OutputType
from .utils import is_external_cluster


logger = get_logger(__name__)


class FileWriter(object):
    def __init__(self, output):
        output_dir = os.path.dirname(output)
        os.makedirs(output_dir, exist_ok=True)
        if int(os.getenv("WORLD_SIZE", "1")) > 1:
            output += f"_{os.getenv('RANK', '0')}"
            logger.info(f"outputs_file is changed to {output}")

        self.output_file = open(output, "w", encoding="utf-8")

    def sample_iterator(self, samples: Dict[str, List[Any]]) -> Iterator[Dict[str, Any]]:
        # 获取batch数据每列的元素个数
        column_item_cnt = dict(zip(samples.keys(), list(map(len, list(samples.values())))))
        assert len(set(column_item_cnt.values())) == 1, (
            f"The data lengths of each column do not correspond, got {column_item_cnt}"
        )
        for sample in zip(*samples.values()):
            yield dict(zip(samples.keys(), sample))

    def write(self, data, output_data):
        data["samples"]["predict"] = output_data.predict
        if output_data.generated_tokens_with_probs:
            data["samples"]["tokens_probs"] = output_data.generated_tokens_with_probs
        if output_data.audio_path is not None:
            data["samples"]["audio"] = output_data.audio_path
        for sample in self.sample_iterator(data["samples"]):
            # 如果预测为None,就不写出
            if sample["predict"] is not None:
                self.output_file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def close(self):
        self.output_file.close()



