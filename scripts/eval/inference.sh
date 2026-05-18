#!/bin/bash

echo "开始推理..."

output_dir="/mnt/workspace/user/workspace-data/model_logs/"

mkdir -p ${output_dir}

FORCE_QWENVL_VIDEO_READER=decord python src/generate.py \
    --model_name_or_path /mnt/workspace/user/workspace-data/model_logs/ \
    --template qwen3_vl \
    --load_from file \
    --inputs /mnt/workspace/user/dataset/mlvu_transformed_qwen3vl.json \
    --outputs /mnt/workspace/user/workspace-data/model_logs/test_log_token.json \
    --prompt_column instruction \
    --video video \
    --batch_size 1 \
    --max_new_tokens 20480 \
    --do_sample=false \
    --video_maxlen=256 \
    --seed=42 \
    --top_p=0.8 \
    --top_k=100 \
    --infer_mode=default \
    --add_tokens_file scripts/span_fps_tokens.json \
    --enable_dynamic_video true \
    --video_fps 2 \
    2>&1 | tee ${output_dir}/inference.log

echo "推理完成！"
