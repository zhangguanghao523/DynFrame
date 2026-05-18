#!/bin/bash

datetime=$(date +"%Y_%m_%d_%H_%M_%S")

# 设置视频读取器环境变量
export FORCE_QWENVL_VIDEO_READER="decord"

output_dir="/mnt/workspace/user/workspace-data/model_logs/qwen3vl4b_full_${datetime}"

run_name="${output_dir##*/}"


deepspeed --master_port 22115 src/train_bash.py \
    --model_name_or_path /mnt/workspace/user/model_pretrained/qwen_lab/Qwen3-VL-4B-Thinking \
    --print_param_status True \
    --stage sft \
    --do_train True \
    --finetuning_type=freeze \
    --freeze_module_prefix=model.visual.patch_embed,model.visual.blocks \
    --deepspeed scripts/ds_zero2.json \
    --dataset actnet_fps \
    --video_fps 0.5 \
    --video_resolution 640000 \
    --video video \
    --template qwen3_vl \
    --cutoff_len 4096 \
    --max_samples 512 \
    --overwrite_cache True \
    --dataloader_num_workers 8 \
    --output_dir ${output_dir} \
    --logging_steps 1 \
    --save_steps 100 \
    --overwrite_output_dir True \
    --run_name ${run_name} \
    --report_to none \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1.0e-5 \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 True \
    --add_tokens_file scripts/span_fps_tokens.json \
