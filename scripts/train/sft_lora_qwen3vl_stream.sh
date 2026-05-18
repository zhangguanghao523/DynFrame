#!/bin/bash

datetime=$(date +"%Y_%m_%d_%H_%M_%S")

output_dir="/mnt/workspace/user/workspace-data/model_logs/lora/qwen3vl4b_lora_${datetime}"

run_name="${output_dir##*/}"

export FORCE_QWENVL_VIDEO_READER="decord"

deepspeed --master_port 22118 src/train_bash.py \
    --model_name_or_path /mnt/workspace/user/model_pretrained/qwen_lab/Qwen3-VL-4B-Thinking \
    --print_param_status=True \
    --stage sft \
    --do_train True \
    --finetuning_type lora \
    --lora_target o_proj,q_proj,k_proj,v_proj \
    --deepspeed scripts/ds_zero2.json \
    --dataset actnet_fps \
    --video_fps 0.5 \
    --video_resolution 640000 \
    --auto_duplicate_videos True \
    --video video \
    --template qwen3_vl \
    --cutoff_len 8192 \
    --streaming \
    --max_steps 32 \
    --accelerator_config=scripts/accelerator.json \
    --overwrite_cache True \
    --output_dir ${output_dir} \
    --logging_steps 1 \
    --save_steps 500 \
    --overwrite_output_dir True \
    --run_name ${run_name} \
    --report_to none \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1.0e-4 \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 True \
    --add_tokens_file scripts/span_fps_tokens.json 
