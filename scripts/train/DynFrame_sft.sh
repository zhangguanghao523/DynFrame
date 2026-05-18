#!/bin/bash

datetime=$(date +"%Y_%m_%d_%H_%M_%S")

# DeepSpeed configuration
deepspeed_config='scripts/ds_zero2.json'

# Model path
model_path="/mnt/workspace/user/model_pretrained/qwen_lab/Qwen3-VL-4B-Thinking"

# Output directory
output_dir="/mnt/workspace/user/workspace-data/model_logs/DynFrame_sft_${datetime}"

# Run name for WandB
run_name="${output_dir##*/}"

export FORCE_QWENVL_VIDEO_READER="decord"

# WandB configuration (optional - set to 'none' if not using)
export WANDB_API_KEY="${WANDB_API_KEY:-""}"
export WANDB_PROJECT="vmcot"
export WANDB_BASE_URL="https://api.bandw.top"
export WANDB_MODE=online

# Training script
TRAINING_SCRIPT="src/train_bash.py"

# Training arguments
TRAINING_ARGS="
    --deepspeed=${deepspeed_config} \
    --model_name_or_path=${model_path} \
    --print_param_status=True \
    --stage=sft \
    --do_train=True \
    --finetuning_type=freeze \
    --freeze_module_prefix=model.visual.patch_embed,model.visual.blocks \
    --dataset=actnet_fps \
    --video_fps=2 \
    --video_maxlen=256 \
    --video_resolution=640000 \
    --auto_duplicate_videos=True \
    --video=video \
    --template=qwen3_vl \
    --cutoff_len=81920 \
    --overwrite_cache=True \
    --dataloader_num_workers=8 \
    --streaming \
    --buffer_size=512 \
    --accelerator_config=scripts/accelerator.json \
    --max_steps=4492 \
    --save_steps=100 \
    --ignore_data_skip=True \
    --output_dir=${output_dir} \
    --logging_steps=1 \
    --overwrite_output_dir=True \
    --run_name=${run_name} \
    --report_to=wandb \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=1.0e-5 \
    --lr_scheduler_type=cosine \
    --warmup_ratio=0.05 \
    --bf16=True \
    --gradient_checkpointing=True \
    --ddp_timeout=180000000 \
    --add_tokens_file=scripts/span_fps_tokens.json \
    --log_level=debug
"

# Print configuration
echo "=========================================="
echo "Training Configuration:"
echo "=========================================="
echo "Model Path: ${model_path}"
echo "Output Directory: ${output_dir}"
echo "DeepSpeed Config: ${deepspeed_config}"
echo "=========================================="
echo ""
echo "Training Arguments:"
echo "${TRAINING_ARGS}"
echo "=========================================="

# Run training
deepspeed --master_port=22118 \
            ${TRAINING_SCRIPT} \
            ${TRAINING_ARGS}


echo "=========================================="
echo "Training completed!"
echo "Output saved to: ${output_dir}"
echo "=========================================="