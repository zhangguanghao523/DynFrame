#!/bin/bash
# GRPO training script for Qwen3VL with 4-dimensional rewards
# Rewards: format, accuracy, fps, time_iou
# Local training using DeepSpeed ZeRO Stage 2

datetime=$(date +"%Y_%m_%d_%H_%M_%S")

# DeepSpeed configuration - ZeRO Stage 2 (more stable for generation)
deepspeed_config='scripts/ds_zero2.json'

# Model path - using the VMCoT checkpoint with dynamic FPS
model_path="/mnt/workspace/user/workspace-data/model_logs/dynamic_fps_qwen3vl4bthink"

# Output directory
output_dir="/mnt/workspace/user/workspace-data/model_logs/grpo/qwen3vl4b_grpo_4reward_${datetime}"

run_name="${output_dir##*/}"


export FORCE_QWENVL_VIDEO_READER="decord"

# WandB configuration (optional - set to 'none' if not using)
export WANDB_API_KEY="${WANDB_API_KEY:-""}"
export WANDB_PROJECT="vmcot"
export WANDB_BASE_URL="https://api.bandw.top"
export WANDB_MODE=online



# Run training with DeepSpeed
# Using master_port to avoid conflicts if multiple trainings are running
deepspeed --master_port 22120 src/train_bash.py \
    --deepspeed=${deepspeed_config} \
    --model_name_or_path=${model_path} \
    --stage=grpo \
    --do_train=True \
    --finetuning_type=freeze \
    --freeze_module_prefix=model.visual.patch_embed,model.visual.blocks \
    --dataset=charades_fps_grpo \
    --template=qwen3_vl \
    --cutoff_len=81920 \
    --enable_dynamic_video=True \
    --overwrite_cache=True \
    --dataloader_num_workers=8 \
    --video_fps=2 \
    --repetition_penalty=1.05 \
    --video_resolution=640000 \
    --video=video \
    --grpo_num_generations=8 \
    --grpo_num_iterations=1 \
    --grpo_format_weight=0.1 \
    --grpo_accuracy_weight=0.6 \
    --grpo_fps_weight=0.3 \
    --grpo_loss_type=grpo \
    --multi_objective_aggregation=normalize_then_sum \
    --grpo_beta=0.02 \
    --grpo_epsilon=0.3 \
    --grpo_scale_rewards=group \
    --temperature=0.7 \
    --top_p=0.8 \
    --top_k=100 \
    --max_new_tokens=20480 \
    --learning_rate=1e-5 \
    --max_grad_norm=5.0 \
    --lr_scheduler_type=cosine \
    --warmup_ratio=0.05 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --max_steps=2000 \
    --save_steps=20 \
    --logging_steps=1 \
    --output_dir=${output_dir} \
    --overwrite_output_dir=True \
    --run_name=${run_name} \
    --report_to=wandb \
    --bf16=True \
    --gradient_checkpointing=True \
    --ddp_timeout=180000000

echo ""
echo "Training completed. Results saved to: ${output_dir}"