CORES_PER_GPU=$(($SLURM_CPUS_PER_TASK / 2))
export OMP_NUM_THREADS=$CORES_PER_GPU

accelerate launch --config_file accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path data/dataset_v0/train \
  --dataset_metadata_path data/dataset_v0/train/metadata.csv \
  --data_file_keys "video,vace_video" \
  --height 400 \
  --width 608 \
  --num_frames 77 \
  --model_paths '["models/VACE1.3/diffusion_pytorch_model.safetensors", "models/VACE1.3/models_t5_umt5-xxl-enc-bf16.pth", "models/VACE1.3/Wan2.1_VAE.pth"]' \
  --learning_rate 1e-4 \
  --num_epochs 200 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/run01" \
  --trainable_models "vace" \
  --extra_inputs "vace_video" \
  --gradient_accumulation_steps 4 \
  --use_wandb \
  --wandb_project "vace-training" \
  --wandb_run_name "test-run" \
  --wandb_entity "" \
  --save_every_n_epochs 1 \
  --validate_every_n_epochs 1 \
  --validation_dataset_base_path "data/dataset_v0/val" \
  --validation_dataset_metadata_path "data/dataset_v0/val/metadata.csv" \
  --validation_prompt "A light moving around an object." \
  --validation_num_frames 77 \
  --validation_height 400 \
  --validation_width 608 \
  --validation_seed 42
#  --use_gradient_checkpointing_offload \