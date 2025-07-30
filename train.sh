accelerate launch --config_file accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path data/dataset_v3 \
  --dataset_metadata_path data/dataset_v3/metadata.csv \
  --data_file_keys "video,vace_video" \
  --height 400 \
  --width 608 \
  --num_frames 77 \
  --model_paths '["models/VACE1.3/diffusion_pytorch_model.safetensors", "models/VACE1.3/models_t5_umt5-xxl-enc-bf16.pth", "models/VACE1.3/Wan2.1_VAE.pth"]' \
  --learning_rate 1e-4 \
  --num_epochs 20 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/Wan2.1-VACE-1.3B_full" \
  --trainable_models "vace" \
  --extra_inputs "vace_video" \
  --gradient_accumulation_steps 5
#  --use_gradient_checkpointing_offload \