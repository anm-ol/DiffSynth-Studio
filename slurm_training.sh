#!/bin/bash

#SBATCH --job-name=vace_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --partition=a100
#SBATCH --gres=gpu:A100:2
#SBATCH --time=24:00:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err

# Load necessary modules or activate environment if needed
# module load <module_name>
source /home/venky/ankitd/miniconda3/bin/activate diffsynth

# Set wandb environment variables (more secure than putting API key in script)
# Make sure WANDB_API_KEY is set in your environment or ~/.bashrc
export WANDB_API_KEY="b2da3eefc9d663802fcfa3be2b75d6469fd71d78"

# Set wandb mode - can be "online", "offline", or "disabled"
export WANDB_MODE=online

# Optional: Set wandb cache directory to avoid filling up home directory
#export WANDB_CACHE_DIR=/tmp/wandb_cache_$SLURM_JOB_ID

# Change to the correct directory
cd /mnt/venky/ankitd/anmol/new_vace_training/DiffSynth-Studio

# Execute the command
#bash train.sh
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
  --output_path "./models/train/run03" \
  --trainable_models "vace" \
  --extra_inputs "vace_video" \
  --gradient_accumulation_steps 16 \
  --use_wandb \
  --wandb_project "vace-training" \
  --wandb_run_name "vace-1.3b-full-training3" \
  --wandb_entity "" \
  --save_every_n_epochs 5 \
  --validate_every_n_epochs 5 \
  --validation_dataset_base_path "data/dataset_v0/val" \
  --validation_dataset_metadata_path "data/dataset_v0/val/metadata.csv" \
  --validation_prompt "A light moving around an object." \
  --validation_num_frames 77 \
  --validation_height 400 \
  --validation_width 608 \
  --validation_seed 42