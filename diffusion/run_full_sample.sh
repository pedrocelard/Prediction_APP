# /bin/bash

MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 2 --scale_time_dim 0 --rgb False";
DIFFUSION_FLAGS="--diffusion_steps 200 --noise_schedule linear";
SAMPLE_FLAGS="--cond_frames 0 --seq_len 15 --cond_frames 0";

python video_sample.py --data_dir final_samples_embryo/ --model_path ./log/EMBRYO_FULLLENGTH_MODEL_openai-2023-06-10-11-14-55-761094/ema_0.9999_200000.pt --batch_size 1 $MODEL_FLAGS $DIFFUSION_FLAGS
