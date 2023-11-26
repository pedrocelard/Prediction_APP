# /bin/bash

MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 2 --scale_time_dim 0 --rgb False";
DIFFUSION_FLAGS="--diffusion_steps 200 --noise_schedule linear";
SAMPLE_FLAGS="--seq_len 15 --cond_frames 0,14,";

python partial_video_sample.py --data_dir final_temporal_samples/ --model_path ./log/openai-2023-09-25-11-18-49-633769/ema_0.9999_300000.pt --batch_size 1 $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS