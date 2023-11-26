# /bin/bash

MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 2 --scale_time_dim 0";
DIFFUSION_FLAGS="--diffusion_steps 200 --noise_schedule linear";
TRAIN_FLAGS="--lr 2e-5 --batch_size 1 --microbatch 1 --seq_len 15 --max_num_mask_frames 4 --uncondition_rate 0.75 --save_interval 50000 --rgb False --num_heads 2";

python video_train.py --data_dir ../../../Datasets/Embryo/EmbryoF1_FULLframes_128_gif/train/ $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS