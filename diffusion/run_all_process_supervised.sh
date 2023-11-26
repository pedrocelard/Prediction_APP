# /bin/bash


FULL_DATA_PATH="../../../Datasets/Embryo/EmbryoF1_20frames_128_gif_foldered/test/";
FULL_MODEL_PATH="./log/EMBRYO_FULLLENGTH_MODEL_openai-2023-06-10-11-14-55-761094/ema_0.9999_200000.pt";

FULL_MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 2 --scale_time_dim 0 --rgb False";
FULL_DIFFUSION_FLAGS="--diffusion_steps 200 --noise_schedule linear";
FULL_SAMPLE_FLAGS="--cond_frames 0, --seq_len 15 --batch_size 1 --num_samples 1"; 

s=0

for file in "$FULL_DATA_PATH"*; do
    # 1-Generate 15 frames from start to finish
    echo "Generating Overview samples"
    python video_sample.py --data_dir $file --model_path $FULL_MODEL_PATH --batch_size 1 $FULL_MODEL_FLAGS $FULL_DIFFUSION_FLAGS $FULL_SAMPLE_FLAGS


    # 1b-Change log directory name
    echo "Changing directory name"
    LOG_OVERVIEW="./log/samples"
    OUTPUT_SAMPLES="final_samples/sample_"$s
    most_recent_dir=$(ls ./log -t | head -n 1)
    rm -r $LOG_OVERVIEW
    mv "./log/$most_recent_dir" $LOG_OVERVIEW
    FILE_OVERVIEW=$(find $LOG_OVERVIEW -type f -name "*.npz" -exec basename {} \;)

    # 1c-Generate images and gifs. npz_to_images.py
    #rm -r ./final_samples/*
    python npz_to_images.py $LOG_OVERVIEW $FILE_OVERVIEW $OUTPUT_SAMPLES
    s=$((s+1))
done




# PARTIAL_DATA_PATH="final_samples/";
# PARTIAL_MODEL_PATH="./log/EMBRYO_PARTIAL_MODEL/ema_0.9999_300000.pt";
# 
# PARTIAL_MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 2 --scale_time_dim 0 --rgb False";
# PARTIAL_DIFFUSION_FLAGS="--diffusion_steps 200 --noise_schedule linear";
# PARTIAL_SAMPLE_FLAGS="--batch_size 1 --seq_len 15 --cond_frames 0,14, ";
# 
# 
# LOG_FILL_OVERVIEW="./log/fill_samples"
# 
# for i in {0..2} # 0..68
# do
# rm "$LOG_FILL_OVERVIEW"/*
# 
# 
#     for f in {0..13} #13
#     do
#         echo "Frame group: $i"
#         sample_folder=$PARTIAL_DATA_PATH"sample_"$i
#         f_plus=$((f+1))
#         PARTIAL_GUIDE_FLAGS="--guide_frames $f,$f_plus,";
#         
#         
#         # 2-Generate 15 frames between each fake frame of the FULL samples
#         python partial_video_sample.py --data_dir $sample_folder --model_path $PARTIAL_MODEL_PATH --batch_size 1 $PARTIAL_MODEL_FLAGS $PARTIAL_DIFFUSION_FLAGS $PARTIAL_SAMPLE_FLAGS $PARTIAL_GUIDE_FLAGS
# 
# 
#         # 2b-Change log directory name
#         echo "Moving .npz file"
#         most_recent_dir=$(ls ./log -t | head -n 1)
#         FILE_OVERVIEW=$(find "./log/"$most_recent_dir -type f -name "*.npz")     
#         echo $FILE_OVERVIEW
#         mv $FILE_OVERVIEW $LOG_FILL_OVERVIEW"/"$f".npz"
#         rm -r "./log/"$most_recent_dir
#     done
#   
#   # 2c-Generate images and gifs. npz_to_images.py
#   python npz_to_fill_images.py $LOG_FILL_OVERVIEW $sample_folder
# done
