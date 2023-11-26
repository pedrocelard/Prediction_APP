
$FULL_DATA_PATH="./dataset/";
$FULL_MODEL_PATH="./log/EMBRYO_FULLLENGTH_MODEL_openai-2023-06-10-11-14-55-761094/ema_0.9999_200000.pt";

$FULL_MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 2 --scale_time_dim 0 --rgb False";
$FULL_DIFFUSION_FLAGS="--diffusion_steps 200 --noise_schedule linear";
$FULL_SAMPLE_FLAGS="--cond_frames 0, --seq_len 15 --batch_size 1 --num_samples 1"; 



# # Construct the command to run the Python script
# $pythonCommand = "python video_sample.py --data_dir $FULL_DATA_PATH --model_path $FULL_MODEL_PATH $FULL_MODEL_FLAGS $FULL_DIFFUSION_FLAGS $FULL_SAMPLE_FLAGS"

# # Execute the Python script
# Invoke-Expression -Command $pythonCommand

$LOG_OVERVIEW="C:\Users\Pedro\Desktop\Work\0_Prediction_Interface\diffusion\log\openai-2023-11-25-12-56-15-060236"
$OUTPUT_SAMPLES="final_samples/sample_"
$FILE_OVERVIEW="1x15x128x128x1.npz"

$pythonCommand = "python npz_to_images.py $LOG_OVERVIEW $FILE_OVERVIEW $OUTPUT_SAMPLES"
Invoke-Expression -Command $pythonCommand