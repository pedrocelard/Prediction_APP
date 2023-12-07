import argparse

import numpy as np
import torch as th
import torch.distributed as dist
import torchvision.transforms as transforms

import os, sys
sys.path.insert(1, os.getcwd()) 
import random
import json

from PIL import Image
import imageio
from natsort import natsorted
from .diffusion_openai.video_datasets import load_data
from .diffusion_openai import dist_util, logger
from .diffusion_openai.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

class Prediction():
    def __init__(self, cond_frames=0, num_samples=1):
        self.device = "cuda:0"
        FULL_DATA_PATH="--data_dir ./diffusion/dataset/ "
        FULL_MODEL_PATH="--model_path ./diffusion/log/EMBRYO_FULLLENGTH_MODEL_openai-2023-06-10-11-14-55-761094/ema_0.9999_200000.pt "

        FULL_MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 2 --scale_time_dim 0 --rgb False "
        FULL_DIFFUSION_FLAGS="--diffusion_steps 200 --noise_schedule linear "
        FULL_SAMPLE_FLAGS=f"--cond_frames {cond_frames}, --seq_len 15 --batch_size 1 --num_samples {num_samples} "

        self.all_flags = " ".join([FULL_DATA_PATH,
                            FULL_MODEL_PATH,
                            FULL_MODEL_FLAGS,
                            FULL_DIFFUSION_FLAGS,
                            FULL_SAMPLE_FLAGS])
        

    def create_argparser(self):
        defaults = dict(
            clip_denoised=True,
            num_samples=10,
            batch_size=10,
            use_ddim=False,
            model_path="",
            seq_len=15,
            sampling_type="generation",
            cond_frames="0,14",
            cond_generation=True,
            resample_steps=1,
            data_dir='',
            save_gt=False,
            seed = 0
        )
        defaults.update(model_and_diffusion_defaults())
        parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, defaults)
        return parser



    def get_new_output_path(self, parent="./data", element="case"):
        # Check parent to get existing folders
        paths = natsorted(os.listdir(parent))

        if(not paths):
            new_id = 1
        else:
            new_id = int(paths[-1].split('_')[-1])+1

        new_path = os.path.join(parent,f"{element}_{new_id}")

        # Create folder if not exists
        if(not os.path.exists(new_path)): os.makedirs(new_path)

        return new_path

    def preprocess_cond_image(self, frames):
        resolution = 128
        seq_len = 15

        arr_list = []
        for frame in frames:
            while min(*frame.size) >= 2 * resolution:
                frame = frame.resize(
                    tuple(x // 2 for x in frame.size), resample=Image.BOX
                )
            scale = resolution / min(*frame.size)
            frame =frame.resize(
                tuple(round(x * scale) for x in frame.size), resample=Image.BICUBIC
            )

            arr = np.array(frame.convert("L"))
            arr = np.expand_dims(arr, axis=2)
            crop_y = (arr.shape[0] - resolution) // 2
            crop_x = (arr.shape[1] - resolution) // 2
            arr = arr[crop_y : crop_y + resolution, crop_x : crop_x + resolution]
            arr = arr.astype(np.float32) / 127.5 - 1
            arr_list.append(arr)

        arr_seq = np.array(arr_list[:15])
        arr_seq = np.transpose(arr_seq, [3, 0, 1, 2])
        if arr_seq.shape[1] > seq_len:
            start = np.random.randint(0, arr_seq.shape[1]-seq_len)
            arr_seq = arr_seq[:,start:start + seq_len]
        out_dict = {}
        return arr_seq, out_dict


    def generate_prediction(self, cond_images=None):
        
        args_str = self.all_flags.split()
        args = self.create_argparser().parse_args(args_str)

        dist_util.setup_dist()

        # logger.configure()
        # Modified logger.log > print
        if args.seed:
            th.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)

        print("creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu")
        )
        model.to(dist_util.dev())
        model.eval()

        cond_kwargs = {}
        cond_frames = []
        if args.cond_generation:
            if(cond_images is None):
                data = load_data(
                    data_dir=args.data_dir,
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    class_cond=args.class_cond,
                    deterministic=True,
                    rgb=args.rgb,
                    seq_len=args.seq_len
                )
                
            num = ""
            for i in args.cond_frames:
                if i == ",":
                    cond_frames.append(int(num))
                    num = ""
                else:
                    num = num + i
            ref_frames = list(i for i in range(args.seq_len) if i not in cond_frames)
            print(f"cond_frames: {cond_frames}")
            print(f"ref_frames: {ref_frames}")
            print(f"seq_len: {args.seq_len}")
            cond_kwargs["resampling_steps"] = args.resample_steps
        cond_kwargs["cond_frames"] = cond_frames

        if args.rgb:
            channels = 3
        else:
            channels = 1

        print("sampling...")
        all_videos = []
        all_gt = []
        while len(all_videos) * args.batch_size < args.num_samples:
            
            if args.cond_generation:
                if(cond_images is None):
                    video, _ = next(data) # torch.Size([1, 1, 15, 128, 128])
                    cond_kwargs["cond_img"] = video[:,:,cond_frames].to(dist_util.dev()) 
                    video = video.to(dist_util.dev())
                else:
                    # cond_kwargs["cond_img"] = transform(cond_image).unsqueeze(0).to(dist_util.dev())
                    video, _ = self.preprocess_cond_image(cond_images)
                    #cond_kwargs["cond_img"] = video[:,:,cond_frames].to(dist_util.dev())
                    video_tensor = th.from_numpy(video[:,:])
                    cond_kwargs["cond_img"] = video_tensor.to(dist_util.dev())

            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )

            sample = sample_fn(
                model,
                (args.batch_size, channels, args.seq_len, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                progress=False,
                cond_kwargs=cond_kwargs
            )

            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 4, 1)
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_videos.extend([sample.cpu().numpy() for sample in gathered_samples])
            print(f"created {len(all_videos) * args.batch_size} samples")

            if args.cond_generation and args.save_gt:

                video = ((video + 1) * 127.5).clamp(0, 255).to(th.uint8)
                video = video.permute(0, 2, 3, 4, 1)
                video = video.contiguous()

                gathered_videos = [th.zeros_like(video) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_videos, video)  # gather not supported with NCCL
                all_gt.extend([video.cpu().numpy() for video in gathered_videos])
                print(f"created {len(all_gt) * args.batch_size} videos")


        arr = np.concatenate(all_videos, axis=0) # Shape  (1, 15, 128, 128, 1)
        # Define the desired minimum and maximum values for the range
        new_min = 0  # New minimum value
        new_max = 255  # New maximum value
    
        # save the generated data in this folder
        case_path = self.get_new_output_path(parent="./data", element="case")
        overviews_info = []

        for sample_arr in arr:

            overview_path = self.get_new_output_path(parent=case_path, element="overview")
            images = []
            imgs_names_list = []
            
            for index, image in enumerate(sample_arr):
                
                # Create a PIL Image from the array
                image_array = np.squeeze(image, axis=2)
                
                # Convert the image to a NumPy array
                #image_array = np.array(image)

                # Find the current minimum and maximum values in the image
                current_min = np.min(image_array)
                current_max = np.max(image_array)

                # Perform the rescaling to the new range
                scaled_image_array = ((image_array - current_min) / (current_max - current_min)) * (new_max - new_min) + new_min

                # Convert the rescaled NumPy array back to a PIL image
                image = Image.fromarray(scaled_image_array.astype(np.uint8))
                
                images.append(image)
                # Save the image
                img_name = f"{index}.jpg"
                real = True if index in cond_frames else False
                imgs_names_list.append([real,img_name])
                image.save(f'{overview_path}/{img_name}')

            # imageio.mimsave(os.path.join(sample_dir_gif,"embryo_"+str(i)+".gif"), images)
            overviews_info.append({"overview_id" : os.path.basename(overview_path),
                                    "score": None,
                                    "mse": None,
                                    "mag": None,
                                    "imgs" :  imgs_names_list
                                    })

        # Create new case json info
        new_case_dict = {"overview_maual_ranking" : False,
                "overview_ranking": None,
                "overview_info": overviews_info,
                "timeline_info": None
                }

        with open(os.path.join(case_path,"case_info.json"), "w") as file:
            json.dump(new_case_dict, file, indent=4)

        if args.cond_generation and args.save_gt:
            arr_gt = np.concatenate(all_gt, axis=0)


        # if dist.get_rank() == 0:

            # shape_str = "x".join([str(x) for x in arr.shape])
            # print(f"saving samples to {os.path.join(logger.get_dir(), shape_str)}")
            # np.savez(os.path.join(logger.get_dir(), shape_str), arr)

            # if args.cond_generation and args.save_gt:
            #     shape_str_gt = "x".join([str(x) for x in arr_gt.shape])
            #     print(f"saving ground_truth to {os.path.join(logger.get_dir(), shape_str_gt)}")
            #     np.savez(os.path.join(logger.get_dir(), shape_str_gt), arr_gt)

        dist.barrier()
        print("sampling complete")
        return case_path
