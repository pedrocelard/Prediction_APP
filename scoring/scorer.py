import os
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms as transforms
import cv2
import json


from diffusion.utils.biodev_DASA import DASA_metric
from diffusion.utils.biodev_dis import BioDevDisc
from natsort import natsorted

from PIL import ImageFile
from PIL import Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.cuda.set_device(0)

class Scorer():
    def __init__(self):
        self.img_size = 128
        self.embryo_classes = ('empty','t2', 't3', 't4', 't5', 't6',
                                't7', 't8', 't9+', 'tB', 'tEB',
                                'tHB', 'tM', 'tPB2', 'tPNa', 'tPNf', 'tSB')
        #Load classifier
        self.bio_dis = BioDevDisc(17, 'cuda:0')
        checkpoint = 'ResNet18_EMBRYO_128_17.pth'
        checkpoint_path = os.path.join("./diffusion/class_models_and_distributions",checkpoint)
        self.bio_dis.load_checkpoint(checkpoint_path)

        self.dasa = DASA_metric()
        self.dasa.load_distribution("./diffusion//class_models_and_distributions/embryo_128_17.txt")


    def bg_subs(self, image1_path, image2_path, num_sectors):
        # Load the two frames
        frame1 = cv2.imread(image1_path)
        frame2 = cv2.imread(image2_path)

        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Apply optical flow to calculate motion vectors
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 60, 3, 5, 1.2, 0)

        # Threshold or process the motion vectors to detect significant movement

        # Example: thresholding based on magnitude
        magnitude = cv2.magnitude(flow[...,0], flow[...,1])
        threshold = 2.0  # Adjust as needed
        significant_movement = magnitude > threshold

        if(significant_movement.any()):
            return True, magnitude.max()
        else:
            return False, magnitude.max()
        
        
    def movement_check(self, image1_path, image2_path, num_sectors):

        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)
        threshold = 80
        
        # Calculate sector dimensions based on the number of sectors
        height, width, _ = image1.shape
        sector_height = height // num_sectors
        sector_width = width // num_sectors
        
        
        mse_values = []
        # Iterate through sectors and calculate the MSE for each
        for i in range(num_sectors):
            for j in range(num_sectors):
                sector1 = image1[i * sector_height:(i + 1) * sector_height, j * sector_width:(j + 1) * sector_width]
                sector2 = image2[i * sector_height:(i + 1) * sector_height, j * sector_width:(j + 1) * sector_width]

                diff = cv2.absdiff(sector1, sector2)
                mse = (diff ** 2).mean()
                mse_values.append(mse)
        
        # Compare differences to the threshold and find sectors with significant movement
        significant_movement = [mse > threshold for mse in mse_values]

        # Report which sectors have significant movement
        for i, has_movement in enumerate(significant_movement):
            if has_movement:
                row = i // num_sectors + 1
                col = i % num_sectors + 1
                # print(f'Sector ({row}, {col}) has significant movement. {image1_path}')
                return True, mse.max()
            else:
                return False, mse.max()
        
    def custom_sort(self, item):
        return item["sequence_score"], (item["mse"]+item["mag"])


    def generate_scoring(self, case_path):                
                                                    
        dict_list = []

        # loop over case overviews
        for overview_folder in natsorted(os.listdir(case_path)):

            # check for valid overview folders
            if(overview_folder !="case_info.json" and overview_folder != ".ipynb_checkpoints"):
                classification = [None]*15
                dasa_metric = 0
                frame = 0
                image1_path=""
                image2_path=""
                num_sectors = 2
                highest_mse = 0
                highest_mag = 0

                overview_full_path = os.path.join(case_path, overview_folder)
                for img_name in natsorted(os.listdir(overview_full_path)):
                    if(img_name!=".ipynb_checkpoints"):
                        img_path = os.path.join(overview_full_path,img_name)
                        im = Image.open(img_path)
                        frame_img = im.convert('L')

                        np_img = np.array(frame_img)#.transpose(2,0,1)
                        stacked_img = np.stack((np_img,)*3, axis=2)

                        trans = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=self.img_size)])
                        inputs = trans(stacked_img)

                        inputs = inputs[None,:,:,:]

                        biof = self.bio_dis(inputs)
                        # class_item = biof.item()

                        class_item = self.embryo_classes[biof.item()]

                        if(class_item == 'tPB2'):    class_result = 0
                        elif(class_item == 'tPNa'):  class_result = 1
                        elif(class_item == 'tPNf'):  class_result = 2
                        elif(class_item == 't2'):    class_result = 3
                        elif(class_item == 't3'):    class_result = 4
                        elif(class_item == 't4'):    class_result = 5
                        elif(class_item == 't5'):    class_result = 6
                        elif(class_item == 't6'):    class_result = 7
                        elif(class_item == 't7'):    class_result = 8
                        elif(class_item == 't8'):    class_result = 9
                        elif(class_item == 't9+'):   class_result = 10
                        elif(class_item == 'tM'):    class_result = 11
                        elif(class_item == 'tSB'):   class_result = 12
                        elif(class_item == 'tB'):    class_result = 13
                        elif(class_item == 'tEB'):   class_result = 14
                        elif(class_item == 'tHB'):   class_result = 15
                        else: class_result = 16

                        classification[frame] = class_result


                        if(frame>0): 
                            image2_path=img_path
                            mov_mse, mse_max = self.movement_check(image1_path, image2_path, num_sectors)
                            mov_bg, mag_max = self.bg_subs(image1_path, image2_path, num_sectors)

                            if(highest_mse<mse_max): highest_mse = mse_max
                            if(highest_mag<mag_max): highest_mag = mag_max

                            # if(mov_mse and mov_bg): print("MOVEMENT!!!",image1_path, mse_max, mag_max)

                            image1_path=image2_path
                        else:
                            image1_path = img_path

                        frame = frame + 1

                # print("classification",classification)    


                dict_obj = {
                    "case": os.path.join(overview_folder),
                    "classification": classification,
                    "sequence_score": float(round((self.dasa.compute_DASA(classification)/3),2)),
                    "mse": float(highest_mse),
                    "mag":float(highest_mag)
                }

                dict_list.append(dict_obj)



        # Rank the dictionary objects by the float value
        sorted_dict_list = sorted(dict_list, key=self.custom_sort) #lambda x: x["sequence_score"],highest_mse+highest_mag)

        
        #Display the sorted list of dictionaries
        # for index, item in enumerate(sorted_dict_list, start=1):
        #     print(f"Rank {index}:")
        #     print("case:", item["case"])
        #     print("classification:", item["classification"])
        #     print("sequence_score:", item["sequence_score"])
        #     print("mse:", item["mse"])
        #     print("mag:", item["mag"])
        #     print()

        return sorted_dict_list

        # with open(os.path.join("ranking.json"), "w") as file:
        #     json.dump(sorted_dict_list, file)




