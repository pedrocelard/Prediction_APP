import os
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms as transforms
import cv2
import json


from utils.biodev_DASA import DASA_metric
from utils.biodev_dis import BioDevDisc
from natsort import natsorted

from PIL import ImageFile
from PIL import Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.cuda.set_device(0)

folder_path = sys.argv[1]

dataset = "EMBRYO"
img_size = 128
num_classes = 17
embryo_classes = ('empty','t2', 't3', 't4', 't5', 't6',
        't7', 't8', 't9+', 'tB', 'tEB',
        'tHB', 'tM', 'tPB2', 'tPNa', 'tPNf', 'tSB')

#Load classifier
bio_dis = BioDevDisc(num_classes, 'cuda:0')
checkpoint = 'ResNet18_'+dataset+'_'+str(img_size)+'_'+str(num_classes)+'.pth'
checkpoint_path = os.path.join("../../Classification/BioLossClassifier/checkpoint/",checkpoint)
bio_dis.load_checkpoint(checkpoint_path)
        
dasa = DASA_metric()
dasa.load_distribution("./class_models_and_distributions/"+dataset.lower()+'_'+str(img_size)+'_'+str(num_classes)+".txt")


        
    
def bg_subs(image1_path, image2_path, num_sectors):
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
    # print("MAGNITUDE", magnitude.max(),image1_path)
    if(significant_movement.any()):
        # print(f'Significant movement. {image1_path}')
        return True, magnitude.max()
    else:
        return False, magnitude.max()
    # You can now use 'significant_movement' to identify regions with movement
    # cv2.imwrite("image1_path.jpeg", significant_movement.astype('uint8') * 255)

    
def movement_check(image1_path, image2_path, num_sectors):
    # print("image1_path",image1_path)
    # print("image2_path",image2_path)
    # print()
    # Load the two images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    threshold = 80
    
    # Calculate sector dimensions based on the number of sectors
    height, width, _ = image1.shape
    sector_height = height // num_sectors
    sector_width = width // num_sectors
    
    
    # Divide the images into four sectors (adjust dimensions as needed)
    # height, width, _ = image1.shape
    # sectors_img1 = [
    #     image1[0:height//2, 0:width//2],
    #     image1[0:height//2, width//2:],
    #     image1[height//2:, 0:width//2],
    #     image1[height//2:, width//2:]
    # ]    
    # sectors_img2 = [
    #     image2[0:height//2, 0:width//2],
    #     image2[0:height//2, width//2:],
    #     image2[height//2:, 0:width//2],
    #     image2[height//2:, width//2:]
    # ]

    
    mse_values = []
    # Iterate through sectors and calculate the MSE for each
    for i in range(num_sectors):
        for j in range(num_sectors):
            sector1 = image1[i * sector_height:(i + 1) * sector_height, j * sector_width:(j + 1) * sector_width]
            sector2 = image2[i * sector_height:(i + 1) * sector_height, j * sector_width:(j + 1) * sector_width]

            diff = cv2.absdiff(sector1, sector2)
            mse = (diff ** 2).mean()
            mse_values.append(mse)
            # if(mse > 80):
            #     print("MSE",mse, "PATH:",image1_path)
    
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
    
    # Calculate the Mean Squared Error (MSE) for each sector
    # for sector_a, sector_b in zip(sectors_img1,sectors_img2):
    #     diff = cv2.absdiff(sector_a, sector_b)
    #     mse = (diff ** 2).mean()
    #     mse_values.append(mse)
    #     if(mse > 80):
    #         print("MSE",mse, "PATH:",image1_path)

#     # Set a threshold (adjust as needed)
#     threshold = 100

#     # Compare differences to the threshold
#     significant_movement = [if(mse > threshold): mse, for mse in mse_values]

#     # Report which sectors have significant movement
#     for i, has_movement in enumerate(significant_movement):
#         if has_movement:
#             print(f'Sector {i+1} has significant movement. {image1_path}')

def custom_sort(item):
    return item["sequence_score"], (item["mse"]+item["mag"])
                                                                
for sample_folder in natsorted(os.listdir(folder_path)):
    dict_list = []
    for folder in natsorted(os.listdir(os.path.join(folder_path,sample_folder))):
        if(folder !="ranking.json" and folder != ".ipynb_checkpoints"):
            classification = [None]*15
            dasa_metric = 0
            frame = 0
            image1_path=""
            image2_path=""
            num_sectors = 2
            highest_mse = 0
            highest_mag = 0

            for img_name in natsorted(os.listdir(os.path.join(folder_path, sample_folder, folder,"embryo_overview_img"))):
                if(img_name!=".ipynb_checkpoints"):
                    img_path = os.path.join(folder_path, sample_folder, folder,"embryo_overview_img",img_name)
                    im = Image.open(img_path)
                    frame_img = im.convert('L')

                    np_img = np.array(frame_img)#.transpose(2,0,1)
                    stacked_img = np.stack((np_img,)*3, axis=2)

                    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=img_size)])
                    inputs = trans(stacked_img)

                    inputs = inputs[None,:,:,:]

                    biof = bio_dis(inputs)
                    # class_item = biof.item()

                    class_item = embryo_classes[biof.item()]

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
                        mov_mse, mse_max = movement_check(image1_path, image2_path, num_sectors)
                        mov_bg, mag_max = bg_subs(image1_path, image2_path, num_sectors)

                        if(highest_mse<mse_max): highest_mse = mse_max
                        if(highest_mag<mag_max): highest_mag = mag_max

                        # if(mov_mse and mov_bg): print("MOVEMENT!!!",image1_path, mse_max, mag_max)

                        image1_path=image2_path
                    else:
                        image1_path = img_path

                    frame = frame + 1

            # print("classification",classification)    

            dict_obj = {
                "case": os.path.join(folder_path, sample_folder, folder),
                "classification": classification,
                "sequence_score": float(round((dasa.compute_DASA(classification)/3),2)),
                "mse": float(highest_mse),
                "mag":float(highest_mag)
            }

            dict_list.append(dict_obj)



    # Rank the dictionary objects by the float value
    sorted_dict_list = sorted(dict_list, key=custom_sort) #lambda x: x["sequence_score"],highest_mse+highest_mag)

    # Display the sorted list of dictionaries
    for index, item in enumerate(sorted_dict_list, start=1):
        print(f"Rank {index}:")
        print("case:", item["case"])
        print("classification:", item["classification"])
        print("sequence_score:", item["sequence_score"])
        print("mse:", item["mse"])
        print("mag:", item["mag"])
        print()

    with open(os.path.join(folder_path, sample_folder, "ranking.json"), "w") as file:
        json.dump(sorted_dict_list, file)




