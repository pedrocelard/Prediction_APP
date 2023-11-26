import numpy as np
from PIL import Image
import imageio
import os
import sys

folder_path = sys.argv[1]
file_path = sys.argv[2]
output_path = sys.argv[3]


# Load the .npz file
#data = np.load('./log/openai-2023-09-25-11-07-13-293487/1x15x128x128x1.npz')

data = np.load(os.path.join(folder_path,file_path))

# Define the desired minimum and maximum values for the range
new_min = 0  # New minimum value
new_max = 255  # New maximum value

# Loop over the arrays in the file
for i, arr in enumerate(data['arr_0']):
    
    sample_dir_img = "./"+output_path+"/case_"+str(i)+"/embryo_overview_img"
    sample_dir_gif = "./"+output_path+"/case_"+str(i)+"/embryo_overview_gif"
    os.makedirs(sample_dir_img)
    os.makedirs(sample_dir_gif)
    images = []
    
    for index, image in enumerate(arr):
        
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
        
        #img = Image.fromarray(image)
        images.append(image)
        # Save the image
        image.save(f'{sample_dir_img}/image_{i}_{index}.jpg')
    imageio.mimsave(os.path.join(sample_dir_gif,"embryo_"+str(i)+".gif"), images)