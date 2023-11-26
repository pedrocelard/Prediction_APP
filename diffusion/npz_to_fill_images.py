import numpy as np
from PIL import Image
import imageio
import os
import sys

folder_path = sys.argv[1]
output_path = sys.argv[2]


# Load the .npz file
# data = np.load('./log/openai-2023-09-25-11-07-13-293487/1x15x128x128x1.npz')

# Define the desired minimum and maximum values for the range
new_min = 0  # New minimum value
new_max = 255  # New maximum value

sample_dir_img = output_path+"/embryo_fill_img"
sample_dir_gif = output_path+"/embryo_fill_gif"
        
if (not os.path.exists(sample_dir_img)): os.makedirs(sample_dir_img)
if (not os.path.exists(sample_dir_gif)): os.makedirs(sample_dir_gif)

images = []
for section in range(0,14):
    
    data = np.load(os.path.join(folder_path,str(section)+".npz"))

    # Loop over the arrays in the file
    for i, arr in enumerate(data['arr_0']):
        
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

            
            #### TODO IMPORTANTE: Las primeras y últimas imágenes de cada sección van a ser repetidas
            # así que es necesario no añadirlas al gif ni guardarlas como imagen
            
            #img = Image.fromarray(image)
            if(section==0):
                images.append(image)
                # Save the image
                image.save(f'{sample_dir_img}/image_{section}_{index}.jpg')
            elif(index!=0):
                images.append(image)
                # Save the image
                image.save(f'{sample_dir_img}/image_{section}_{index}.jpg')
                
            
imageio.mimsave(os.path.join(sample_dir_gif,"embryo_full.gif"), images)
