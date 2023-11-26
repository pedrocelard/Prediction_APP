import torch
import torchvision.transforms as transforms
import numpy as np

from io import BytesIO
from torchvision import models
from PIL import ImageFile
from PIL import Image
from .resnet import ResNet18

torch.backends.cudnn.enabled=True
ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.cuda.set_device(0)


class ImageClassifier:
    def __init__(self):
        # Load the pre-trained ResNet-18 model
        self.device = "cuda:0"
        self.model = ResNet18(num_classes=17)
        checkpoint = torch.load('./classification/model/ResNet18_EMBRYO_128_17.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['net'])
        self.model.to(self.device)

        self.model.eval()

        # Define the image transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128))
        ])

    def class_to_number(self, model_output):
        classes = ['empty','t2','t3','t4','t5','t6','t7','t8',
                   't9+','tB','tEB','tHB','tM','tPB2','tPNa','tPNf','tSB']
        class_item = classes[model_output]

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
    
        return class_result, class_item
    
    def classify_image(self, img):
        np_img = np.array(img.convert('L'))
        stacked_img = np.stack((np_img,)*3, axis=2)

        # Load and preprocess the image
        input_tensor = self.transform(stacked_img).unsqueeze(0).to(self.device)

        # Make the prediction
        with torch.no_grad():
            output = self.model(input_tensor)

        # Get the predicted class index
        _, predicted_class = torch.max(output, 1)

        return self.class_to_number(predicted_class.item())
    



        