import os
import torch
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image

#check for cpu or gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#set image loader
imsize= 224
loader = transforms.Compose([transforms.Resize(imsize), 
                             transforms.CenterCrop(224), 
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                             ])


class Prediction():
    def __init__(self,class_names=['covid','non'],
                 model_name=os.path.join("model","covid_resnet18_epoch50.pt")):
        self.class_names = class_names
        self.model= torch.load(model_name, map_location='cpu') 
        self.model.eval()


    def image_loader(self,image_path):
        """load image, returns cuda tensor"""
        image = Image.open(image_path).convert("RGB")
        image = loader(image).float()
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
        return image


    def predict(self,image_path):
        cur_img= self.image_loader(image_path)
        model_output= self.model(cur_img)
        cur_pred = model_output.max(1, keepdim=True)[1]
        data = self.class_names[int(cur_pred.data.numpy())]
        print("Covid predicted label:%s" %(data))
        return data