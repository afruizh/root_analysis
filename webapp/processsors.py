import torch
import torchvision


from PIL import Image
import numpy as np

from skimage.morphology import erosion

from dependecies.segroot.model import SegRoot
from dependecies.segroot.dataloader import pad_pair_256, normalize
from torchvision.transforms import v2 as transforms


import onnxruntime as ort
import cv2 as cv

import os

MODELS_PATH = r"./models"

def pad_256(img_path):
    image = Image.open(img_path)
    W, H = image.size
    img, _ = pad_pair_256(image, image)
    NW, NH = img.size
    img = torchvision.transforms.ToTensor()(img)
    img = normalize(img)
    return img, (H, W, NH, NW)

def pad_256_np(np_img):
    #image = Image.open(img_path)
    image = Image.fromarray(np_img)
    W, H = image.size
    img, _ = pad_pair_256(image, image)
    NW, NH = img.size
    img = torchvision.transforms.ToTensor()(img)
    img = normalize(img)
    return img, (H, W, NH, NW)

def merge_images(files, path=""):
    
    is_array = False
    if type(files[0]) == np.ndarray:
        is_array = True
        

    final_img = []
    resize_factor = 0.4
    offset0 = 930
    offset1 = 305 
    for index, file in enumerate(files):

        if is_array:
            img = file
        else:
            img = cv.imread(file)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        #img = cv.resize(img, (0,0), fx=resize_factor, fy=resize_factor)
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        
        if index == 0:
            img = img[0:img.shape[0]-offset0,0:img.shape[1]]
            final_img = img
        elif index == len(file)-1:
            final_img = cv.vconcat([final_img, img])
        else:
            #final_img = np.concatenate((final_img, img), axis=1)
            img = img[0:img.shape[0]-offset1,0:img.shape[1]]
            final_img = cv.vconcat([final_img, img])
        
    final_img = cv.resize(final_img, (0,0), fx=resize_factor, fy=resize_factor)  
     
    #cv.imwrite(path, final_img)
    print(final_img.shape)
    
    return final_img

class RootSegmentor():
    
    def __init__(self, model_type):
        
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model_type = model_type
        
        if model_type != "seg_model":        
            self.initialize()

        return
    
    def initialize(self):
        
        width = 8
        depth = 5

        weights_path = os.path.join(MODELS_PATH, r"best_segnet-(8,5)-0.6441.pt")
        
        if self.model_type == "segroot":
            #weights_path = os.path.join(r"D:\local_mydev\roots_finetuning\SegRoot0\weights\best_segnet-(8,5)-0.6441.pt"
            #weights_path = r"D:\local_mydev\SegRoot\weights\best_segnet-(8,5)-0.6441.pt"
            #weights_path = r"\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\4.Scripts\AndresRuiz\local_mydata_backup\model\roots\best_segnet-(8,5)-0.6441.pt"
            #weights_path = os.path.join(MODELS_PATH, r"AcceleratedBreedingInitiative\4.Scripts\AndresRuiz\local_mydata_backup\model\roots\best_segnet-(8,5)-0.6441.pt")
            weights_path = os.path.join(MODELS_PATH, r"best_segnet-(8,5)-0.6441.pt")
        elif self.model_type == "segroot_finetuned":
            #weights_path = r"\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\4.Scripts\AndresRuiz\local_mydata_backup\model\roots\segroot-(8,5)_finetuned.pt"
            #weights_path = os.path.join(MODELS_PATH, r"AcceleratedBreedingInitiative\4.Scripts\AndresRuiz\local_mydata_backup\model\roots\segroot-(8,5)_finetuned.pt")
            weights_path = os.path.join(MODELS_PATH, r"segroot-(8,5)_finetuned.pt")
        elif self.model_type == "segroot_finetuned_dec":
            #weights_path = r"\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\4.Scripts\AndresRuiz\local_mydata_backup\model\roots\segroot-(8,5)_finetuned_dec_full.pt"
            #weights_path = r"\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\4.Scripts\AndresRuiz\local_mydata_backup\model\roots\segroot-(8,5)_finetuned_clas.pt"
            #weights_path = os.path.join(MODELS_PATH, r"AcceleratedBreedingInitiative\4.Scripts\AndresRuiz\local_mydata_backup\model\roots\segroot-(8,5)_finetuned_clas.pt")
            weights_path = os.path.join(MODELS_PATH, r"segroot-(8,5)_finetuned.pt")
            
        self.model = SegRoot(width, depth).to(self.device)

        if self.device.type == "cpu":
            print("load weights to cpu")
            #print(weights_path.as_posix())
            self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        else:
            print("load weights to gpu")
            #print(weights_path.as_posix())
            self.model.load_state_dict(torch.load(weights_path))
            
        for p in self.model.parameters():
            p.requires_grad = False
            
        self.model.eval()
                
        return    
    
    def predict(self, img_path):
        
        if self.model_type == "seg_model":
            
            print(str(type(img_path)))
            
            if type(img_path) == np.ndarray:
                img = img_path  
            else:
                img = cv.imread(img_path)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            
            weights_path = r"\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\4.Scripts\AndresRuiz\local_mydata_backup\model\roots\roots_model.onnx"
            weights_path = os.path.join(MODELS_PATH,"roots_model.onnx") 
            ort_sess = ort.InferenceSession(weights_path
                                ,providers=ort.get_available_providers()
                                )
            
            dim = img.shape
            
            transforms_list = []
            transforms_list.append(transforms.ToTensor())
            transforms_list.append(transforms.Resize((800,800)))
            #transforms_list.append(transforms.CenterCrop(800))
            #transforms_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

            apply_t =  transforms.Compose(transforms_list) 

            img = apply_t(img)

            outputs = ort_sess.run(None, {'input': [img.numpy()]})
            
            print(outputs)

            #np_res = outputs[0][0]

            output_image = outputs[0][:,:,1]
            final = cv.resize(output_image, (dim[0], dim[1]))
                
            return final
            
        else:

            thres = 0.9
            
            print(str(type(img_path)))
            
            if type(img_path) == np.ndarray:
                img, dims = pad_256_np(img_path)    
            else:
                img, dims = pad_256(img_path)
        
            H, W, NH, NW = dims
            
            img = img.to(self.device)
            
            img = img.unsqueeze(0)
            output = self.model(img)

            output = torch.squeeze(output)
            torch.cuda.empty_cache()

            prediction = output
            
            prediction[prediction >= thres] = 1.0
            prediction[prediction < thres] = 0.0
            
            if self.device.type == "cpu":
                prediction = prediction.detach().numpy()
            else:
                prediction = prediction.cpu().detach().numpy()
                
            prediction = erosion(prediction)
            # reverse padding
            prediction = prediction[
                (NH - H) // 2 : (NH - H) // 2 + H, (NW - W) // 2 : (NW - W) // 2 + W
            ]

            return prediction