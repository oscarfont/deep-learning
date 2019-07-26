'''
Created on May 5, 2019

@author: deckyal
'''
import numpy as np
import torch
from torch.utils import data
import re
from PIL import Image
import cv2

import os



#Making native class loader
class MNIST(torch.utils.data.Dataset):
    
    def __init__(self,dataDir = './data/mnist/processed/training.pt',transform = None, is_noisy = False):
        self.data, self.labels = torch.load(dataDir)
        self.transform = transform
        self.is_noisy = is_noisy
        
    def __getitem__(self, index):
        data = self.data[index]
        lbl = self.labels[index]
        
        data = Image.fromarray(data.numpy(), mode='L')
        
        if self.is_noisy : 
            opencvImage = np.expand_dims(np.array(data),2)
            opencvImage = addNoise(opencvImage,var=.00001)
            opencvImage = np.squeeze(opencvImage,2)
            imgNoisy = Image.fromarray(opencvImage,mode='L')
                
            if self.transform is not None:
                imgNoisy = self.transform(imgNoisy)

        if self.transform is not None : 
            data = self.transform(data)
        
        if not self.is_noisy : 
            return data,lbl
        else : 
            return data,lbl,imgNoisy
    
        pass
    def __len__(self):
        return len(self.data)
    

class CKDataset(data.Dataset):
    
    def __init__(self, troot = "./data/CK/",transform = None, is_noisy = False):
        'initialization'
        #Read initalizes the list of file path and possibliy label as well. 
        label = [0,1,2,3,4,5,6,7]
        images = []
        labels = []
        
        for l in label : 
            dir = troot+str(l)
            for file in os.listdir(dir):
                if file.endswith(".jpg"):
                    #print(file)
                    #print(os.path.join("/mydir", file))
                    images.append(os.path.join(dir, file))
                    labels.append(str(l))
        
        self.labels = labels 
        self.images = images
        self.transform = transform
        self.is_noisy = is_noisy 
    
    def __getitem__(self,index):
        #Read all data, transform etc.
        
        img = Image.open(self.images[index])
            
        if self.is_noisy : 
            opencvImage = cv2.imread(self.images[index])#cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            opencvImage = addNoise(opencvImage,var=.01)
            imgNoisy = Image.fromarray(opencvImage)
                
            if self.transform is not None:
                imgNoisy = self.transform(imgNoisy)
                
        if self.transform is not None:
            img = self.transform(img)
        
        label = int(self.labels[index])
        
        #print(img.shape,torch.FloatTensor([label]))
        
        if not self.is_noisy : 
            return img,label
        else : 
            return img,label,imgNoisy
    def __len__(self):
        #Len
        return len(self.labels)
    
    

def addNoise (image,noise_type="gauss",var = .01):
    """
    Generate noise to a given Image based on required noise type
    
    Input parameters:
        image: ndarray (input image data. It will be converted to float)
        
        noise_type: string
            'gauss'        Gaussian-distrituion based noise
            'poission'     Poission-distribution based noise
            's&p'          Salt and Pepper noise, 0 or 1
            'speckle'      Multiplicative noise using out = image + n*image
                           where n is uniform noise with specified mean & variance
    """
    row,col,ch= image.shape
    if noise_type == "gauss":       
        mean = 0.0
        #var = 0.001
        sigma = var**0.5
        gauss = np.array(image.shape)
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        #print(gauss)
        noisy = image + gauss*255
        return noisy.astype('uint8')
    elif noise_type == "s&p":
        s_vs_p = 0.5
        amount = 0.09
        out = image
        # Generate Salt '1' noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 255
        # Generate Pepper '0' noise
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type =="speckle":
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy
    else:
        return image
    
    
    
    
    
    
    
    