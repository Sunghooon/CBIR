#!/usr/bin/env python
# coding: utf-8


# Thanks for Code from https://www.kaggle.com/mirzarahim/introduction-to-pca-image-compression-example
# Introduction to PCA: Image Compression example

# https://github.com/vivekrmk/Image-Compression-Principal-Component-Analysis-Pytorch/blob/main/Pytorch_PCA_journey.ipynb
# https://github.com/Erikfather/PCA-python/blob/master/Face_Rec.py



# Checking GPU Units

import torch
import os.path
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import collections
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

print('TORCH AVAILABLE :',torch.cuda.is_available())
print('WHICH DEVICE :', torch.cuda.get_device_name(0))
#print(torch.cuda.device_count())



DATA_PATH = 'shopee-product-matching/'

import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2, matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.image import imread
from IPython.display import Image
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

train = pd.read_csv(DATA_PATH + 'train.csv')
train['image'] = DATA_PATH + 'train_images/' + train['image']

N_SAMPLE = int(input('how many samples do you want for training : '))

#N_SAMPLE = 5000
K = 120 # PCA, num of principal components

sample = train.head(N_SAMPLE)
tmp = sample.groupby('label_group').posting_id.agg('unique').to_dict()
sample['target'] = sample.label_group.map(tmp)
image_idx = sample['image']

filename = 'pca_materials/image_materials/' + str(N_SAMPLE) + '_trained_image_feature.csv'



def getF1score(col):
    def f1score(row):
        n = len( np.intersect1d(row.target,row[col]) )
        return 2*n / (len(row.target)+len(row[col]))
    return f1score

def getPrecision(col): # col = oof_cnn
    def precision(row):
        
        a = np.in1d(row.target,row[col])
        temp = collections.Counter(a)
        correct = temp[True]/len(a)
        return correct

    return precision

def getRecall(col):
    def recall(row):
        return 1/len(row[col])
    return recall


class ImageTransform():
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    def __call__(self, img):
        return self.data_transform(img)



class Img_Dataset(torch.utils.data.Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img = img.convert('L')
        img_transformed = self.transform(img)
        
        return img_transformed


# Preparing train dataset, 2021. 5. 19
train_img_list = image_idx

mean = (0.0,)
std = (1.0,)

train_dataset = Img_Dataset(file_list = train_img_list,
                            transform=ImageTransform(mean, std))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False)

batch_iterator = iter(train_dataloader)
images = next(batch_iterator)

print(images.size())


# Making **train** features, 2021. 5. 22
from sklearn.decomposition import PCA

K = 1000
DEVICE = 'cuda'

train_feature = []
train_feature = torch.tensor(train_feature)


# transfer to DEVICE (GPU memory)
train_feature = train_feature.to(DEVICE)

with torch.no_grad():
    for batch in tqdm(train_dataloader):
        batch = batch.to(DEVICE)
        batch = batch.permute(0,2,3,1)[:,:,:,0]
        idx, row, col = batch.shape
        batch = batch.view([len(batch),-1])
        train_feature = torch.cat([train_feature, batch], dim = 0)
    
    print('Analyzing Principal Components...')
    train_feature = train_feature.data.cpu().numpy()
    pca_analysis = PCA()
    pca_analysis.fit(train_feature)

    var_cumulative = np.cumsum(pca_analysis.explained_variance_ratio_)*100
    k = np.argmax(var_cumulative>95)
    print("Number of components explaining 95% variance : ", k)

    plt.figure(figsize=[10,5])
    plt.title('Cumulative Explained Variance explained by the components')
    plt.ylabel('Cumulative Explained Variance')
    plt.xlabel('# of Principal Components')
    plt.axvline(x=k, color="k", linestyle="--")
    plt.axhline(y=95, color="r", linestyle="--")
    ax = plt.plot(var_cumulative)
    # Uncomment this line if you want to get this graph
    #plt.show()

    if k > 1000:
        K = 1000
    else:
        K = k

    pca_feature = PCA(n_components = K)
    principalComponents = pca_feature.fit_transform(train_feature)
    principalComponents = torch.tensor(principalComponents)
    principalComponents = principalComponents.to(DEVICE)
    
    train_feature = principalComponents
    
        
    

    # Saving **train** Features 2021. 5. 19
    train_feature = train_feature.data.cpu().numpy()
    np.savetxt(filename, train_feature, delimiter=",")

    # Loading **train** Features 2021. 5. 19
    train_feature = np.loadtxt(filename, delimiter=",")
    train_feature = torch.from_numpy(train_feature)
    train_feature = train_feature.to(DEVICE)

    # l2 norm to kill all the sim in 0-1   ** train_feature
    from sklearn.preprocessing import normalize
    train_feature = train_feature.data.cpu().numpy()
    train_feature = np.vstack(train_feature)
    train_feature = normalize(train_feature)
    train_feature = torch.from_numpy(train_feature)
    train_feature = train_feature.to(DEVICE)

print('Done!')
print('File is saved at ' , filename)



