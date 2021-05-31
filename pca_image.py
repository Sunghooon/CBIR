#!/usr/bin/env python
# coding: utf-8


# Thanks for Code from https://www.kaggle.com/mirzarahim/introduction-to-pca-image-compression-example
# Introduction to PCA: Image Compression example

# https://github.com/vivekrmk/Image-Compression-Principal-Component-Analysis-Pytorch/blob/main/Pytorch_PCA_journey.ipynb
# https://github.com/Erikfather/PCA-python/blob/master/Face_Rec.py



# Checking GPU Units

import torch
import sys
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

N_SAMPLE = int(input('how many samples did you use for training : '))

sample = train.head(N_SAMPLE)
tmp = sample.groupby('label_group').posting_id.agg('unique').to_dict()
sample['target'] = sample.label_group.map(tmp)
image_idx = sample['image']

filename = 'pca_materials/image_materials/' + str(N_SAMPLE) + '_trained_image_feature.csv'
exportname = 'pca_materials/image_materials/' + 'result.csv'

if os.path.isfile(filename) == False:
    print('Feature data is not exist, you should run > python pca_imageMake.py first !')
    sys.exit()

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


DEVICE = 'cuda'

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

# Checking train_feature with train_feature, 2021. 5. 19
preds = []
CHUNK = 100

print('Finding similar images...')
CTS = len(train_feature)//CHUNK
if len(train_feature)%CHUNK != 0:
    CTS += 1
    
for j in tqdm(range(CTS)):
    a = j*CHUNK
    b = (j+1)*CHUNK
    b = min(b, len(train_feature))
    
    distances = torch.matmul(train_feature, train_feature[a:b].T).T
    distances = distances.data.cpu().numpy()
    
    for k in range(b-a):
        IDX = np.where(distances[k,]>0.95)[0][:]
        o = sample.iloc[IDX].posting_id.values
        preds.append(o)
    
sample['predicted'] = preds

# 2021. 5. 22. Scoring, LATEST

sample['f1'] = sample.apply(getF1score('predicted'),axis=1)

sample['Prec'] = sample.apply(getPrecision('predicted'),axis=1)

sample['Rec'] = sample.apply(getRecall('predicted'),axis=1)
print('F1 score for baseline = ', sample.f1.mean())
print('precision = ', sample.Prec.mean())
print('recall = ', sample.Rec.mean())

sample[['posting_id','predicted']].to_csv(exportname, index=False)
print('Done !')
print('Result file was saved at ' + exportname)


# score history
# 2021. 5. 22, samples:8000, K=1000
# F1 score for baseline =  0.7745054213895112
# precision =  0.781956476856477
# recall =  0.8097427292601712

