#!/usr/bin/env python
# coding: utf-8


# Thanks for Code from https://www.kaggle.com/mirzarahim/introduction-to-pca-image-compression-example
# Introduction to PCA: Image Compression example

# https://github.com/vivekrmk/Image-Compression-Principal-Component-Analysis-Pytorch/blob/main/Pytorch_PCA_journey.ipynb
# https://github.com/Erikfather/PCA-python/blob/master/Face_Rec.py



# Checking GPU Units

from scipy.sparse.construct import kronsum
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

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())



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

N_SAMPLE = 5000
n_test = 1000
K = 120 # PCA, num of principal components

sample = train.head(N_SAMPLE)
tmp = sample.groupby('label_group').posting_id.agg('unique').to_dict()
sample['target'] = sample.label_group.map(tmp)
test   = train.loc[N_SAMPLE+1:N_SAMPLE+n_test]
test = test.reset_index(drop=True) # initialize indexing
image_idx = sample['image']

filename = 'pca_image_materials/' + str(N_SAMPLE) + '_trained_image_feature.csv'



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

# if file exists
if os.path.isfile(filename):
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

# if file not exists
else:
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

    DEVICE = 'cuda'

    train_feature = []
    train_feature = torch.tensor(train_feature)

    # transfer to DEVICE (GPU memory)
    train_feature = train_feature.to(DEVICE)

    a = 1
    with torch.no_grad():
        for batch in tqdm(train_dataloader):
            batch = batch.to(DEVICE)
            batch = batch.permute(0,2,3,1)[:,:,:,0]
            idx, row, col = batch.shape
            batch = batch.view([len(batch),-1])
            train_feature = torch.cat([train_feature, batch], dim = 0)
        
        
        train_feature = train_feature.data.cpu()
        pca_analysis = PCA()
        pca_analysis.fit(train_feature)    
        var_cumulative = np.cumsum(pca_analysis.explained_variance_ratio_[:K])*100
        #k = np.argmax(var_cumulative>60)
        #print("Number of components explaining 80% variance : ", k)

        plt.figure(figsize=[10,5])
        plt.title('Cumulative Explained Variance explained by the components')
        plt.ylabel('Cumulative Explained Variance')
        plt.xlabel('# of Principal Components')
        plt.axvline(x=K, color="k", linestyle="--")
        #plt.axhline(y=80, color="r", linestyle="--")
        ax = plt.plot(var_cumulative)
        plt.show()

        pca_feature = PCA(n_components = K) # switching K, # of PC
        
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
    #print('chunk', a, 'to', b)
    
    #distances = torch.cdist(train_feature, train_feature[a:b], p=2.0).T
    distances = torch.matmul(train_feature, train_feature[a:b].T).T
    distances = distances.data.cpu().numpy()
    
    #print(type(distances))
    #print(distances.shape)
    '''
    for k in range(b-a):
        IDX = np.argmin(distances[k][:])
        o = sample.iloc[IDX].label_group
        preds.append(o)
    '''
    
    for k in range(b-a):
        #IDX = np.argmax(distances[k][:])
        IDX = np.where(distances[k,]>0.90)[0][:]
        o = sample.iloc[IDX].posting_id.values
        preds.append(o)
        #print(len(IDX))
    
sample['predicted_label'] = preds


# prepare test features 2021. 5. 19
'''
test_image_idx = test['image']
test_img_list = test_image_idx

test_dataset = Img_Dataset(file_list = test_img_list,
                            transform=ImageTransform(mean, std))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

batch_iterator = iter(test_dataloader)
test_images = next(batch_iterator)

len(batch_iterator)
#print(test_images.size())
'''







'''# Making **test** features, 2021. 5. 19 
DEVICE = 'cuda'

test_feature = []
test_feature = torch.tensor(test_feature)

# transfer to DEVICE (GPU memory)
test_feature = test_feature.to(DEVICE)

a = 1
with torch.no_grad():
    for batch in tqdm(test_dataloader):
        batch = batch.to(DEVICE)
        batch = batch.permute(0,2,3,1)[:,:,:,0]
        idx, row, col = batch.shape
        batch = batch.view([len(batch),-1])

        U,S,V = torch.pca_lowrank(batch, q=len(batch), center=True, niter=3)
        V = torch.tensor(V)
        
        test_feature = torch.cat([test_feature, V.T[:,:K]], dim = 0)

        a = a + 1'''
        

'''
# Saving **test** Features 2021. 5. 19
test_feature = test_feature.data.cpu().numpy()
np.savetxt(filename, test_feature, delimiter=",")



# Loading **test** Features 2021. 5. 19
test_feature = np.loadtxt(filename, delimiter=",")
test_feature = torch.from_numpy(test_feature)
test_feature = test_feature.to(DEVICE)

# l2 norm to kill all the sim in 0-1    ** test_feature
test_feature = test_feature.data.cpu().numpy()
test_feature = np.vstack(test_feature)
test_feature = normalize(test_feature)
test_feature = torch.from_numpy(test_feature)
test_feature = test_feature.to(DEVICE)
'''






# 2021. 5. 22. Scoring, LATEST

sample['f1'] = sample.apply(getF1score('predicted_label'),axis=1)

sample['Prec'] = sample.apply(getPrecision('predicted_label'),axis=1)

sample['Rec'] = sample.apply(getRecall('predicted_label'),axis=1)
print('F1 score for baseline = ', sample.f1.mean())
print('precision = ', sample.Prec.mean())
print('recall = ', sample.Rec.mean())

# score history
# 2021. 5. 22, samples:8000, K=1000
# F1 score for baseline =  0.7745054213895112
# precision =  0.781956476856477
# recall =  0.8097427292601712



'''# Checking train_feature with test_feature, 2021. 5. 19
preds = []
CHUNK = 100

print('Finding similar images...')
CTS = len(test_feature)//CHUNK
if len(test_feature)%CHUNK != 0:
    CTS += 1
    
for j in tqdm(range(CTS)):
    a = j*CHUNK
    b = (j+1)*CHUNK
    b = min(b, len(test_feature))
    #print('chunk', a, 'to', b)
    
    #distances = torch.cdist(train_feature, test_feature[a:b], p=2.0).T
    distances = torch.matmul(train_feature, test_feature[a:b].T).T
    distances = distances.data.cpu().numpy()
    
    #print(type(distances))
    #print(distances.shape)
    
    for k in range(b-a):
        #IDX = np.argmax(distances[k][:])
        IDX = np.where(distances[k,]>0.9)[0][:]
        o = sample.iloc[IDX].label_group.values
        preds.append(o)
        #print(len(IDX))
        
test['predicted_label'] = preds




# Scoring
# Calculate Precision
correct = 0
for i in range(len(test)):
    if len( np.intersect1d(test['predicted_label'][i], test['label_group'][i])) == 1:
        correct = correct + 1

precision = correct/len(test) * 100
print('num of correct : ', correct)
print('precision : ', precision)

# Calculate Recall
correct = 0
recall = 0
temp = 0

for i in range(len(test)):
    if len( np.intersect1d(test['predicted_label'][i], test['label_group'][i])) == 1:
        L = len(test['predicted_label'][i])
        correct = correct + 1
        temp = 1/L
        recall = recall + temp

recall = recall / correct
print('recall : ', recall)

# Calculate F1 score
print('f1 : ', 2*(precision * recall)/(precision + recall))'''

