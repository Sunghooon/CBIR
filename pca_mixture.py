
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


#print(torch.cuda.device_count())


DRAWING_PR = True
DATA_PATH = 'shopee-product-matching/'

import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2, matplotlib.pyplot as plt
from matplotlib.pyplot import getp, imshow
from matplotlib.image import imread
from IPython.display import Image
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

print('TORCH AVAILABLE :',torch.cuda.is_available())
print('WHICH DEVICE :', torch.cuda.get_device_name(0))
print('DRAWING PR CURVES :', DRAWING_PR)

train = pd.read_csv(DATA_PATH + 'train.csv')
train['image'] = DATA_PATH + 'train_images/' + train['image']

print('Select the number of samples you had used, both for text and img feature data are required.')
N_SAMPLE = int(input('how many samples did you use for training : '))

sample = train.head(N_SAMPLE)
tmp = sample.groupby('label_group').posting_id.agg('unique').to_dict()
sample['target'] = sample.label_group.map(tmp)
image_idx = sample['image']



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


def combine_for_sub(row):
    x = np.concatenate([row.predicted_img,row.predicted_text])
    return ' '.join( np.unique(x) )

def combine_for_cv(row):
    x = np.concatenate([row.predicted_img,row.predicted_text])
    return np.unique(x)

def PCA_svd(X, k, center=True):
    n = X.size()[0]
    ones = torch.ones(n).view([n,1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
    H = torch.eye(n) - h
    H = H.cuda()
    X_center =  torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)

    components  = v[:k].t()
    
    return components


filename = 'pca_materials/image_materials/' + str(N_SAMPLE) + '_trained_image_feature.csv'
DEVICE = 'cuda'
if os.path.isfile(filename) == False:
    print('Feature data is not exist, you should run > python pca_imageMake.py first !')
    sys.exit()

# Loading **train** Features 2021. 5. 19
img_train_feature = np.loadtxt(filename, delimiter=",")
img_train_feature = torch.from_numpy(img_train_feature)
img_train_feature = img_train_feature.to(DEVICE)

# l2 norm to kill all the sim in 0-1   ** img_train_feature
from sklearn.preprocessing import normalize
img_train_feature = img_train_feature.data.cpu().numpy()
img_train_feature = np.vstack(img_train_feature)
img_train_feature = normalize(img_train_feature)
img_train_feature = torch.from_numpy(img_train_feature)
img_train_feature = img_train_feature.to(DEVICE)

filename = 'pca_materials/text_materials/' + str(N_SAMPLE) + '_trained_text_feature.csv'
DEVICE = 'cuda'
if os.path.isfile(filename) == False:
    print('Feature data is not exist, you should run > python pca_titleMake.py first !')
    sys.exit()

# Loading **train** Features 2021. 5. 19
txt_train_feature = np.loadtxt(filename, delimiter=",")
txt_train_feature = torch.from_numpy(txt_train_feature)
txt_train_feature = txt_train_feature.to(DEVICE)

# l2 norm to kill all the sim in 0-1   ** txt_train_feature
from sklearn.preprocessing import normalize
txt_train_feature = txt_train_feature.data.cpu().numpy()
txt_train_feature = np.vstack(txt_train_feature)
txt_train_feature = normalize(txt_train_feature)
txt_train_feature = torch.from_numpy(txt_train_feature)
txt_train_feature = txt_train_feature.to(DEVICE)

if DRAWING_PR:
    prec_img_list = []
    rec_img_list = []

    prec_title_list = []
    rec_title_list = []

    prec_mix_list = []
    rec_mix_list = []

    step = np.arange(0,1,0.1)
    for iter in range(len(step)):
        print('iteration #',iter)
        # for image part
        # Checking img_train_feature with img_train_feature, 2021. 5. 19
        preds = []
        CHUNK = 100

        print('Finding similar images...')
        CTS = len(img_train_feature)//CHUNK
        if len(img_train_feature)%CHUNK != 0:
            CTS += 1
            
        for j in tqdm(range(CTS)):
            a = j*CHUNK
            b = (j+1)*CHUNK
            b = min(b, len(img_train_feature))
            
            distances = torch.matmul(img_train_feature, img_train_feature[a:b].T).T
            distances = distances.data.cpu().numpy()
            
            for k in range(b-a):
                IDX = np.where(distances[k,]>step[iter])[0][:]
                o = sample.iloc[IDX].posting_id.values
                preds.append(o)
            
        sample['predicted_img'] = preds
        sample['f1_img'] = sample.apply(getF1score('predicted_img'),axis=1)
        sample['prec_img'] = sample.apply(getPrecision('predicted_img'), axis=1)
        sample['rec_img'] = sample.apply(getRecall('predicted_img'), axis=1)
        #print('CV Score =', sample.f1_img.mean())
        #print('precision = ',sample.prec_img.mean())   
        #print('recall = ',sample.rec_img.mean())
        prec_img_list.append(sample.prec_img.mean())
        rec_img_list.append(sample.rec_img.mean())

        # for text part 
        # Checking train_text_feature with train_text_feature, 2021. 5. 21
        preds = []
        CHUNK = 100

        print('Finding similar text...')
        CTS = len(txt_train_feature)//CHUNK
        if len(txt_train_feature)%CHUNK != 0:
            CTS += 1
            
        for j in tqdm(range(CTS)):
            a = j*CHUNK
            b = (j+1)*CHUNK
            b = min(b, len(txt_train_feature))
            
            distances = torch.matmul(txt_train_feature, txt_train_feature[a:b].T).T
            distances = distances.data.cpu().numpy()

            for k in range(b-a):
                IDX = np.where(distances[k,]>step[iter])[0][:]
                o = sample.iloc[IDX].posting_id.values
                preds.append(o)
            
        sample['predicted_text'] = preds
        sample['f1_txt'] = sample.apply(getF1score('predicted_text'),axis=1)
        sample['prec_txt'] = sample.apply(getPrecision('predicted_text'), axis=1)
        sample['rec_txt'] = sample.apply(getRecall('predicted_text'), axis=1)
        #print('CV Score =', sample.f1_txt.mean())
        #print('precision = ',sample.prec_txt.mean())   
        #print('recall = ',sample.rec_txt.mean())
        prec_title_list.append(sample.prec_txt.mean())
        rec_title_list.append(sample.rec_txt.mean())

        sample['predicted_mix'] = sample.apply(combine_for_cv, axis=1)
        sample['f1'] = sample.apply(getF1score('predicted_mix'),axis=1)
        sample['prec'] = sample.apply(getPrecision('predicted_mix'), axis=1)
        sample['rec'] = sample.apply(getRecall('predicted_mix'), axis=1)
        #print('CV Score =', sample.f1.mean())
        #print('precision = ',sample.prec.mean())   
        #print('recall = ',sample.rec.mean())
        prec_mix_list.append(sample.prec.mean())
        rec_mix_list.append(sample.rec.mean())



    plt.plot(rec_title_list, prec_title_list, label='text pca')
    plt.plot(rec_img_list, prec_img_list, label='image pca')
    plt.plot(rec_mix_list, prec_mix_list, label='mixed pca')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curves with PCA methods')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

else:
    # for image part
    # Checking img_train_feature with img_train_feature, 2021. 5. 19
    preds = []
    CHUNK = 100

    print('Finding similar images...')
    CTS = len(img_train_feature)//CHUNK
    if len(img_train_feature)%CHUNK != 0:
        CTS += 1
        
    for j in tqdm(range(CTS)):
        a = j*CHUNK
        b = (j+1)*CHUNK
        b = min(b, len(img_train_feature))
        
        distances = torch.matmul(img_train_feature, img_train_feature[a:b].T).T
        distances = distances.data.cpu().numpy()
        
        for k in range(b-a):
            IDX = np.where(distances[k,]>0.95)[0][:]
            o = sample.iloc[IDX].posting_id.values
            preds.append(o)


    sample['predicted_img'] = preds
    sample['f1_img'] = sample.apply(getF1score('predicted_img'),axis=1)
    sample['prec_img'] = sample.apply(getPrecision('predicted_img'), axis=1)
    sample['rec_img'] = sample.apply(getRecall('predicted_img'), axis=1)
    print('PCA IMAGE SCORE -----------------------------------')
    print('CV Score =', sample.f1_img.mean())
    print('precision = ',sample.prec_img.mean())   
    print('recall = ',sample.rec_img.mean())

    # for text part 
    # Checking train_text_feature with train_text_feature, 2021. 5. 21
    preds = []
    CHUNK = 100

    print('Finding similar text...')
    CTS = len(txt_train_feature)//CHUNK
    if len(txt_train_feature)%CHUNK != 0:
        CTS += 1
        
    for j in tqdm(range(CTS)):
        a = j*CHUNK
        b = (j+1)*CHUNK
        b = min(b, len(txt_train_feature))
        
        distances = torch.matmul(txt_train_feature, txt_train_feature[a:b].T).T
        distances = distances.data.cpu().numpy()

        for k in range(b-a):
            IDX = np.where(distances[k,]>0.95)[0][:]
            o = sample.iloc[IDX].posting_id.values
            preds.append(o)
        
    sample['predicted_text'] = preds
    sample['f1_txt'] = sample.apply(getF1score('predicted_text'),axis=1)
    sample['prec_txt'] = sample.apply(getPrecision('predicted_text'), axis=1)
    sample['rec_txt'] = sample.apply(getRecall('predicted_text'), axis=1)
    print('PCA TEXT SCORE -----------------------------------')
    print('CV Score =', sample.f1_txt.mean())
    print('precision = ',sample.prec_txt.mean())   
    print('recall = ',sample.rec_txt.mean())
    sample['predicted_mix'] = sample.apply(combine_for_cv, axis=1)
    sample['f1'] = sample.apply(getF1score('predicted_mix'),axis=1)
    sample['prec'] = sample.apply(getPrecision('predicted_mix'), axis=1)
    sample['rec'] = sample.apply(getRecall('predicted_mix'), axis=1)
    print('PCA MIX SCORE -----------------------------------')
    print('CV Score =', sample.f1.mean())
    print('precision = ',sample.prec.mean())   
    print('recall = ',sample.rec.mean())

    sample['matches'] = sample.apply(combine_for_sub,axis=1)

    sample[['posting_id','predicted_mix']].to_csv('pca_materials/mixed_materials/pca_mixture_result.csv', index=False)



