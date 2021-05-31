DATA_PATH = 'shopee-product-matching/'
filename = 'cnn_feature_materials/feature.csv'

import os.path
import psutil
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2, matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import sys

from PIL import Image

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

# import cudf, cuml, cupy
# from cuml.feature_extraction.text import TfidfVectorizer
# from cuml.neighbors import NearestNeighbors

def getMetric(col):
    def f1score(row):
        n = len( np.intersect1d(row.target,row[col]) )
        return 2*n / (len(row.target)+len(row[col]))
    
    return f1score

def myMetric(col):
    def precision(row):
        n = len( np.intersect1d(row.target,row[col]) )
        if len(row[col]) == 0:
            return 0
        else:
            return n / len(row[col])
    
    return precision
        
def myMetric_recall(col):
    def recall(row):
        n = len( np.intersect1d(row.target,row[col]) )
        return n / len(row.target)
    return recall


def combine_for_sub(row):
    x = np.concatenate([row.oof_text,row.oof_cnn])
    return ' '.join( np.unique(x) )

def combine_for_cv(row):
    x = np.concatenate([row.oof_text,row.oof_cnn])
    return np.unique(x)


class ShopeeImageDataset(Dataset):
    def __init__(self, img_path, transform):
        self.img_path = img_path
        self.transform = transform
        
    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.img_path)

class ShopeeImageEmbeddingNet(nn.Module):
    def __init__(self):
        super(ShopeeImageEmbeddingNet, self).__init__()
              
        model = models.resnet18(True)
        model.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        self.model = model
        
    def forward(self, img):        
        out = self.model(img)
        return out



def run():
    threshold = 0.95
    torch.multiprocessing.freeze_support()
    COMPUTE_CV = True

    # COMPUTE_CV = False

    train = pd.read_csv(DATA_PATH + 'train.csv')
    train = train.head(6000)
    train['image'] = DATA_PATH + 'train_images/' + train['image']
    tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
    train['target'] = train.label_group.map(tmp)
    # train_gf = cudf.read_csv(DATA_PATH + 'train.csv')
    
    if os.path.isfile(filename) == False:
        print('Feature data is not exist, creating...')
        #sys.exit()

        imagedataset = ShopeeImageDataset(train['image'].values,
        transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]))
        
        imageloader = torch.utils.data.DataLoader(
            imagedataset,
            batch_size=10, shuffle=False, num_workers=2
        )

        DEVICE = 'cuda'

        imgmodel = ShopeeImageEmbeddingNet()
        imgmodel = imgmodel.to(DEVICE)

        imagefeat = []
        with torch.no_grad():
            for data in tqdm(imageloader):
                data = data.to(DEVICE)
                feat = imgmodel(data)
                feat = feat.reshape(feat.shape[0], feat.shape[1])
                feat = feat.data.cpu().numpy()
                
                imagefeat.append(feat)
            
            
        from sklearn.preprocessing import normalize

        # l2 norm to kill all the sim in 0-1
        imagefeat = np.vstack(imagefeat)
        imagefeat = normalize(imagefeat)

        imgmodel = ShopeeImageEmbeddingNet()
        imgmodel = imgmodel.to(DEVICE)

        # if don't have feature.csv
        np.savetxt(filename, imagefeat, delimiter=',')
        #df = pd.DataFrame(imagefeat)
        #df.to_csv(READ_PATH + 'feature.csv', index=False)
    else:
        # if have feature.csv
        imagefeat = np.loadtxt(filename, delimiter=',')
        from sklearn.preprocessing import normalize

        # l2 norm to kill all the sim in 0-1
        imagefeat = np.vstack(imagefeat)
        imagefeat = normalize(imagefeat)

        imagefeat = torch.from_numpy(imagefeat)
        imagefeat = imagefeat.cuda()


    preds = []
    CHUNK = 1024*4


    print('Finding similar images...')
    CTS = len(imagefeat)//CHUNK
    if len(imagefeat)%CHUNK!=0: CTS += 1
    for j in range( CTS ):
        
        a = j*CHUNK
        b = (j+1)*CHUNK
        b = min(b, len(imagefeat))
        print('chunk',a,'to',b)
        
        distances = torch.matmul(imagefeat, imagefeat[a:b].T).T
        distances = distances.data.cpu().numpy()
        # distances = np.dot(imagefeat[a:b,], imagefeat.T)
        
        for k in range(b-a):
            # IDX = cupy.where(distances[k,]>0.95)[0]
            IDX = np.where(distances[k,]>threshold)[0][:]
            o = train.iloc[IDX].posting_id.values
    #         o = train.iloc[cupy.asnumpy(IDX)].posting_id.values
            preds.append(o)
            
    del imagefeat

    print('done')



    train['oof_cnn'] = preds # preds for prediction

    if COMPUTE_CV:
        train['f1'] = train.apply(getMetric('oof_cnn'),axis=1)
        print('CV score for baseline =',train.f1.mean())
        

    if COMPUTE_CV:
        train['precision'] = train.apply(myMetric('oof_cnn'),axis=1)
        print('precision = ',train.precision.mean())


    if COMPUTE_CV:
        train['recall'] = train.apply(myMetric_recall('oof_cnn'),axis=1)
        print('recall = ', train.recall.mean())

        
    from sklearn.feature_extraction.text import TfidfVectorizer
    model = TfidfVectorizer(stop_words=None, binary=True, max_features=5000)
    text_embeddings = model.fit_transform(train.title).toarray()
    print('text embeddings shape',text_embeddings.shape)

    text_embeddings = torch.from_numpy(text_embeddings)
    text_embeddings = text_embeddings.cuda()


    preds = []
    CHUNK = 1024*4

    print('Finding similar titles...')
    CTS = len(train)//CHUNK
    if len(train)%CHUNK!=0: CTS += 1
    CTS_index = 0
    for j in range( CTS ):
        
        a = j*CHUNK
        b = (j+1)*CHUNK
        b = min(b,len(train))
        print('chunk',a,'to',b)
        
        # COSINE SIMILARITY DISTANCE
        # cts = np.dot( text_embeddings, text_embeddings[a:b].T).T
        cts = torch.matmul(text_embeddings, text_embeddings[a:b].T).T
        cts = cts.data.cpu().numpy()
        print(cts.shape)
        for k in range(b-a):
            # IDX = np.where(cts[k,]>0.7)[0]
            IDX = np.where(cts[k,]>threshold)[0]
            o = train.iloc[IDX].posting_id.values
            preds.append(o)
            CTS_index += 1
    del model, text_embeddings


    train['oof_text'] = preds

    if COMPUTE_CV:
        train['f1'] = train.apply(getMetric('oof_text'),axis=1)
        print('CV score for baseline =',train.f1.mean())
        
    if COMPUTE_CV:
        train['precision_text'] = train.apply(myMetric('oof_text'),axis=1)
        print('precision = ',train.precision_text.mean())   
        
    if COMPUTE_CV:
        train['recall_text'] = train.apply(myMetric_recall('oof_text'),axis=1)
        print('recall = ',train.recall_text.mean())
    # 0.6137154152579091 0.7
    # 0.6507316994356058 0.6




    if COMPUTE_CV:
        tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
        train['target'] = train.label_group.map(tmp)

        train['oof'] = train.apply(combine_for_cv,axis=1)

        train['f1'] = train.apply(getMetric('oof'),axis=1)
        print('CV Score =', train.f1.mean() )

        train['precision_mix'] = train.apply(myMetric('oof'),axis=1)
        print('precision = ',train.precision_mix.mean())   
        
        train['recall_mix'] = train.apply(myMetric_recall('oof'),axis=1)
        print('recall = ',train.recall_mix.mean())

    train['matches'] = train.apply(combine_for_sub,axis=1)

    train[['posting_id','matches']].to_csv('cnn_feature_materials/' + 'submission.csv',index=False)



if __name__ == '__main__':
    DRAWING_PR = False
    print('DRAWING PR CURVES :', DRAWING_PR)
    
    if DRAWING_PR == False:
        run()
