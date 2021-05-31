#!/usr/bin/env python
# coding: utf-8

# Thanks for Code from https://wikidocs.net/24603
# Extracting features by using TF-IDF features



from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import linear_kernel
import collections
import matplotlib.pyplot as plt
import os.path


N_SAMPLE = int(input('how many samples do you want for training : '))
DATA_PATH = 'shopee-product-matching/'
train = pd.read_csv(DATA_PATH + 'train.csv')
train = train.head(N_SAMPLE)
text_data = train['title']
tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
train['target'] = train.label_group.map(tmp)

filename = 'pca_materials/text_materials' + str(N_SAMPLE) + '_trained_text_feature.csv'

def getF1score(col):
    def f1score(row):
        n = len( np.intersect1d(row.target,row[col]) )
        return 2*n / (len(row.target)+len(row[col]))
    return f1score

def getPrecision(col): 
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

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(text_data)


# Extracting features from text_data
feature = tfidf_vectorizer.transform(text_data).toarray()
print('feature shape : ',feature.shape)



print('Analyzing Principal Components...')

# feature analysis
from sklearn.decomposition import PCA, IncrementalPCA
pca_analysis = PCA()
pca_analysis.fit(feature)

var_cumulative = np.cumsum(pca_analysis.explained_variance_ratio_)*100
k = np.argmax(var_cumulative>95)


print("Number of components explaining 95% variance: "+ str(k))

plt.figure(figsize=[10,5])
plt.title('Cumulative Explained Variance explained by the components')
plt.ylabel('Cumulative Explained variance')
plt.xlabel('Principal components')
plt.axvline(x=k, color="k", linestyle="--")
plt.axhline(y=95, color="r", linestyle="--")
ax = plt.plot(var_cumulative)
# Uncomment this line if you want to get this graph
#plt.show()

# Principal Components Analysis, Latest one 2021. 5. 22
from sklearn.decomposition import PCA

K = k
DEVICE = 'cuda'
train_feature = []
train_feature = torch.tensor(train_feature)
train_feature = train_feature.to(DEVICE)

batch = range(0, len(feature), 10)
a = 0
print('calculating principal components...')
with torch.no_grad():

    pca_feature = PCA(n_components = K)
    principalComponents = pca_feature.fit_transform(feature)
    principalComponents = torch.tensor(principalComponents)
    principalComponents = principalComponents.to(DEVICE)

    train_feature = principalComponents            

# Saving **train** Features 2021. 5. 19
train_feature = train_feature.data.cpu().numpy()
np.savetxt(filename, train_feature, delimiter=",")

train_feature = torch.from_numpy(train_feature)
train_feature = train_feature.to(DEVICE)

print('Done!')
print('File is saved at ' , filename)