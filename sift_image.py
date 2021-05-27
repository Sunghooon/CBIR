#!/usr/bin/env python
# coding: utf-8

# Thanks for explanation and code, 
# https://liverungrow.medium.com/sift-bag-of-features-svm-for-classification-b5f775d8e55f
# https://www.programmersought.com/article/12294296973/
# SIFT (Bag of features) + SVM for classification + K-means clustering
# 2021. 5. 23

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import pandas as pd
import torch
from sklearn.metrics.pairwise import linear_kernel
import collections
from sklearn import svm

N_SAMPLES = 1000
DATA_PATH = 'shopee-product-matching/'
train = pd.read_csv(DATA_PATH + 'train.csv')
train['image'] = DATA_PATH + 'train_images/' + train['image']
origin = train
train = train.head(N_SAMPLES)
text_data = train['title']
tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
train['target'] = train.label_group.map(tmp)


# Experiment Part ---------------------------------------------
# SIFT matching between query image and trained image

img1 = cv2.imread(train['image'].iloc[19], 0)          # queryImage
img1_label = train['label_group'].iloc[19]
img2_idx = np.where(train['label_group'] == img1_label)
img2 = cv2.imread(train['image'].iloc[img2_idx[0][0]], 0)          # trainImage


# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
print('des1 shape :', des1.shape)
print('des2 shape :', des2.shape)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
print('# of matched features :', len(good))

        # cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
print((img3).shape)
plt.figure(figsize=(10, 7))
plt.imshow(img3)
plt.title('SIFT feature matching results within same label groups')
plt.show()


# SIFT matching between query image and trained image

img2 = cv2.imread(train['image'].iloc[1], 0)         

# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
print('des1 shape :', des1.shape)
print('des2 shape :', des2.shape)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
print('# of matched features :', len(good))

        # cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
print((img3).shape)
plt.figure(figsize=(10, 7))
plt.imshow(img3)
plt.title('SIFT feature matching results within different label groups')
plt.show()



# Implementation Part ---------------------------------------------
import joblib
from sklearn import svm
import cv2

n_cluster = 50
n_sift_feat = 200

def calcSiftFeature(img):
    #Set image sift feature key points to a maximum of 200
    sift = cv2.SIFT_create(n_sift_feat)
    #sift = cv2.SURF_create()

    #Calculate the feature points and feature point description of the picture
    keypoints, features = sift.detectAndCompute(img, None)
    #temp = features[np.random.randint(features.shape[0], size=n_samples)]

    return features


#Calculation word bag
def learnVocabulary(features):
    wordCnt = n_cluster # wordCnt is the number of categories  
    #criteria indicates the mode of iteration stop eps --- precision 0.1, max_iter --- meet more than the maximum number of iterations 20
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    #Get the initial center point of k-means clustering
    flags = cv2.KMEANS_RANDOM_CENTERS
    # Label, center = kmeans(input data (feature), number of clusters K, preset label, cluster stop condition, number of repeated clusters, initial cluster center point
    compactness, labels, centers = cv2.kmeans(features, wordCnt, None, criteria, 20, flags)
    return centers



#Calculate feature vector
def calcFeatVec(features, centers):
    featVec = np.zeros((1, n_cluster))
    for i in range(0, features.shape[0]):
        #The characteristic point of the i-th picture
        fi = features[i]
        diffMat = np.tile(fi, (n_cluster, 1)) - centers
        #axis=1Sum by line, that is, the distance between the feature and each center point
        sqSum = (diffMat**2).sum(axis=1)
        dist = sqSum**0.5
        #Ascending order
        sortedIndices = dist.argsort()
        #Remove the smallest distance, that is, find the nearest center point
        idx = sortedIndices[0]
        #The center point corresponds to +1
        featVec[0][idx] += 1
    
    return featVec



# Build training set's word bag, After calculating the word bag, you need to save the word bag.
# You will need this word bag to find the feature vector of the picture in the subsequent test and 
# when predicting the new picture

def build_center():
    features = np.float32([]).reshape(0, 128)
    for idx in tqdm(range(0, N_SAMPLES)):
        img = cv2.imread(train['image'].iloc[idx], 0)

        # Get sift feature points of image
        #kp, img_feat = sift.detectAndCompute(img, None)
        img_feat = calcSiftFeature(img)
        # pick 'n_samples' descriptors from each image
        #des_sample = img_feat[np.random.randint(img_feat.shape[0], size=n_samples)]
        features = np.append(features, img_feat, axis=0)

    print('features shape :', features.shape)
    #features = np.reshape(features, (N_SAMPLES*n_samples,-1))
    # Training set of word bags
    centers = learnVocabulary(features)
    # Save the word bag
    filename = 'sift_svm_materials/' + str(n_cluster) + 'svm_centers_.npy'
    np.save(filename, centers)
    print('Word bag :', centers.shape)



# Find the feature vector of the picture through the word bag. In the experiment, path refers to the path,
# the training set path and the test set path. Enter the training set path to learn the picture feature
# vector and label, which is used to input the SVM classifier to train the model.
# Enter the set path, used to input the obtained image feature vector into the trained SVM classifier
# to obtain the prediction result, and compare it with the fact label to obtain the correct rate of the
# model on the test set.

def cal_vec():
    filename = 'sift_svm_materials/' + str(n_cluster) + 'svm_centers_.npy'
    centers = np.load(filename)
    data_vec = np.float32([]).reshape(0, n_cluster)
    labels = np.float32([])
    for idx in range(0, N_SAMPLES):
        img = cv2.imread(train['image'].iloc[idx], 0)
        img_f = calcSiftFeature(img) # shouldn't use features we already have?
        img_vec = calcFeatVec(img_f, centers)
        data_vec = np.append(data_vec, img_vec, axis = 0)
        #labels = np.append(labels, idx)
        labels = np.append(labels, train['label_group'].iloc[idx])
    print('data vec :', data_vec.shape)
    print('feature of the training set was calculated successfully')
    return data_vec, labels



# Train SVM classifier
def SVM_Train(data_vec, labels):
    # Set SVM model param
    #clf = svm.SVC(decision_function_shape='ovr')
    #clf = svm.SVC(kernel='poly', degree=8)
    clf = svm.SVC(kernel='rbf')
    clf.fit(data_vec,labels)
    joblib.dump(clf, 'sift_svm_materials//svm_model.m')

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


# Test
from sklearn.metrics import f1_score

def SVM_Test():
    filename = 'sift_svm_materials//' + str(n_cluster) + 'svm_centers_.npy'
    
    clf = joblib.load('sift_svm_materials//svm_model.m')
    centers = np.load(filename)
    
    data_vec, labels = cal_vec()
    res = clf.predict(data_vec)
    num_test = data_vec.shape[0]
    acc = 0
    #print('labels :',labels)
    #print('num_test :',num_test)
    #print('res :', res)
    # return 1, 2
    #return labels, res # debugging
    for i in range(num_test):
        if labels[i]  == res[i]:
            acc += 1
    # Scoring
    
    return acc/num_test, res



build_center()
data_vec, labels = cal_vec()
SVM_Train(data_vec, train['label_group'])
acc, res = SVM_Test()



from sklearn.metrics import f1_score

f1 = f1_score(train['label_group'], res, average='macro')
recall = (f1*acc)/(2*acc-f1)
print('precision :', acc)
print('f1 score :', f1_score(train['label_group'], res, average='macro'))
print('recall :', recall)

# score history, 2021. 5. 25
# precision : 0.204
# f1 score : 0.12294857012525906
# recall : 0.08798941410879545



# plotting histogram
x = range(0, n_cluster)
y = data_vec[0]
plt.figure(figsize=(10,6))
plt.title('#1 image cluster histogram')
plt.xlabel('cluster index')
plt.ylabel('how many matches')
plt.bar(x, y)




def predict(img):
    filename = 'sift_svm_materials//' + str(n_cluster) + 'svm_centers_.npy'

    clf = joblib.load('sift_svm_materials//svm_model.m')
    centers = np.load(filename)
    features = calcSiftFeature(img)
    featVec = calcFeatVec(features, centers)
    case = np.float32(featVec)
    pred = clf.predict(case)
    return pred



# demo with img 
print('Demo -----------------------------------------------------------------------------')
print('LEFT : input')
print('RIGHT : output')
print('it predicts label_group from input image, output is one of image from output label')
img_idx = 1
plt.figure(figsize=(100,100))
for i in range(0,10):
    
    given_img = cv2.imread(origin['image'][i], 0)
    pred = predict(given_img)
    #print(res) # int value
    IDX = np.where(np.array(origin['label_group']) == pred)
    #print(IDX[0])
    gt_img = cv2.imread(origin['image'][IDX[0][0]],0)
    
    plt.subplot(10,2,img_idx)
    plt.imshow(given_img, cmap='bone')
    img_idx += 1
    
    plt.subplot(10,2,img_idx)
    plt.imshow(gt_img, cmap='bone')
    img_idx += 1
    
plt.show()

