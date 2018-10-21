#! /usr/bin/python
# -*- coding: utf8 -*-

""" GAN-CLS """
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.prepro import *
from tensorlayer.cost import *
import numpy as np
import scipy
from scipy.io import loadmat
import time, os, re, nltk
from konlpy.tag import Kkma, Hannanum, Komoran, Mecab, Twitter
from utils import *
from model import *
import model
import matplotlib.pyplot as plt

twitter=Twitter()

###======================== PREPARE DATA ====================================###
print("Loading data from pickle ...")
import pickle

# 나의 vocab 가져오기
with open("./data/word2index.pkl", 'rb') as f:
    vocab = pickle.load(f)

# train/test image
with open("./data/image_train.pkl", 'rb') as f:
    images_train = pickle.load(f)
with open("./data/image_test.pkl", 'rb') as f:
    images_test = pickle.load(f)

# test/train index
with open("./data/test_idx.pkl", 'rb') as f:
    captions_ids_train = pickle.load(f)
with open("./data/train_idx.pkl", 'rb') as f:
    captions_ids_test = pickle.load(f)
# caption load
with open("./data/posc2idx.pkl", 'rb') as f:
    posc2idx = pickle.load(f)

# 데이터 확인하기 : 301개의 이미지 데이터까지만 로드
images_train=images_train[:151]
images_test=images_test[:150]
train_idx=captions_ids_train[:151]
test_idx=captions_ids_test[:150]

ni = int(np.ceil(np.sqrt(batch_size)))
n_images_train=len(images_train)
n_images_test=len(images_test)
n_captions_train=len(train_idx)
n_captions_test=len(test_idx)


print(".......데이터 확인하기.......")
print("images_train : ",len(images_train)) # 151개의 이미지
print("images_test : ",len(images_test)) # 150개의 이미지
print("vocab: ", len(vocab)) # 76303 의 pos(형태소 개수)
print("posc2idx : ",len(posc2idx)) # 301의 스토리
print("test_idx : " ,len(test_idx)) # 150
print("train_idx : " , len(train_idx)) # 151
print(train_idx)
print(test_idx)
captions_ids_train=[posc2idx[id-1] for id in train_idx]
captions_ids_test=[posc2idx[id+1] for id in test_idx]

length=[]
for i in posc2idx:
    length.append(len(i))
print(np.min(length))  # 12


print(images_train[0][:,:,:3])
