#! /usr/bin/python
# -*- coding: utf8 -*-
# path 찍어보기
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

#from utils import *
#from model import *
#import model
import matplotlib.pyplot as plt

###======================== PREPARE DATA ====================================###
print("Loading data from pickle ...")
import pickle
with open("_vocab.pickle", 'rb') as f:
    vocab = pickle.load(f)

with open("_image_train.pickle", 'rb') as f:
    _, images_train = pickle.load(f)
# with open("_image_test.pickle", 'rb') as f:
#     _, images_test = pickle.load(f)
# with open("_n.pickle", 'rb') as f:
#     n_captions_train, n_captions_test, n_captions_per_image, n_images_train, n_images_test = pickle.load(f)
# with open("_caption.pickle", 'rb') as f:
#     captions_ids_train, captions_ids_test = pickle.load(f)
# # images_train_256 = np.array(images_train_256)
# # images_test_256 = np.array(images_test_256)




plt.imshow(np.array(images_train[0])/255)
plt.show()
