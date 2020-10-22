#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:41:56 2020

@author: absagargupta
"""
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import pickle
import numpy as np
from keras.layers import Activation, LeakyReLU
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sb
#name = "CNN_basic_1.0.0.1"
import os
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

def model_archi(num_classes,krs,input_shape):
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=krs,
                     input_shape=input_shape))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))
    
    # model.add(Conv2D(64, krs))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU(alpha=0.05))
    # model.add(Dropout(0.25))
    
    # model.add(Conv2D(128, krs))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.25))
    
    model.add(Flatten())
    
    # model.add(Dense(1024))
    # model.add(LeakyReLU(alpha=0.05))
    # model.add(Dropout(0.5))
    
    # model.add(Dense(512))
    # model.add(LeakyReLU(alpha=0.05))
    # model.add(Dropout(0.5))
    
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    return model
    
    
    