#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 19:22:57 2020

@author: sagar
"""
#%% Augmentations
from Augmentations import augmentations
from sklearn.model_selection import train_test_split

import pickle
import keras
num_classes = 2

dbfile = open("/home/sagar/MTech/Placement/goyal/dictionaries/imgs", 'rb')
xtr = pickle.load(dbfile)
dbfile.close()

dbfile = open("/home/sagar/MTech/Placement/goyal/dictionaries/labels", 'rb')
ytr = pickle.load(dbfile)
dbfile.close()
Xtrain = xtr
ytrain = ytr
y = keras.utils.to_categorical(ytrain, num_classes)
Xtrain, ytrain = augmentations(Xtrain, y)

Xtrain,Xtest,ytrain,ytest = train_test_split(Xtrain,ytrain,shuffle = True)

#%% CNN
import os
import tensorflow as tf
import gc
from model_archi import model_archi
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
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import pandas as pd
patience = 5
krs = (3,3)
input_shape = (128,128,1)
unique = np.unique(ytrain)
#%% choosing model
class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
#%%
model = model_archi(num_classes,krs,input_shape)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

earlystopper= EarlyStopping(patience=patience, verbose=1)
logger= keras.callbacks.CSVLogger("/home/sagar/MTech/Placement/goyal/logs/hi"+'.log', separator=',', append=True)

#%% fitting model
Xtrain = np.expand_dims(Xtrain,axis = -1)
Xtest = np.expand_dims(Xtest,axis = -1)
history = model.fit(Xtrain, ytrain,
          batch_size=4,
          epochs=20,
          verbose=1,
          validation_data=(Xtest, ytest), callbacks = [earlystopper, logger, MyCustomCallback()])

#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
#print(history.history.keys())
# summarize history for accuracy

result = model.predict(Xtest)
#plt.imsave("/home/sagar/MTech/Project/Expressions/pain/painful/models/"+  name +".jpg")
ty = unique[ytest.argmax(1)]
ry = unique[result.argmax(1)]
# evaluation = model.evaluate(ry,ty)
print(confusion_matrix(ty,ry))
print(classification_report(ty, ry))
model.save("/home/sagar/MTech/Placement/goyal/models/hi")
cm1 = np.array([0,1])
cm = confusion_matrix(ty, ry)
cm_new = np.append(cm1,cm).reshape(3,2)
pd.DataFrame(cm_new).to_csv("/home/sagar/MTech/Placement/goyal/logs/hi"+"_Confusion_matrix.csv", header=None, index=False)
cr = precision_recall_fscore_support(ty,ry)
cr_new = np.array(cr).T
cr = precision_recall_fscore_support(ty,ry)
cr_new = np.array(cr).T
cr_names = np.array(["Precision","Recall","F1","Support"])
cr_new = np.append(cr_names,cr_new).reshape(4,3)
pd.DataFrame(cr_new).to_csv("/home/sagar/MTech/Placement/goyal/logs/hi"+"_Precision_recall_f_score_support.csv", header=None, index=False)



