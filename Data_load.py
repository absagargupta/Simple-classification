#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:59:40 2020

@author: sagar
"""

import cv2
import os
from PIL import Image
from numpy import asarray
import numpy as np
import glob

DIGITAL_PEN_FILE = "/home/sagar/MTech/Placement/goyal/datasets/All_digital_pens/*.jpg"
PEN_FILE = "/home/sagar/MTech/Placement/goyal/datasets/Normal_pens/*.jpg"
digi = []
for i in glob.glob(DIGITAL_PEN_FILE):
    #print(i)
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(128,128))
    digi.append(img)
digi = np.array(digi)
label_digi = np.ones(len(glob.glob(DIGITAL_PEN_FILE)))

normal_pen = []
for i in glob.glob(PEN_FILE):
    #print(i)
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(128,128))
    normal_pen.append(img)
normal_pen = np.array(normal_pen)
label_normal = np.zeros(len(glob.glob(PEN_FILE)))

total_imgs = np.concatenate((digi, normal_pen))
total_labels = np.concatenate((label_digi,label_normal))


import pickle
dbfile = open("/home/sagar/MTech/Placement/goyal/dictionaries/imgs", 'wb')
pickle.dump(total_imgs, dbfile) 
dbfile.close()
    
import pickle
dbfile = open("/home/sagar/MTech/Placement/goyal/dictionaries/labels", 'wb')
pickle.dump(total_labels, dbfile) 
dbfile.close()