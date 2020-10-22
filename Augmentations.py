#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 22:00:07 2020
Here in this code we are doing the augmentation of existing images. 
The current augmentations listed are 
flip
gaussian blur
sharpening
scaling
LinearContrast
Shear
Brightness
Perspective Transform 
Gaussian Noise
Impulse Noise
Sigmoid Contrast
Uncomment the ones you wish to use from their function and their list in augmentations

@author: absagargupta
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import cv2
import pickle
import random
from scipy.ndimage import rotate
import imgaug.augmenters as iaa
import random



# images = np.load("/home/absagargupta/MTech/Project/Expressions/pain/painful/augmentations/Xtrain.npy")
# label = np.load("/home/absagargupta/MTech/Project/Expressions/pain/painful/augmentations/ytrain.npy")
def augmentations(input_features, label):
    def flip(img):
            img = cv2.flip(img,1)
            return img
        
    # def anticlockwiserotation(image):
    #     angle = 90
        
    #     scale = 1.0
    #     h,w,_ = image.shape
    #     center = (h/2,w/2)
    #     M = cv2.getRotationMatrix2D(center, angle, scale)
    #     rotated = cv2.warpAffine(image, M, (w, h))
    #     return rotated
    
    
    def Gaussianblur(image):
        img = cv2.blur(image, ksize = (10,10))
        return img
    
    def sharpening(img):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel)
        return img
    
    def scaling(img):
        return cv2.resize(img, (128,128))
    
    # def LinearContrast(image):
    #     alpha = 3
    #     aug = iaa.LinearContrast(alpha = alpha, per_channel=True)
    #     image_aug = aug.augment_image(image)
    #     return image_aug
        
    # def Shear(image):
    #     alpha =3
    #     aug = iaa.Affine(shear=(alpha))
    #     image_aug = aug.augment_image(image)
    #     return image_aug    
    
    # shear_factor = 0.2
    # shear_factor = (-(shear_factor), shear_factor)
    
    # def Brightness(image):
    #     alpha = 3
    #     aug = iaa.AddToBrightness((alpha))
    #     image_aug = aug.augment_image(image)
    #     return image_aug
    
    # def PerspectiveTransform(image):
    #     scale = 2
    #     aug = iaa.PerspectiveTransform(scale=scale, cval=0, mode='constant', keep_size=True, fit_output=True)
    #     image_aug = aug.augment_image(image)
    #     return image_aug
    
    # def GaussianNoise(image):
    #     severity = 2
    #     aug = iaa.imgcorruptlike.GaussianNoise(severity=severity)
    #     image_aug = aug.augment_image(image)
    #     return image_aug
    
    # def ImpulseNoise(image):
    #     severity = 2
    #     aug = iaa.imgcorruptlike.ImpulseNoise(severity=severity)
    #     image_aug = aug.augment_image(image)
    #     return image_aug
    
    # def SigmoidContrast(image):
    #     cutoff = 1
    #     gain = 4
    #     aug = iaa.SigmoidContrast(gain=gain, cutoff=cutoff, per_channel = True)
    #     image_aug = aug.augment_image(image)
    #     return image_aug
    
    
    augmentations = [flip, Gaussianblur, sharpening, scaling]#, Shear, Brightness, PerspectiveTransform, GaussianNoise, ImpulseNoise, SigmoidContrast]
    
    
    
    new_list_image = []
    new_list_label = []
    input_features = np.squeeze(np.array(input_features))
    
    
        
    for i in range(input_features.shape[0]):
        rn = random.uniform(0, 1)
        if rn > 0.7:
            image_to_be_appended = random.choice(augmentations)(input_features[i])
            new_list_image.append(image_to_be_appended)
            new_list_label.append(label[i])
            
            # Uncomment the below code if you wish to increase the number of images and take the normal image along with augmentations
            image_to_be_appended = input_features[i]
            new_list_image.append(image_to_be_appended)
            new_list_label.append(label[i])
            use_augmentation = True
            
        else:
            image_to_be_appended = input_features[i]
            new_list_image.append(image_to_be_appended)
            new_list_label.append(label[i])
            
    images_augmented = np.array(new_list_image)
    labels_augmented = np.array(new_list_label)
    return images_augmented, labels_augmented
