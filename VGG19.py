# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 14:34:15 2020

@author: rohan
"""

from tensorflow.keras.applications.vgg19 import VGG19

model=VGG19(weights="imagenet")

model.save('vgg19.h5')