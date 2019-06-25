#!/usr/bin/env python

# General
import numpy as np
import matplotlib.pyplot as plt
import random

# Keras
# Define Model

# Load Training Data
train_depth = np.load("./train_depth.npy")
train_joy   = np.load("./train_joy.npy")
test_depth  = np.load("./test_depth.npy")
test_joy    = np.load("./test_joy.npy")

print train_depth.shape
for i in range(10):
    image = train_depth[i]
    image = np.reshape(image,(128,128)) 
    print image.shape
    plt.imshow(image,cmap="gray")
    plt.show()

"""
print a_train_depth.shape
print b_train_depth.shape

train_depth = np.concatenate([a_train_depth, b_train_depth],0)
train_joy   = np.concatenate([a_train_joy, b_train_joy],0)
test_depth  = np.concatenate([a_test_depth, b_test_depth],0)
test_joy    = np.concatenate([a_test_joy, b_test_joy],0)

print train_depth.shape
sort_num = range(384)
random.shuffle(sort_num)
train_depth = train_depth[[sort_num],:]
train_depth = np.reshape(train_depth,(384,128,128,1))
train_joy   = train_joy[[sort_num],:]
train_joy   = np.reshape(train_joy,(384,2))
test_depth = test_depth[[sort_num],:]
test_depth = np.reshape(test_depth,(384,128,128,1))
test_joy   = test_joy[[sort_num],:]
test_joy   = np.reshape(test_joy,(384,2))

np.save('train_depth.npy',train_depth)
np.save('train_joy.npy',train_joy)
np.save('test_depth.npy',test_depth)
np.save('test_joy.npy',test_joy)
"""
