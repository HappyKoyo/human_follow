#!/usr/bin/env python

# General
import numpy as np
import matplotlib.pyplot as plt

# Keras
# Define Model
import keras
from keras import layers,models,losses
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.layers.normalization import BatchNormalization

# Constant
EPOCHS = 300 

# Initial Setting
model = models.Sequential()

# Model Description
# Conv1 128 -> 60
model.add(Conv2D(64, kernel_size=9, strides=(2,2), activation='relu', input_shape=(128,128,1)))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())
# Conv2 60 -> 28
model.add(Conv2D(32, kernel_size=5, strides=(2,2), activation='relu'))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())
# Conv3 28 -> 13
model.add(Conv2D(16, kernel_size=3, strides=(2,2), activation='relu'))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())
# Dence1 13*13 -> 20
model.add(Flatten())
model.add(Dense(20,activation="relu"))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())
# Dence2 20-> 2
model.add(Dense(2))

# Optimze Manner
model.compile(optimizer='adam',
              loss=losses.mean_squared_error,
              metrics=['accuracy'])
keras.optimizers.Adam(lr=0.3,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0)

#loss=losses.mean_absolute_percentage_error,

# Load Training Data
train_depth = np.load("train_depth.npy")
train_joy   = np.load("train_joy.npy")
test_depth  = np.load("test_depth.npy")
test_joy    = np.load("test_joy.npy")
print train_depth.shape
print train_joy.shape

# train model
hist = model.fit(train_depth, train_joy,
                 batch_size=64,
                 verbose=1,
                 epochs=EPOCHS,
                 validation_split=0.2)

model.save_weights("new_weight.h5")
score = model.evaluate(test_depth,test_joy,verbose=0)
print("Test loss:",score[0])
print("Test accuracy:",score[1])
print hist.history["acc"]
print hist.history["val_acc"]

plt.plot(range(1,EPOCHS+1),hist.history["acc"],label="training")
plt.plot(range(1,EPOCHS+1),hist.history["val_acc"],label="valication")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.show()

plt.plot(range(1,EPOCHS+1),hist.history["loss"],label="training")
plt.plot(range(1,EPOCHS+1),hist.history["val_loss"],label="valication")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()
