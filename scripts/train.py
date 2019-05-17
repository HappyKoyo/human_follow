#!/usr/bin/env python

# General
import numpy as np

# ROS
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

# Keras
# Define Model
from keras import layers,models
from keras.layers.core import Dense
model = models.Sequential()

# Expand Model using Dense

model.add(Dense(64, activation='relu', input_shape=(64 * 64,)))
model.add(Dense(10, activation='softmax'))

# 
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#from keras.models

class HumanFollowTrain:
    def __init__(self):
        self.color_image_sub = rospy.Subscriber('/camera/color/image_raw',Image,self.ColorImageCB)
        self.depth_image_sub = rospy.Subscriber('/camera/depth/image_rect_raw',Image,self.DepthImageCB)
        self.joy_input_sub   = rospy.Subscriber('/teleop_velocity_smoother/raw_cmd_vel',Twist,self.JoyInputCB)

        self.color_img = ()
        self.depth_img = ()
        self.joy_input = {"x":0, "theta":0}
        self.bridge = CvBridge()
        
    def ColorImageCB(self,msg):
        try:
            self.color_img = self.bridge.imgmsg_to_cv2(msg)
        except CvBridgeError as error_msg:
            print(error_msg)

    def DepthImageCB(self,msg):
        try:
            self.depth_img = self.bridge.imgmsg_to_cv2(msg)
        except CvBridgeError as error_msg:
            print(error_msg)
    
    def JoyInputCB(self,msg):
        self.joy_input["x"]     = msg.linear.x
        self.joy_input["theta"] = msg.angular.z

    def genTrainData(self):
        # convert image to (float64, 1*128*128)
        resized_depth_img = cv2.resize(self.depth_img,dsize=(128,128))
        resized_depth_img = resized_depth_img.astype(np.float64)
        resized_depth_img = resized_depth_img.reshape(1,128,128)
        # scaling depth from 0 to 1
        for h_i in range(128):
            for w_i in range(128):
                if resized_depth_img[0][h_i][w_i] < 1 or 3000 < resized_depth_img[0][h_i][w_i]:
                    resized_depth_img[0][h_i][w_i] = 0
                else:
                    resized_depth_img[0][h_i][w_i] = 1-float(resized_depth_img[0][h_i][w_i]-30)/2980
        return resized_depth_img

    def main(self):
        r = rospy.Rate(10) # main loop Hz
        img_num = 0
        train_depth_imgs = np.zeros(128*128).reshape(1,128,128)
        while not rospy.is_shutdown() and img_num < 256:
            r.sleep()
            img_num = img_num + 1
            # generate traindata from self class depth image
            train_data = self.getTrainDepth()
            joy_data = self.getTrainJoy()
            # append datasets
            train_depth_imgs = np.append(train_depth_imgs,train_data,axis=0)
            print img_num
        # save training data
        np.save('depth_data.npy',train_depth)
        np.save('joy_data.npy',train_joy)

if __name__ == '__main__':
    rospy.init_node('human_follow_train',anonymous=True)
    human_follow_train = HumanFollowTrain()
