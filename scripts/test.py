#!/usr/bin/env python

# General
import numpy as np
import random
import matplotlib.pyplot as plt

# ROS
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

# Keras
import keras
from keras import layers,models,losses
from keras.layers import Dense, Conv2D, Flatten, Dropout


class HumanFollowTrain:
    def __init__(self):
        self.color_image_sub = rospy.Subscriber('/camera/color/image_raw',Image,self.ColorImageCB)
        self.depth_image_sub = rospy.Subscriber('/camera/depth/image_rect_raw',Image,self.DepthImageCB)
        self.joy_input_sub   = rospy.Subscriber('/teleop_velocity_smoother/raw_cmd_vel',Twist,self.JoyInputCB)
        self.joy_pub = rospy.Publisher('/cmd_vel_mux/input/teleop',Twist,queue_size=1)

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

    def getTrainDepth(self):
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

        resized_depth_img = resized_depth_img.reshape(1,128,128,1)
        return resized_depth_img

    def getTrainJoy(self):
        joy_data = np.zeros(2).reshape(1,2)
        joy_data[0][0] = self.joy_input["x"]
        joy_data[0][1] = self.joy_input["theta"]
        return joy_data

    def main(self):
        r = rospy.Rate(1) # main loop Hz

        model = models.Sequential()
        # Model Description
        model.add(Conv2D(64, kernel_size=9, strides=(2,2), activation='relu', input_shape=(128,128,1)))
        model.add(Dropout(0.2))
        model.add(Conv2D(32, kernel_size=5, strides=(2,2), activation='relu'))
        model.add(Dropout(0.2))
        model.add(Conv2D(16, kernel_size=3, strides=(2,2), activation='relu'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(20,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(2))
        model.load_weights("3c1d_adam_00001_mse.h5") 

        while not rospy.is_shutdown():
            r.sleep()
            # get and append data
            depth_data = self.getTrainDepth()
            joy_data = self.getTrainJoy()
            vel = model.predict(depth_data)
            cmd_vel = Twist()
            cmd_vel.linear.x = vel[0][0]
            cmd_vel.angular.z = vel[0][1]
            print cmd_vel
            self.joy_pub.publish(cmd_vel)

if __name__ == '__main__':
    rospy.init_node('human_follow_train',anonymous=True)
    human_follow_train = HumanFollowTrain()
    human_follow_train.main()

