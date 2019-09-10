#!/usr/bin/env python

# General
import numpy as np
import random

# ROS
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

# Const definision
DATA_LENGTH = 1000

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
        return resized_depth_img

    def getTrainJoy(self):
        joy_data = np.zeros(2).reshape(1,2)
        joy_data[0][0] = self.joy_input["x"]
        joy_data[0][1] = self.joy_input["theta"]
        return joy_data

    def main(self):
        r = rospy.Rate(10) # main loop Hz
        img_num = 0
        # create first black image
        all_depth = np.zeros(128*128).reshape(1,128,128)
        all_joy   = np.zeros(2).reshape(1,2)
        while not rospy.is_shutdown() and img_num < DATA_LENGTH:
            r.sleep()
            img_num = img_num + 1
            # get and append data
            depth_data = self.getTrainDepth()
            joy_data = self.getTrainJoy()
            all_depth = np.append(all_depth,depth_data,axis=0)
            all_joy   = np.append(all_joy,joy_data,axis=0)
            print img_num
        # delete first black image
        train_depth = np.delete(all_depth,0,axis=0)
        train_joy   = np.delete(all_joy,0,axis=0)
        # save each data
        np.save('all_depth.npy',train_depth)
        np.save('all_joy.npy',train_joy)
        print "training data is saved!"

if __name__ == '__main__':
    rospy.init_node('human_follow_train',anonymous=True)
    human_follow_train = HumanFollowTrain()
    human_follow_train.main()

