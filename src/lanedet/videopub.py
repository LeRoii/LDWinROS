#!/usr/bin/env python3
#coding:utf-8

import rospy
import sys
sys.path.append('.')
import cv2
import os
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

videoPath = '/space/data/hk/1.avi'

def pubVideo():
    rospy.init_node('videopub',anonymous = True)
    pub = rospy.Publisher('/camera/image', Image, queue_size = 1)
    rate = rospy.Rate(10)
    bridge = CvBridge()
    cap = cv2.VideoCapture(videoPath)
    ret, frame = cap.read()
    cnt = 0
    rospy.loginfo('video frame cnt:%d', cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.loginfo('frame end')
            break
        pub.publish(bridge.cv2_to_imgmsg(frame,"bgr8"))
        rospy.loginfo('frame cnt:%d', cnt)
        cnt = cnt+1
        # cv2.imshow("lala",frame)
        # cv2.waitKey(0)
        # if cnt == 224:
        # str = input()
        # if str == 'a':
        #     pass
        rate.sleep()



def pubImg():
    image_path = '/home/iairiv/data/kitti'
    imagedir = os.listdir(image_path)
    imagedir.sort()
    for i in imagedir:
        #print(i)
        image = os.path.join(image_path,i)
        image = cv2.imread(image)

    rospy.init_node('videopub',anonymous = True)
    pub = rospy.Publisher('/camera/rgb/image_raw', Image, queue_size = 10)
    rate = rospy.Rate(5)
    bridge = CvBridge()

    imgNum = len(imagedir)
    i=0
    while not rospy.is_shutdown():
        imagePath = os.path.join(image_path,imagedir[i])
        image = cv2.imread(imagePath)
        image = image[:,:,0:800]
        image = cv2.resize(image,(1280,720))
        # cv2.imshow("lala",image)
        # cv2.waitKey(0)
        i = i+1
        if i >= imgNum:
            rospy.loginfo('frame end')
            break
        pub.publish(bridge.cv2_to_imgmsg(image,"bgr8"))
        rospy.loginfo('frame cnt:%d', i)
        #cnt = cnt+1
        # cv2.imshow("lala",image)
        # cv2.waitKey(0)
        rate.sleep()


if __name__ == '__main__':
    try:
        pubVideo()
    except rospy.ROSInterruptException:
        pass
