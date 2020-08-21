#!/usr/bin/env python3

import os
# import xml.dom.minidom
import cv2
import sys
import time
import threading

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Vector3

import global_config
from config import *
from model import LaneNet
from utils.transforms import *
from utils.postprocess import *
import torch.backends.cudnn as cudnn
from utils.prob2lines import getLane
from utils.lanenet_postprocess import LaneNetPostProcessor
# import matplotlib.pyplot as plt
from Detection import Detection

from lane_obj import Lane

CFG = global_config.cfg

g_frameCnt = 0
g_videoPlay = True
g_keyboardinput = ''
g_writevideo = False

class Lane_warning:
    def __init__(self):
        self.image_pub = rospy.Publisher("lanedetframe", Image,queue_size = 1)
        self.maskimg_pub = rospy.Publisher("lanedetmask", Image,queue_size = 1)
        self.binimg_pub = rospy.Publisher("lanedetbin", Image,queue_size = 1)
        self.morphoimg_pub = rospy.Publisher("lanedetmorph", Image,queue_size = 1)
        # self.bridge = CvBridge()
        # self.yolo_result = rospy.Subscriber("YOLO_detect_result", Float64MultiArray, self.callbackyolo)
        # self.image_sub = rospy.Subscriber("YOLO_detect_result", Image, self.callbackRos)
        self.image_sub = rospy.Subscriber("/camera/image", Image, self.callbackRos)
        self.velo_sub = rospy.Subscriber("velocity", Vector3, self.veloCallback)

        # self.image_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image，queue_size=1, buff_size=110592*6)
        # self.weights_file = '/home/no1/code/lane/src/lanedet/lane_waring_final/experiments/exp1/exp1_best.pth'
        self.weights_file = '/space/code/lane/src/lanedet/lane_waring_final/experiments/exp1/exp1_best.pth'
        # self.weights_file = '/space/code/lane/src/lanedet/lane_waring_final/experiments/exp0/exp0_best.pth'
        self.CUDA = torch.cuda.is_available()
        self.postprocessor = LaneNetPostProcessor()
        self.warning = Detection()
        self.band_width = 1.5
        self.image_X = CFG.IMAGE_WIDTH
        self.image_Y = CFG.IMAGE_HEIGHT
        self.car_X = self.image_X/2
        self.car_Y = self.image_Y
        self.model = LaneNet(pretrained=False, embed_dim=4, delta_v=.5, delta_d=3.)
        self.save_dict = torch.load(self.weights_file, map_location='cuda:0')
        self.model.load_state_dict(self.save_dict['net'])
        # self.model.load_state_dict(torch.load(self.weights_file, map_location='cuda:0'))
        if self.CUDA: self.model.cuda()
        self.model.set_test()
        self.lastlane = np.ndarray(4,)
        self.bridge = CvBridge()

        self.leftlane = Lane('left')
        self.rightlane = Lane('right')

        # self.out = cv2.VideoWriter('testwrite.avi',cv2.VideoWriter_fourcc(*'MJPG'), 15.0, (CFG.IMAGE_WIDTH, CFG.IMAGE_HEIGHT),True)

    def transform_input(self, img):
        _set = "IMAGENET"
        mean = IMG_MEAN[_set]
        std = IMG_STD[_set]
        # transform_img = Resize((800, 288))
        transform_img = Resize((512, 256))
        transform_x = Compose(ToTensor(), Normalize(mean=mean, std=std))
        #img_org = img[255:945, :, :]
        img = transform_img({'img': img})['img']
        x = transform_x({'img': img})['img']
        # print(x)
        x.unsqueeze_(0)
        x = x.to('cuda')
        return x

    def detection(self, input):

        #startt = time.time()
        if self.CUDA:
            input = input.cuda()
        with torch.no_grad():
            output = self.model(input, None)

       # print('detection use：', time.time()-startt)
        return self.cluster(output)

    def cluster(self,output):
        #startt = time.time()

        global g_frameCnt

        embedding = output['embedding']
        embedding = embedding.detach().cpu().numpy()
        embedding = np.transpose(embedding[0], (1, 2, 0))
        binary_seg = output['binary_seg']
        bin_seg_prob = binary_seg.detach().cpu().numpy()
        bin_seg_pred = np.argmax(bin_seg_prob, axis=1)[0]
        # seg = bin_seg_pred * 255

        postprocess_result = self.postprocessor.postprocess(
            binary_seg_result=bin_seg_pred,
            instance_seg_result=embedding)

        if postprocess_result['mask_image'] is None:
            print('cant find any lane!!!')
        else:
            self.maskimg_pub.publish(self.bridge.cv2_to_imgmsg(postprocess_result['mask_image'], "bgr8"))
            self.binimg_pub.publish(self.bridge.cv2_to_imgmsg(postprocess_result['binary_img'], "mono8"))
            self.morphoimg_pub.publish(self.bridge.cv2_to_imgmsg(postprocess_result['morpho_img'], "mono8"))

        # cv2.imwrite(str(g_frameCnt)+'_mask.png', postprocess_result['mask_image'])
        # cv2.imwrite(str(g_frameCnt)+'_binary.png', postprocess_result['binary_img'])

        return postprocess_result


    def color(self, signal):
        if signal == 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        return color

    def process(self, frame):
        cropImg = cropRoi(frame)
        input_image = self.transform_input(cropImg)
        startt = time.time()
        postProcResult = self.detection(input_image)
        
        # cv2.imwrite('cropImg.png', cropImg)
        debugImg = frame.copy()

        # if len(postProcResult['fit_params']) > 0:
        self.leftlane.updateLane(postProcResult['fit_params'])
        self.rightlane.updateLane(postProcResult['fit_params'])
        if self.leftlane.detectedLostCnt > 3 and self.rightlane.detectedLostCnt > 3:
            self.leftlane.initLane()
            self.rightlane.initLane()
            print('!!!!!!! detected not fit')
        lanePoints = {
            'lanes':[self.leftlane.points,self.rightlane.points]
        }
        signal = self.warning.detect(lanePoints)
        color = (0, 0, 255) if signal == 1 else (0, 255, 0)
        color = (0, 255, 0)
        #draw lane
        for idx in range(11):
            cv2.line(frame, (int(self.leftlane.points[idx][0]), int(self.leftlane.points[idx][1])), (int(self.leftlane.points[idx+1][0]), int(self.leftlane.points[idx+1][1])), color, 10)
            cv2.line(frame, (int(self.rightlane.points[idx][0]), int(self.rightlane.points[idx][1])), (int(self.rightlane.points[idx+1][0]), int(self.rightlane.points[idx+1][1])), color, 10)

        print('all use：', time.time()-startt)

        # debug
        plot_y = np.linspace(CFG.LANE_START_Y, CFG.LANE_END_Y, 12)
        for fit_param in postProcResult['fit_params']:
            fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2]
            
            if self.leftlane.detectedLeftLane[0][0] == int(fit_x[0]):
                color = (255,0,0)
            elif self.leftlane.detectedRightLane[0][0] == int(fit_x[0]):
                color = (255,255,0)
            else:
                color = (0,255,0)

            for j in range(11):
                cv2.line(debugImg, (int(fit_x[j]), int(plot_y[j])), (int(fit_x[j+1]), int(plot_y[j+1])), color, 10)
                cv2.line(debugImg, (0,int(plot_y[j])), (int(self.image_X),int(plot_y[j])), (0,0,255), 3)
            cv2.putText(debugImg, '{}'.format(abs(int(fit_x[5])-960)), (int(fit_x[0]), int(plot_y[0])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness=2)
            
        # cv2.imwrite(str(g_frameCnt)+'debug.png', debugImg)
        # cv2.imwrite(str(g_frameCnt)+'input_image.png', frame)

        cv2.imwrite(str(1)+'debug.png', debugImg)
        cv2.imwrite(str(1)+'input_image.png', cropImg)
        # debug end

    def callbackRos(self, data):
        print('callbackros')

        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        # self.out.write(cv_image)

        self.process(cv_image)
        
        
        # cv2.imwrite('result.png', cv_image)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

    def veloCallback(self, data):
        print('velo callback',data.vx,data.vy,data.vz)

def listenkeyboard():
    global g_videoPlay, g_keyboardinput, g_writevideo
    while True:
        g_keyboardinput = input()
        if g_keyboardinput == 'a':
            g_videoPlay = not g_videoPlay
        elif g_keyboardinput == 'r':
            print('g_keyboardinput rrrrrr')
            g_writevideo = True

def test():
    global g_videoPlay, g_keyboardinput, g_writevideo, g_frameCnt
    ic = Lane_warning()
    rospy.init_node("lanedetnode", anonymous=True)
    cap = cv2.VideoCapture('/space/data/xq/cross/5.avi')
    ret, frame = cap.read()
    rate = rospy.Rate(10)
    rospy.loginfo('video frame cnt:%d', cap.get(cv2.CAP_PROP_FRAME_COUNT))

    listernerTh = threading.Thread(target=listenkeyboard)
    listernerTh.setDaemon(True)
    listernerTh.start()

    # out = cv2.VideoWriter('testwrite.avi',cv2.VideoWriter_fourcc(*'MJPG'), 15.0, (1920,1200),True)
    
    while not rospy.is_shutdown():
        while not g_videoPlay:
            time.sleep(1)
            if g_keyboardinput == 's':
                g_keyboardinput = ''
                break

        ret, frame = cap.read()
        if not ret:
            rospy.loginfo('frame end')
            break

        g_frameCnt = g_frameCnt + 1
        rospy.loginfo('frame cnt:%d', g_frameCnt)
        cv_image = frame.copy()

        # # if g_writevideo:
        # #     print('write video')
        # #     out.write(cv_image)

        ic.process(cv_image)
        ic.image_pub.publish(ic.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        rate.sleep()

    # out.release()

def cropRoi(img):

    # return img[int(vanishPtY-cropedImgHeight/2):int(vanishPtY+cropedImgHeight/2), int(vanishPtX-cropedImgWidth/2):int(vanishPtX+cropedImgWidth/2), :].copy()

    return img[CFG.CROP_IMG_Y:CFG.CROP_IMG_Y+CFG.CROP_IMG_HEIGHT, CFG.CROP_IMG_X:CFG.CROP_IMG_X+CFG.CROP_IMG_WIDTH, :].copy()

def main(args):
    ic = Lane_warning()
    rospy.init_node("lanedetnode", anonymous=True)
    rospy.loginfo('model init end')
    rospy.spin()


if __name__ == "__main__":
    # main(sys.argv)
    test()








