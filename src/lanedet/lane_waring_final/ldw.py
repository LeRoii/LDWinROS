#!/usr/bin/env python3

import os
# import xml.dom.minidom
import cv2
import sys
import time
import threading

# import rospy
# from sensor_msgs.msg import Image
# from std_msgs.msg import String
# from cv_bridge import CvBridge, CvBridgeError
# from std_msgs.msg import Float64MultiArray

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

CFG = global_config.cfg

g_frameCnt = 0
g_videoPlay = True
g_keyboardinput = ''
g_writevideo = False

class Lane:
    def __init__(self, laneIdx):
        self.age = 0
        self.points = np.zeros([12,2], dtype = int)
        self.image_X = 1920
        self.laneIdx = laneIdx
        self.isInit = False
        self.lostCnt = 0

        self.detectedLeftLane = np.zeros([12,2], dtype = int)
        self.detectedRightLane = np.zeros([12,2], dtype = int)

        self.detectedLostCnt = 0

    def reset(self):
        self.points = np.zeros([12,2], dtype = int)
        self.isInit = False
        self.lostCnt = 0
        self.age = 0

        print('\n{} lane reset'.format(self.laneIdx))
        return True

    def isLost(self):
        self.lostCnt = self.lostCnt+1
        if self.age > 30:
            if self.lostCnt > 6:
                return self.reset()
        elif self.age > 10:
            if self.lostCnt > 4:
                return self.reset()
        elif self.lostCnt > 3:
                return self.reset()
        return  False

    def findDetectedLeftRightLane(self, lane_x_coords):
        

        if len(lane_x_coords) >= 2:
            sorted_lane_x_coords = sorted(lane_x_coords[:2],key=(lambda x : x[5]))

            self.detectedLeftLane[:,0] = sorted_lane_x_coords[0]
            self.detectedRightLane[:,0] = sorted_lane_x_coords[1]
            self.detectedLeftLane[:,1] = self.detectedRightLane[:,1] = np.linspace(CFG.LANE_START_Y, 1100, 12)

        else:
            if lane_x_coords[0][0] < self.image_X/2:
                self.detectedLeftLane[:,0] = lane_x_coords[0]
                self.detectedLeftLane[:,1] = np.linspace(CFG.LANE_START_Y, 1100, 12)
            else:
                self.detectedRightLane[:,0] = lane_x_coords[0]
                self.detectedRightLane[:,1] = np.linspace(CFG.LANE_START_Y, 1100, 12)

    def initLane(self):
        self.points = self.detectedLeftLane.copy()  if self.laneIdx == 'left' else self.detectedRightLane.copy()
        self.isInit = True
        self.age = self.age+1

    def updateLane(self, fit_params):
        plot_y = np.linspace(CFG.LANE_START_Y, 1100, 12)
        lane_x_coords = []
        # detectedLeftLaneX = [0] * 12
        # detectedRightLaneX = [0] * 12

        for fit_param in fit_params:
            fitx = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2]
            # if abs(fitx[5] - self.image_X/2) > 550:
            #     continue
            lane_x_coords.append(fitx.astype(np.int))

        lane_x_coords.sort(key=(lambda x : abs(x[5] - self.image_X/2)))
        self.findDetectedLeftRightLane(lane_x_coords)
                
        if not self.isInit:
            self.initLane()
        else:
            detectedPoints = self.points.copy()
            minDist = 500
            bestFit = -1
            for idx in range(len(lane_x_coords)):
                # if lane_x_coords[idx][0] == (self.detectedRightLane[0][0] if self.laneIdx == 'left' else self.detectedLeftLane[0][0]):
                #     continue
                detectedPoints[:,0] = lane_x_coords[idx]
                dist = np.linalg.norm(self.points - detectedPoints)
                print('{}, dist:{}'.format(self.laneIdx,dist))
                if dist < minDist:
                    bestFit = idx
                    minDist = dist

            if bestFit == -1:
                if self.isLost():
                    self.initLane()
            else:
                # if self.age < 10 and lane_x_coords[bestFit][0] != (detectedLeftLane[0][0] if self.laneIdx == 'left' else detectedRightLane[0][0]):
                #     self.initLane(detectedLeftLane, detectedRightLane)
                # else:
                #     self.lostCnt = 0
                #     self.age = self.age+1
                #     detectedPoints[:,0] = lane_x_coords[bestFit]
                #     self.points[:,0] = np.average([self.points[:,0], detectedPoints[:,0]], axis=0, weights=[0.8,0.2])
                self.lostCnt = 0
                self.age = self.age+1
                detectedPoints[:,0] = lane_x_coords[bestFit]
                self.points[:,0] = np.average([self.points[:,0], detectedPoints[:,0]], axis=0, weights=[0.9,0.1])

                if int(lane_x_coords[bestFit][0]) == (self.detectedLeftLane[0][0] if self.laneIdx == 'left' else self.detectedRightLane[0][0]):
                    self.detectedLostCnt = 0
                else:
                    self.detectedLostCnt = self.detectedLostCnt + 1

                if self.detectedLostCnt > 5:
                    self.initLane()

        print('\n{} lane, age:{}, lost cnt:{}, detectedLostCnt:{}'.format(self.laneIdx, self.age, self.lostCnt, self.detectedLostCnt))


class Lane_warning:
    def __init__(self):
        # self.image_pub = rospy.Publisher("lanedetframe", Image,queue_size = 1)
        # self.maskimg_pub = rospy.Publisher("lanedetmask", Image,queue_size = 1)
        # self.binimg_pub = rospy.Publisher("lanedetbin", Image,queue_size = 1)
        # self.morphoimg_pub = rospy.Publisher("lanedetmorph", Image,queue_size = 1)
        # self.bridge = CvBridge()
        # self.yolo_result = rospy.Subscriber("YOLO_detect_result", Float64MultiArray, self.callbackyolo)
        # self.image_sub = rospy.Subscriber("YOLO_detect_result", Image, self.callbackRos)
        # self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callbackRos)
        # self.image_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image，queue_size=1, buff_size=110592*6)
        self.weights_file = '/space/code/roslane/src/lanedet/lane_waring_final/experiments/exp1/exp1_best.pth'
        self.CUDA = torch.cuda.is_available()
        self.postprocessor = LaneNetPostProcessor()
        self.warning = Detection()
        self.band_width = 1.5
        self.image_X = 1920
        self.image_Y = 1200
        self.car_X = self.image_X/2
        self.car_Y = self.image_Y
        self.model = LaneNet(pretrained=False, embed_dim=4, delta_v=.5, delta_d=3.)
        self.save_dict = torch.load(self.weights_file, map_location='cuda:0')
        self.model.load_state_dict(self.save_dict['net'])
        # self.model.load_state_dict(torch.load(self.weights_file, map_location='cuda:0'))
        if self.CUDA: self.model.cuda()
        self.model.set_test()
        self.lastlane = np.ndarray(4,)
        # self.bridge = CvBridge()

        self.leftlane = Lane('left')
        self.rightlane = Lane('right')

    def transform_input(self, img):
        return prep_image(img)

    def detection(self, input, raw):

        #startt = time.time()
        if self.CUDA:
            input = input.cuda()
        with torch.no_grad():
            output = self.model(input, None)

       # print('detection use：', time.time()-startt)


        return self.cluster(output,raw)

    def cluster(self,output,raw):
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
            instance_seg_result=embedding,
            source_image=raw
        )
        # prediction = postprocess_result['points']
        # prediction = np.array(prediction)
        #print('cluster use：', time.time()-startt)

        # self.maskimg_pub.publish(self.bridge.cv2_to_imgmsg(postprocess_result['mask_image'], "bgr8"))
        # self.binimg_pub.publish(self.bridge.cv2_to_imgmsg(postprocess_result['binary_img'], "mono8"))
        # self.morphoimg_pub.publish(self.bridge.cv2_to_imgmsg(postprocess_result['morpho_img'], "mono8"))

        # cv2.imwrite(str(g_frameCnt)+'_mask.png', postprocess_result['mask_image'])
        # cv2.imwrite(str(g_frameCnt)+'_binary.png', postprocess_result['binary_img'])

        return postprocess_result

    def write(self, output, img, signal, color):
        # output[:,:,1] = output[:,:,1]+255
        # for i in range(len(output)):
        #     line = np.array(output[i])
        #     line[:,1] = line[:,1]+255
        #     output[i] = line.tolist()
        #     print('point num:',len(output[i]))
        #     for j in range(len(output[i])):
        #         #output[i][j][1] = output[i][j][1] + 255
        #         #print(output[i][j])
        #         cv2.circle(img, (int(output[i][j][0]),int(output[i][j][1])), 5, color, -1) #画成小圆点
        #     #cv2.line(img, (int(output[i][0][0]), int(output[i][0][1])), (int(output[i][-1][0]), int(output[i][-1][1])),color, 3)
        # if signal == 1:
        #     cv2.putText(img, "WARNING", (1300, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, thickness=10)
        # #plt.imshow(img)
        # #plt.show()
        # return img

        # lane_x_coords = []
        # plot_y = np.linspace(700, 1100, 12)
        # for fit_param in output['fit_params']:
        #     fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2]
        #     if abs(fit_x[5] - self.image_X/2) > 550:
        #         continue
        #     lane_x_coords.append(fit_x)
            
        #         #cv2.line(img, (0,int(plot_y[idx])), (int(self.image_X),int(plot_y[idx])),color, 3)

        # if len(lane_x_coords) == 2:
        #     if lane_x_coords[0][0] < lane_x_coords[1][0]:
        #         leftlaneDetect = lane_x_coords[0]
        #         rightlaneDetect = lane_x_coords[1]
        #     else:
        #         leftlaneDetect = lane_x_coords[1]
        #         rightlaneDetect = lane_x_coords[0]

        # if len(self.leftlane)==0 and len(self.rightlane)==0:
        #     self.leftlane = leftlaneDetect
        #     self.rightlane = rightlaneDetect
        # else:
        #     print('self.leftlane:',self.leftlane)
        #     print('leftlaneDetect:',leftlaneDetect)
        #     self.leftlane = np.average([self.leftlane, leftlaneDetect], axis=0, weights=[0.8,0.2])
        #     self.rightlane = np.average([self.rightlane, rightlaneDetect], axis=0, weights=[0.8,0.2])
        #     print('self.leftlane:',self.leftlane)

        self.leftlane.updateLane(output['fit_params'])
        self.rightlane.updateLane(output['fit_params'])


        # for idx in range(11):
        #     cv2.line(img, (int(self.leftlane.points[idx]),int(plot_y[idx])), (int(self.leftlane[idx+1]),int(plot_y[idx+1])),color, 10)
        #     cv2.line(img, (int(self.rightlane.points[idx]),int(plot_y[idx])), (int(self.rightlane[idx+1]),int(plot_y[idx+1])),color, 10)
        
        for idx in range(11):
            cv2.line(img, (int(self.leftlane.points[idx][0]), int(self.leftlane.points[idx][1])), (int(self.leftlane.points[idx+1][0]), int(self.leftlane.points[idx+1][1])), color, 10)
            cv2.line(img, (int(self.rightlane.points[idx][0]), int(self.rightlane.points[idx][1])), (int(self.rightlane.points[idx+1][0]), int(self.rightlane.points[idx+1][1])), color, 10)
            
            
        return img




    # def draw_anchor(self,image,xmin,ymin,xmax,ymax,id):
    #
    #     result = cv.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), thickness=2)
    #     result = cv.putText(img, id, (x1, y1), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), thickness=1)
    #     try:
    #         self.image_pub.publish(self.bridge.cv2_to_imgmsg(result, "bgr8"))
    #     except CvBridgeError as e:
    #         print(e)
    #     return img

    def color(self, signal):
        if signal == 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        return color

    #ros下的代码，还没测试过。无ros用另一个测。
    # def callbackyolo(self,data):
    #     #image = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callbackRos)
    #     print('callbackyolo')
    #     print(data)
    #     #return self.yolodata(data,image)

    def yolodata(self,box,image):
        xmin = box.xmin
        ymin = box.ymin
        xmax = box.xmax
        ymax = box.ymax
        idclass = box.Class
        return self.draw_anchor(image,xmin,ymin,xmax,ymax,id)


    def callbackRos(self, data):
        print('callbackros')

        # try:
        #     cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        #     # cv_image = cv2.resize(cv_image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        #     input_image = self.transform_input(cv_image)
        #     prediction = self.detection(input_image, cv_image)
        #     if prediction[0] is None:
        #         result = cv_image
        #     else:
        #         #print(prediction)
        #         signal = self.warning.detect(prediction)
        #         color = self.color(signal)
        #         result = self.write(prediction, cv_image, signal, color)
        # except CvBridgeError as e:
        #     print(e)
        # #return result

        # # cv2.imshow("image windows", result)
        # # cv2.waitKey(3)
        # try:
        #     self.image_pub.publish(self.bridge.cv2_to_imgmsg(result, "bgr8"))
        # except CvBridgeError as e:
        #     print(e)

        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        input_image = self.transform_input(cv_image)
        startt = time.time()
        postProcResult = self.detection(input_image, cv_image)
        
        
        if len(postProcResult['fit_params']) > 0:
            self.leftlane.updateLane(postProcResult['fit_params'])
            self.rightlane.updateLane(postProcResult['fit_params'])
            # signal = self.warning.detect(prediction)
            signal = 0
            color = (0, 255, 0) if signal == 0 else (0, 0, 255)
            #draw lane
            for idx in range(11):
                cv2.line(cv_image, (int(self.leftlane.points[idx][0]), int(self.leftlane.points[idx][1])), (int(self.leftlane.points[idx+1][0]), int(self.leftlane.points[idx+1][1])), color, 10)
                cv2.line(cv_image, (int(self.rightlane.points[idx][0]), int(self.rightlane.points[idx][1])), (int(self.rightlane.points[idx+1][0]), int(self.rightlane.points[idx+1][1])), color, 10)
        
        print('all use：', time.time()-startt)
        # cv2.imwrite('result.png', cv_image)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

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
    cap = cv2.VideoCapture('/space/data/road/3.avi')
    ret, frame = cap.read()
    rate = rospy.Rate(10)
    print('video frame cnt:%d'%cap.get(cv2.CAP_PROP_FRAME_COUNT))

    listernerTh = threading.Thread(target=listenkeyboard)
    listernerTh.setDaemon(True)
    listernerTh.start()

    # out = cv2.VideoWriter('testwrite.avi',cv2.VideoWriter_fourcc(*'MJPG'), 15.0, (1920,1200),True)
    
    while 1:
        while not g_videoPlay:
            time.sleep(1)
            if g_keyboardinput == 's':
                g_keyboardinput = ''
                break

        ret, frame = cap.read()
        if not ret:
            print('frame end')
            break

        g_frameCnt = g_frameCnt + 1
        print('frame cnt:%d', g_frameCnt)
        

        cv_image = frame.copy()
        
        #cv_image = cv2.imread('/space/data/road/1.jpg')
        cropImg = cropRoi(cv_image)
        input_image = ic.transform_input(cropImg)
        startt = time.time()
        postProcResult = ic.detection(input_image, cv_image)
        
        cv2.imwrite('cropImg.png', cropImg)
        debugImg = frame.copy()

        if len(postProcResult['fit_params']) > 0:
            ic.leftlane.updateLane(postProcResult['fit_params'])
            ic.rightlane.updateLane(postProcResult['fit_params'])
            if ic.leftlane.detectedLostCnt > 3 and ic.rightlane.detectedLostCnt > 3:
                ic.leftlane.initLane()
                ic.rightlane.initLane()
                print('!!!!!!! detected not fit')
            lanePoints = {
                'lanes':[ic.leftlane.points,ic.rightlane.points]
            }
            signal = ic.warning.detect(lanePoints)
            color = (0, 0, 255) if signal == 1 else (0, 255, 0)
            color = (0, 255, 0)
            #draw lane
            for idx in range(11):
                cv2.line(cv_image, (int(ic.leftlane.points[idx][0]), int(ic.leftlane.points[idx][1])), (int(ic.leftlane.points[idx+1][0]), int(ic.leftlane.points[idx+1][1])), color, 10)
                cv2.line(cv_image, (int(ic.rightlane.points[idx][0]), int(ic.rightlane.points[idx][1])), (int(ic.rightlane.points[idx+1][0]), int(ic.rightlane.points[idx+1][1])), color, 10)
        print('all use：', time.time()-startt)
        #cv2.imwrite('result.png', cv_image)
        # ic.image_pub.publish(ic.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

        cv2.imshow('aa',cv_image)

        # if g_writevideo:
        #     print('write video')
        #     out.write(cv_image)

        plot_y = np.linspace(CFG.LANE_START_Y, 1100, 12)
        for fit_param in postProcResult['fit_params']:
            fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2]
            
            if ic.leftlane.detectedLeftLane[0][0] == int(fit_x[0]):
                color = (255,0,0)
            elif ic.leftlane.detectedRightLane[0][0] == int(fit_x[0]):
                color = (255,255,0)
            else:
                color = (0,255,0)

            for j in range(11):
                cv2.line(debugImg, (int(fit_x[j]), int(plot_y[j])), (int(fit_x[j+1]), int(plot_y[j+1])), color, 10)
                cv2.line(debugImg, (0,int(plot_y[j])), (int(ic.image_X),int(plot_y[j])), (0,0,255), 3)
            cv2.putText(debugImg, '{}'.format(abs(int(fit_x[5])-960)), (int(fit_x[0]), int(plot_y[0])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness=2)
            
        # cv2.imwrite(str(g_frameCnt)+'debug.png', debugImg)
        # cv2.imwrite(str(g_frameCnt)+'input_image.png', frame)

        cv2.imwrite(str(1)+'debug.png', debugImg)
        cv2.imwrite(str(1)+'input_image.png', cropImg)


    # out.release()

def cropRoi(img):
    vanishPtX = CFG.VANISH_POINT_X
    vanishPtY = CFG.VANISH_POINT_Y
    cropedImgWidth = CFG.CROP_IMG_WIDTH#1536   #512*3
    cropedImgHeight = CFG.CROP_IMG_HEIGHT#768   #256*3

    # return img[int(vanishPtY-cropedImgHeight/2):int(vanishPtY+cropedImgHeight/2), int(vanishPtX-cropedImgWidth/2):int(vanishPtX+cropedImgWidth/2), :].copy()

    return img[CFG.CROP_IMG_Y:CFG.CROP_IMG_Y+CFG.CROP_IMG_HEIGHT, :, :].copy()

    # return img[255:945, :, :].copy()



def prep_image(img):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
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

def main(args):
    ic = Lane_warning()
    rospy.init_node("lanedetnode", anonymous=True)
    rospy.loginfo('model init end')
    rospy.spin()
    # while not rospy.is_shutdown():
    #     print("laneeemain")
    #     try:
    #         rospy.spin()
    #     except KeyboardInterrupt:
    #         print("Shutting down")
    #     rate.sleep()
    print('endddd')






if __name__ == "__main__":
    # imgPath = '/home/iairiv/data/fc2_save_2020-06-03-110140-0000/images/20.jpg'
    # image = cv2.imread(imgPath)
    # cv2.imshow("12313",image)
    # cv2.waitKey(0)

    # main(sys.argv)
    test()
    # img = cv2.imread('/home/iairiv/Desktop/1.png')
    # #cv2.imshow('111',img)

    # ic = Lane_warning()
    # #ic.callback(image)
    # image_path = '/home/iairiv/data/fc2_save_2020-06-03-110140-0000/images'
    # imagedir = os.listdir(image_path)
    # imagedir.sort()
    # for i in imagedir:
    #     print(i)
    #     image = os.path.join(image_path,i)
    #     image = cv2.imread(image)
    #     ic.callback(image)

    # lane = Lane()
    # print(lane.points)








