import os
# import xml.dom.minidom
import cv2 as cv
import json
import numpy as np
import argparse
from config import *
from model import LaneNet
from utils.transforms import *
from utils.postprocess import *
import time
import torch.backends.cudnn as cudnn
import os
from utils.prob2lines import getLane
from utils.lanenet_postprocess import LaneNetPostProcessor
import numpy as np
from Detection import Detection


class Lane_warning:
    def __init__(self):
        self.weights_file = '/home/iairiv/PycharmProjects/lane_waring/experiments/exp1/exp1_best.pth'
        self.band_width = 1.5
        self.image_X = 1920
        self.image_Y = 1200
        self.car_X = self.image_X/2
        self.car_Y = self.image_Y

    def lane_detect(self,imagepath):
        postprocessor = LaneNetPostProcessor()
        weight_path = self.weights_file
        _set = "IMAGENET"
        mean = IMG_MEAN[_set]
        std = IMG_STD[_set]
        transform_img = Resize((512, 256))
        transform_x = Compose(ToTensor(), Normalize(mean=mean, std=std))
        transform = Compose(transform_img, transform_x)
        net = LaneNet(pretrained=False, embed_dim=4, delta_v=.5, delta_d=3.)
        # print(net)
        save_dict = torch.load(weight_path, map_location='cuda:0')
        net.load_state_dict(save_dict['net'])
        net.eval()
        # cudnn.benchmark = True
        # cudnn.fastest = True

        net = net.to('cuda')
        # for img_name in os.listdir(img_root):
        img = cv2.imread(imagepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB for net model input
        # img = img[:600, :, :]
        img_org = img[255:945, :, :]
        img = transform_img({'img': img_org})['img']
        x = transform_x({'img': img})['img']
        x.unsqueeze_(0)
        x = x.to('cuda')
        start = time.time()
        with torch.no_grad():
            output = net(x)
        end = time.time()
        print(end - start)

        embedding = output['embedding']
        embedding = embedding.detach().cpu().numpy()
        embedding = np.transpose(embedding[0], (1, 2, 0))
        binary_seg = output['binary_seg']
        bin_seg_prob = binary_seg.detach().cpu().numpy()
        bin_seg_pred = np.argmax(bin_seg_prob, axis=1)[0]
        seg = bin_seg_pred * 255

        postprocess_result = postprocessor.postprocess(
            binary_seg_result=bin_seg_pred,
            instance_seg_result=embedding,
            source_image=cv2.cvtColor(img_org, cv2.COLOR_RGB2BGR)
        )
        # cv2.imwrite("./output/{}_result.jpg".format(img_name[:-4]), postprocess_result['source_image'])
        # cv2.imwrite("./output/{}_mask_image.jpg".format(img_name[:-4]), postprocess_result['mask_image'])
        points = postprocess_result['points']
        points = np.array(points)
        # print(points)
        return points

    def draw_anchor(self,imagepath,txtpath):
        # 读取图片
        # print()
        img = cv.imread(imagepath)
        # cv.imshow('1.jpg', img)
        # cv.waitKey()
        f = open(txtpath, 'r')
        for line in f.readlines():
            line = line.strip('\n')
            # print(line)
            class_label = line.split()[0]
            score = line.split()[1]
            # print(class_label)
            # print(line.split()[1].isdigit())
            if line.split()[2].isdigit():
                x1 = int(line.split()[2])
                x2 = int(line.split()[4])
                y1 = int(line.split()[3])
                y2 = int(line.split()[5])
                # print((x1,y1),(x2, y2))
                cv.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)
                cv.putText(img, class_label + score, (x1, y1), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), thickness=1)
        # cv.imshow('head', img)
        # cv.waitKey()
        # print(savepath)
        # cv.imwrite(savepath, img) # save picture
        return img

    def draw_line(self,image, arr):
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                arr[i][j][1] = arr[i][j][1] + 255
                # print(arr[i][j])
                # cv.circle(image, (int(arr[i][j][0]),int(arr[i][j][1])), 5, (0, 0, 213), -1) #画成小圆点
            cv.line(image, (int(arr[i][0][0]), int(arr[i][0][1])), (int(arr[i][-1][0]), int(arr[i][-1][1])),
                    (0, 255, 0), 3)
        # cv.imshow('1.jpg', image)
        # cv.waitKey()
        img = image[0:925, :, :]
        # cv.imwrite(savepath, img)
        return img

    def color(self, signal):
        if signal == 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        return color

    def draw_warn(self, image):
        image_w = image.shape[1]
        image_h = image.shape[0]

        image = cv.circle(image, (image_w / 2, image_h - 20), 5, (255, 0, 0), thickness=1)  # 在图片上方中心·偏下的位置画一个红色实心园代表偏离

        return image

    def test(self,num):
        print('数字是：',num)


if __name__ == "__main__":
    raw_image = '/home/iairiv/PycharmProjects/lane_waring/fc2_save_2020-06-03-110140-0000/images'
    yolo_result = '/home/iairiv/PycharmProjects/lane_waring/fc2_save_2020-06-03-110140-0000/predictions'
    save_path = '/home/iairiv/PycharmProjects/PyProjec/data/result/tt'
    # raw_image = './fc2_save_2020-06-03-110140-0000/images'
    # yolo_result = './fc2_save_2020-06-03-110140-0000/predictions'
    # save_path = './data/result/test'
    imagelist = os.listdir(raw_image)
    imagelist.sort()
    imagelist.sort(key=lambda x: int(x[:-4]))
    # print(imagelist)
    l = Lane_warning()
    d = Detection()
    # l.test(10)
    for i in range(len(imagelist)):
        print(imagelist[i])
        txtname = imagelist[i][:-4]+'.txt'
        imagepath = os.path.join(raw_image, imagelist[i])
        savepath = os.path.join(save_path,imagelist[i])
        txtpath = os.path.join(yolo_result,txtname)
        time1 = time.time()
        npy=l.lane_detect(imagepath)
        time2 = time.time()-time1
        print('检测用时：',time2)
        # npy = ''
        # print(npy)

        """
        npy格式：
        [list([[870.9816883091108, 406.6777859391837], [856.7879767234359, 414.69712461274247], [843.9775018549878, 423.9372884816137], [832.054188020063, 434.13195064150057], [818.4189841627133, 445.7902053306843], [802.8820921011121, 459.074436582368], [776.5348755599128, 474.7339424922549], [795.0056206281922, 464.3891149060479], [767.2852197057337, 482.1254030424973], [743.9958779504989, 501.63592213597786], [756.3442301034454, 491.291094549771], [724.2885905300133, 514.1196983600485], [688.2121816351824, 539.9645070043103], [700.8100576435521, 529.6196794181035], [666.0197562859075, 558.1879377693965], [678.6176498354456, 547.8431101831897], [628.6809013117029, 581.275443241514], [646.193110073645, 570.9306156553071], [592.4659756196808, 608.9832490066002], [605.3133064589582, 598.6384214203932], [618.1606372982358, 588.2935938341864], [565.4068643718515, 630.771642224542], [578.2541535003578, 620.4268146383353], [502.9505961243924, 670.5045944740033], [519.9836256963155, 660.1597668877963], [537.0166552682383, 649.8149393015894], [554.0496848401614, 639.4701117153825], [469.15348105048815, 697.1473178205819], [482.25018199737076, 686.802490234375]])
        list([[813.0744603277535, 397.30968291183996], [775.4667812122418, 406.8957940463362], [733.8955953336157, 418.0105306691137], [679.7540086942605, 431.22048213564113], [617.59778406113, 446.3859084556843], [554.9166587148181, 463.23135796908673], [592.0679999588323, 452.88653038287987], [519.0474939649521, 472.1918187634698], [426.70711097829206, 494.8362489897629], [468.891682106551, 484.4914214035561], [363.1794162493194, 510.41499591695856], [405.3640100881104, 500.07016833075164], [284.18026510024373, 529.7878102269666], [326.3647295471256, 519.4429826407599], [187.14605475553583, 553.5833845467403], [229.33043235241638, 543.2385569605334], [68.9131975608615, 582.577445194639], [111.09767385688689, 572.2326176084321], [153.2821501529118, 561.8877900222252]])
        list([[964.9051254490054, 397.54613310715246], [969.8274046147045, 403.11774839203935], [991.4027572470903, 427.5393392628637], [999.0810041175474, 436.2304613836881], [1007.8601263559659, 446.1677077720906], [1023.8303993887234, 456.96265069369616], [1037.309270372258, 469.2372078731142], [1050.436573441919, 483.6673436658136], [1041.025750291446, 473.3225160796066], [1055.8481259167515, 489.6159331223061], [1078.3004054443225, 507.301348456021], [1100.689537997544, 527.2642906452047], [1091.0070443732343, 516.9194630589978], [1119.0097637298336, 539.3281144766972], [1102.4567363113104, 528.9832868904903], [1137.2798986122698, 554.2829905542834], [1165.5718490400534, 583.6854774212015], [1155.6177749090793, 573.3406498349946], [1198.455759904099, 605.5229218581627], [1182.4713671167765, 595.1780942719558], [1225.0859655904476, 631.6782373888739], [1214.860162102279, 621.333409802667], [1204.6343586141104, 610.9885822164601]])
        list([[1053.4753725652836, 406.5843716325431], [1067.272211805337, 410.8195001010237], [1084.3695099458819, 416.22094937028555], [1159.9947919221847, 440.11293819032875], [1186.2982037950192, 448.4228347252155], [1216.2433497661957, 457.8833049905711], [1250.1782904732033, 468.6042427852236], [1288.496640898558, 480.709915687298], [1331.6449592656925, 494.3415895659348], [1298.9005368943078, 483.9967619797279], [1347.3893696832502, 499.3156612001616], [1401.8028574458597, 516.506295039736], [1462.8137198402785, 535.7811573949353], [1430.0691307919747, 525.4363298087285], [1498.4565575600761, 547.0416364998653], [1575.1268492188594, 571.2637644800647], [1542.3823835516382, 560.9189368938579], [1628.3968998269581, 588.0930912412446], [1595.652295643287, 577.7482636550377]])]
        """
        time3 = time.time()
        img = l.draw_anchor(imagepath,txtpath)
        time4 = time.time()-time3
        print('画框用时：', time4)
        time5 = time.time()
        img = l.draw_line(img,npy)
        time6 = time.time()-time5
        print('画线用时',time6)
        # is_bia = d.detect(npy)     # .npy文件的地址        is_bia == 1:偏离   is_bia == 0:未偏离
        #
        # if is_bia == 1:
        #     img = l.draw_warn(img)          # 图片上画偏离标志
        #
        # cv.imwrite(savepath, img)







