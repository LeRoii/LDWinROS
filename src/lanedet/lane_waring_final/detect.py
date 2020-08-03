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



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", '-i', type=str,
                        default="/home/iairiv/yolov4/darknet-master/output/fc2_save_2020-06-03-110140-0000/images",
                        help="Path to demo img")
    parser.add_argument("--weight_path", '-w', type=str,
                        default="./experiments/exp1/exp1_best.pth",
                        help="Path to model weights")
    parser.add_argument("--band_width", '-b', type=float, default=1.5, help="Value of delta_v")
    parser.add_argument("--visualize", '-v', action="store_true", default=False, help="Visualize the result")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    img_root = args.img_path
    weight_path = args.weight_path

    point_size = 1
    point_color = (0, 0, 255)  # BGR
    thickness = 4

    _set = "IMAGENET"
    mean = IMG_MEAN[_set]
    std = IMG_STD[_set]
    # transform_img = Resize((800, 288))
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
    for img_name in os.listdir(img_root):
        img = cv2.imread(os.path.join(img_root, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB for net model input
        # img = img[:600, :, :]
        img = img[255:945, :, :]
        img = transform_img({'img': img})['img']
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

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        seg_img = np.zeros_like(img)
        lane_seg_img = embedding_post_process(embedding, bin_seg_pred, args.band_width, 4)
        lane_coords = getLane.polyfit2coords_tusimple(lane_seg_img, resize_shape=(256, 512), y_px_gap=10, pts=56)
        for i in range(len(lane_coords)):
            lane_coords[i] = sorted(lane_coords[i], key=lambda pair: pair[1])
        color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')
        for i, lane_idx in enumerate(np.unique(lane_seg_img)):
            if lane_idx == 0:
                continue
            seg_img[lane_seg_img == lane_idx] = color[i-1]

        img = cv2.addWeighted(src1=seg_img, alpha=0.8, src2=img, beta=1., gamma=0.)
        for lane in lane_coords:
            for p in lane:
                if p[0] == -1:
                    continue
                cv2.circle(img, (p[0], p[1]), point_size, point_color, thickness)


        cv2.imwrite("./output/{}_result.jpg".format(img_name[:-4]), img)

    if args.visualize:
        cv2.imshow("", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
