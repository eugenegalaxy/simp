import os, sys
import numpy as np
import cv2
from nofever.nofever.mask_scanner import MaskFinder
from nofever.nofever.detection.yolov5.face_detect import YOLO

base_path = os.path.dirname(os.path.abspath(__file__))
yolo_rel_path = 'nofever/nofever/detection/yolov5'
full_yolo_path = os.path.join(base_path, yolo_rel_path)
sys.path.insert(0, full_yolo_path)


def get_xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = [0, 0, 0, 0]
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    y = [int(x) for x in y]
    return y


def mask_finder_and_binary_img_creator(dir_path):

    img_list = []
    if os.path.isdir(dir_path):
        img_list = os.listdir(dir_path)

    parent_dir = os.path.dirname(dir_path)
    binary_ful_dir = os.path.join(parent_dir, 'binary')
    print(binary_ful_dir)

    if not os.path.isdir(os.path.join(binary_ful_dir)):
        os.mkdir(os.path.join(binary_ful_dir))

    YOLO_MASK = YOLO(weights='mask')
    YOLO_MASK.load_yolov5()

    for im in img_list:
        img = cv2.imread(os.path.join(dir_path, im))
        img_height, img_width, channels = img.shape
        findings = YOLO_MASK.predict(img)

        strongest_finding = []
        if len(findings) > 0:
            findings.sort(reverse=True, key=lambda index: index[2])
            strongest_finding = findings[0]
            BB_xywh = strongest_finding[1]
            x1, y1, x2, y2 = get_xywh2xyxy(BB_xywh)
            binary_img = np.zeros((img_height, img_width), dtype="uint8")
            for x in range(x1, x2):
                for y in range(y1, y2):
                    binary_img[y][x] = 1

            full_binary_img_path = os.path.join(binary_ful_dir, im)
            # full_binary_img_path = os.path.join(binary_ful_dir, 'binary.jpg')
            # print('Saving {}'.format(binary_img))
            cv2.imwrite(full_binary_img_path, binary_img)
        else:
            print('{}  -> Image has no detections'.format(im))


dir_path = "/home/eugenegalaxy/Desktop/image_inpaint-RR_all_670img_480x480/images/orig"
mask_finder_and_binary_img_creator(dir_path)