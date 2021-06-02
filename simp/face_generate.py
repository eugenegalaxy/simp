import os
import sys
import numpy as np
import cv2
import traceback
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from simp.submodules.nofever.nofever.mask_scanner import MaskFinder
from simp.submodules.nofever.nofever.detection.yolov5.face_detect import YOLO
from simp.submodules.face_generator.face_generator.GAN_infer import Predictor


# YOLO model cannot be found unless added to system paths...
base_path = os.path.dirname(os.path.abspath(__file__))
yolo_rel_path = 'submodules/nofever/nofever/detection/yolov5'
full_yolo_path = os.path.join(base_path, yolo_rel_path)
sys.path.insert(0, full_yolo_path)

# Loading YOLO only once (hehe) instead of inside functions
YOLO_MASK = YOLO(weights='mask')
YOLO_MASK.load_yolov5()


def get_xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = [0, 0, 0, 0]
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    y = [int(x) for x in y]
    return y


def detect_facemask_and_create_binary_tensor(src_img):
    img_height, img_width, _ = src_img.shape
    # findings = YOLO_MASK.predict(img, view_img=True)
    findings = YOLO_MASK.predict(src_img)
    strongest_finding = []
    tensor = []
    if len(findings) > 0:
        findings.sort(reverse=True, key=lambda index: index[2])
        strongest_finding = findings[0]
        BB_xywh = strongest_finding[1]
        x1, y1, x2, y2 = get_xywh2xyxy(BB_xywh)
        binary_img = np.zeros((img_height, img_width), dtype="uint8")
        for x in range(x1, x2):
            for y in range(y1, y2):
                binary_img[y][x] = 1
        tensor = torch.Tensor(binary_img).unsqueeze(0).unsqueeze(0).to('cuda:0')
    else:
        print('{}  -> Image has no detections'.format(src_img))

    if len(tensor) == 0:
        return None
    else:
        return tensor


def detect_and_generate(img_path, save_gan_img_pt=None, save_mask_img_pt=None):
    # Parsing input
    if type(img_path) == str:
        if os.path.isfile(img_path):
            _, ext = os.path.splitext(img_path)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                img = cv2.imread(img_path)
        elif os.path.isdir(img_path):
            for one_img in os.listdir(img_path):
                _, ext = os.path.splitext(one_img)
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    img = cv2.imread(os.path.join(img_path, one_img))
    elif type(img_path) == np.ndarray:
        img = img_path
    else:
        raise TypeError("'src_img' must be either '/path/to/img.jpg' or '/path/to/dir' or 'numpy.ndarray' object (e.g. from 'cv2.imread'). ")

    # Open image, run YOLO MASK FINDER from NoFever, find mask boundary box, create binary map for its area, convert that to torch.Tensor
    binary_tensor = detect_facemask_and_create_binary_tensor(img)
    # Assert that at least 1 mask was found -> Otherwise nothing to remove/generate
    assert(binary_tensor is not None), 'No mask detections. Binary Tensor is None'
    # Initialize Image Inpaintor
    PD = Predictor()
    # Run GAN network to remove facemask area from src image, and generate missing parts inside.
    img_gan, img_orig, img_masks = PD.generate_face(img, binary_tensor)

    if save_gan_img_pt is not None:
        save_image(img_gan.clone().cpu(), save_gan_img_pt)

    if save_mask_img_pt is not None:
        save_image(img_masks.clone().cpu(), save_mask_img_pt)

    return img_gan, img_orig, img_masks


def run_GAN_through_directory(dir_path):
    binary_pt = os.path.join(dir_path, 'binary')
    gan_pt = os.path.join(dir_path, 'gan')

    if not os.path.isdir(binary_pt):
        os.mkdir(binary_pt)
    if not os.path.isdir(gan_pt):
        os.mkdir(gan_pt)

    for img_file in tqdm(os.listdir(dir_path)):
        print(img_file)
        filename, ext = os.path.splitext(img_file)
        img_full_path = os.path.join(dir_path, img_file)
        if os.path.isfile(img_full_path):
            save_mask_pt = os.path.join(binary_pt, filename + '_binary.jpg')
            save_gan_pt = os.path.join(gan_pt, filename + '_gan.jpg')
            gan, orig, mask = detect_and_generate(img_full_path, save_mask_img_pt=save_mask_pt, save_gan_img_pt=save_gan_pt)


if __name__ == '__main__':
    try:
        # DOESNT WORK PROPERLY. SAVES THE FIRST IMAGE ALL THE TIME
        # # Provide path to image and open it (will be replaced by dynamic code)
        path = '/home/eugenegalaxy/Desktop/master_thesis_tests/yolo_dataset/for_report'
        run_GAN_through_directory(path)
    except Exception:
        problem = traceback.format_exc()
        print(problem)
