import os
import cv2
import torch
import numpy as np

from simp.submodules.nofever.nofever.detection.yolov5.models.experimental import attempt_load
from simp.submodules.nofever.nofever.detection.yolov5.utils.datasets import letterbox
from simp.submodules.nofever.nofever.detection.yolov5.utils.general import (
    check_img_size, non_max_suppression, scale_coords, xyxy2xywh, plot_one_box)
from simp.submodules.nofever.nofever.detection.yolov5.utils.torch_utils import select_device, time_synchronized


class YOLO(object):
    # Path to YOLOv5 weighs for facial features detection.
    WEIGHTS_FACE = 'weights/face_w_glasses_yolov5s_e300_default_hyp.pt'
    # Path to YOLOv5 weighs for mask status detection.
    # WEIGHTS_MASK = 'weights/mask_yolov5m_e300_evolved_hyp.pt'
    WEIGHTS_MASK = 'weights/RR_aug_e100_best.pt'
    # ratio from 0 to 1. Discards detection with confidence less than this value.
    DETECTION_CONFIDENCE_THRESHOLD = 0.20
    # ratio from 0 to 1. Discards detections with Intersection-over-Union less than this value.
    IOU_THRESHOLD = 0.6
    # Width/heiht of images. Predictor converts source images to this resolution (smaller -> faster, but less accurate)
    IMG_SIZE = 448

    def __init__(self, weights='far'):
        self.g_file_pt = os.path.dirname(__file__)

        if weights == 'far':
            self.weights = os.path.join(self.g_file_pt, self.WEIGHTS_FACE)
        elif weights == 'mask':
            self.weights = os.path.join(self.g_file_pt, self.WEIGHTS_MASK)
        else:
            err_msg1 = "Requested keyword for parameter 'weights' is not identified. \n"
            err_msg2 = "Requested keyword: '{}'. Available keywords: 'far', 'close', 'mask'.".format(weights)
            raise ValueError(err_msg1 + err_msg2)
        # self.imgsz = 640
        self.model = None
        self.colors = None
        self.names = None
        self.device = ''
        self.half = None

    def load_yolov5(self):
        # device = 'cpu' or '0' or '0,1,2,3'
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def predict(self, src_img, view_img=False, print_enabled=False):

        if src_img is None:
            return None

        imgsz = check_img_size(self.IMG_SIZE, s=self.model.stride.max())

        # Don't know how but these two lines make code faster-----
        img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None
        # ----------------------------------------------------------------------

        img = letterbox(src_img, new_shape=imgsz)[0]  # Padded resize

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=self.DETECTION_CONFIDENCE_THRESHOLD, iou_thres=self.IOU_THRESHOLD)
        t2 = time_synchronized()

        all_detections = []
        for i, det in enumerate(pred):
            p = self.g_file_pt
            print_str = ''

            print_str += '%gx%g ' % img.shape[2:]  # print string

            if det is not None and len(det):

                if print_enabled:
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        print_str += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], src_img.shape).round()

                for *xyxy, conf, cls in det:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh

                    if print_enabled:
                        print(('%g ' * 5) % (cls, *xywh))

                    cls_cpu = int(cls.cpu().numpy().item())  # To get value from Tensor object located on CUDA
                    conf_cpu = conf.cpu().numpy().item()  # same as comment above
                    conf_cpu = round(conf_cpu, 4)
                    cls_name = self.names[int(cls)]
                    one_detection = []
                    one_detection.append(cls_cpu)
                    one_detection.append(xywh)
                    one_detection.append(conf_cpu)
                    one_detection.append(cls_name)
                    all_detections.append(one_detection)

                    label = '%s %.2f' % (cls_name, conf)
                    plot_one_box(xyxy, src_img, label=label, color=[255, 0, 0], line_thickness=2)
                if view_img:
                    cv2.imshow(p, src_img)
                    cv2.waitKey()
                    cv2.destroyAllWindows()

                if print_enabled:
                    # Print time (inference + NMS)
                    print('%sDone. (%.3fs)' % (print_str, t2 - t1))

        # Returns list of sublists. Each sublist=[class_index, [bounding box data], confidence, label]
        return all_detections, src_img


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
