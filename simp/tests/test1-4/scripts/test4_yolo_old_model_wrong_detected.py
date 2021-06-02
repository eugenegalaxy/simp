import os
import sys
import time


import numpy as np
import cv2

from simp.submodules.nofever.nofever.utils import DebugPrint
from simp.submodules.nofever.nofever.detection.yolov5.face_detect import YOLO

BASE_PT = os.path.dirname(os.path.abspath(__file__))

DEBUG_MODE = True
DEBUG = DebugPrint(DEBUG_MODE)

TEST_NAME = 'old_incorrect_detections'
REL_PT = 'yolo_dataset/mask_incorrect_detections'
IMG_PT = os.path.join(BASE_PT, REL_PT)


class LogWriter():
    def __init__(self, file_pt):
        self.file_pt = file_pt
        try:
            with open(file_pt, 'a') as file:
                file.close
        except Exception as e:
            print(e)

    def write(self, msg):
        with open(self.file_pt, 'a') as file:
            t = time.localtime()
            current_time = time.strftime("[%H:%M:%S] ", t)
            file.write(current_time + msg + '\n')
            file.close


log_pt = '{}_results.txt'.format(TEST_NAME)
full_log_pt = os.path.join(BASE_PT, log_pt)
LOG = LogWriter(full_log_pt)

# YOLO model cannot be found unless added to system paths...
yolo_rel_path = '/home/eugenegalaxy/Documents/projects/simp/simp/submodules/nofever/nofever/detection/yolov5'
sys.path.insert(0, yolo_rel_path)

# Loading YOLO only once (hehe) instead of inside functions
YOLO_MASK = YOLO(weights='mask')
YOLO_MASK.load_yolov5()


def detect_facemask(src_img):
    # Parsing input
    file_pt = None
    if type(src_img) == str:
        if os.path.isfile(src_img):
            file_pt = os.path.basename(src_img)
            _, ext = os.path.splitext(src_img)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                src_img = cv2.imread(src_img)
        elif os.path.isdir(src_img):
            for one_img in os.listdir(src_img):
                _, ext = os.path.splitext(one_img)
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    file_pt = os.path.basename(one_img)
                    src_img = cv2.imread(os.path.join(src_img, one_img))
    elif type(src_img) == np.ndarray:
        pass
    else:
        raise TypeError("'src_img' must be either '/path/to/img.jpg' or '/path/to/dir' or 'numpy.ndarray' object.")

    # findings, bb_img = YOLO_MASK.predict(img, view_img=True)
    findings, bb_img = YOLO_MASK.predict(src_img)
    strongest_finding = []
    if len(findings) > 0:
        findings.sort(reverse=True, key=lambda index: index[2])
        strongest_finding = findings[0]
        result_dic = {"guess": strongest_finding[3], "conf": strongest_finding[2]}
        if file_pt is not None:
            result_dic["file_name"] = file_pt
        return result_dic, bb_img
    else:
        return None, None


def run_YOLO_through_directory(dir_path, generate_images=True):
    parent_dir = os.path.dirname(dir_path)
    detections_pt = os.path.join(parent_dir, 'test_results_{}'.format(TEST_NAME))
    if not os.path.isdir(detections_pt):
        os.mkdir(detections_pt)

    LOG.write('Images with bounding boxes will be saved to: {}'.format(detections_pt))

    all_findings = []

    LOG.write('')
    LOG.write('Starting mask detection process =======================================================================')
    for idx, img_file in enumerate(os.listdir(dir_path)):
        filename, ext = os.path.splitext(img_file)
        img_full_path = os.path.join(dir_path, img_file)
        if os.path.isfile(img_full_path):
            if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                finding, bb_img = detect_facemask(img_full_path)
                if finding is not None:
                    # DEBUG(finding)
                    LOG.write('Image {0} -> Guess: {1} , Confidence: {2:1.2f} , File Name: {3}'.format(
                        idx, finding["guess"], finding["conf"], finding["file_name"]))
                    all_findings.append(finding)
                    if generate_images is True:
                        save_mask_pt = os.path.join(detections_pt, '{0}.jpg'.format(
                            filename))
                        cv2.imwrite(save_mask_pt, bb_img)
                else:
                    all_findings.append({'guess': 'no_detections', "conf": 0, 'file_name': img_file})
                    LOG.write('Image {0} -> No detections , File Name: {1}'.format(idx, img_file))
    return all_findings


LOG.write('================================================================================================')
LOG.write('Starting Test: Old incorrect detections =============================================')

LOG.write('')
LOG.write('Test: Accuracy of a NEW YOLOv5 RR model comparing to accuracy of the old model.')
LOG.write('Description: Detection results of the old model are saved as a name of a file of that image.')
LOG.write('IMPORTANT: Results will be computed manually by observing the image, old guess, and  the new guess.')
LOG.write('YOLOv5 weights: simp/submodules/nofever/nofever/detection/yolov5/weights/RR_aug_e100_best.pt')

LOG.write('Dataset path: {}'.format(IMG_PT))

num_of_img = len([name for name in os.listdir(IMG_PT)])
LOG.write('Images found in a given Dataset: {}'.format(num_of_img))

t_start = time.time()
results = run_YOLO_through_directory(IMG_PT, generate_images=True)
t_end = time.time()
LOG.write('Ending mask detection process =============================================================================')
LOG.write('Elapsed time: {:1.2f} seconds'.format(t_end - t_start))
LOG.write('{0} images successfuly scanned out of {1} files in the provided directory.'.format(len(results), num_of_img))
LOG.write('')
LOG.write('Results: Counted manually by inspecting image vs old guess vs new guess.')
LOG.write('')

total_cnt_correct = 0

total_cnt_mask_on = 0
total_cnt_mask_off = 0
total_cnt_mask_wrong = 0
total_cnt_no_detections = 0

total_avg_conf = 0
avg_conf_mask_on = 0
avg_conf_mask_off = 0
avg_conf_mask_wrong = 0
for idx, det in enumerate(results):
    DEBUG('{0} -> {1}'.format(idx, det))
    if det['guess'] == TEST_NAME:
        total_cnt_correct += 1

    if det['guess'] == 'mask_on':
        total_cnt_mask_on += 1
        avg_conf_mask_on += det['conf']
    elif det['guess'] == 'mask_off':
        total_cnt_mask_off += 1
        avg_conf_mask_off += det['conf']
    elif det['guess'] == 'mask_wrong':
        total_cnt_mask_wrong += 1
        avg_conf_mask_wrong += det['conf']
    elif det['guess'] == 'no_detections':
        total_cnt_no_detections += 1
    else:
        DEBUG('What happened? det["guess"] is {}'.format(det['guess']))

    total_avg_conf += det['conf']

num_of_scans = len(results)
total_avg_conf /= num_of_scans

LOG.write('Average total confidence: {:1.2f}'.format(total_avg_conf))

if total_cnt_mask_on != 0:
    avg_conf_mask_on /= total_cnt_mask_on
    LOG.write('Label "mask_on" was found {0} times with average confidence of {1:1.2f}'.format(
        total_cnt_mask_on, avg_conf_mask_on))
else:
    LOG.write('Label "mask_on" was found 0 times.')

if total_cnt_mask_off != 0:
    avg_conf_mask_off /= total_cnt_mask_off
    LOG.write('Label "mask_off" was found {0} times with average confidence of {1:1.2f}'.format(
        total_cnt_mask_off, avg_conf_mask_off))
else:
    LOG.write('Label "mask_off" was found 0 times.')

if total_cnt_mask_wrong != 0:
    avg_conf_mask_wrong /= total_cnt_mask_wrong
    LOG.write('Label "mask_wrong" was found {0} times with average confidence of {1:1.2f}'.format(
        total_cnt_mask_wrong, avg_conf_mask_wrong))
else:
    LOG.write('Label "mask_wrong" was found 0 times.')

if total_cnt_no_detections != 0:
    LOG.write('Label "no_detections" was found {} times.'.format(total_cnt_no_detections))
else:
    LOG.write('Label "no_detections" was found 0 times.')

LOG.write('================================================================================================')
LOG.write('End of a test.')
