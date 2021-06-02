import os
import time
import pathlib
import traceback
import copy
import random
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

from simp.submodules.nofever.nofever.utils import DebugPrint
from simp.submodules.deepface.deepface import DeepFace


BASE_PT = os.path.dirname(os.path.abspath(__file__))

DEBUG_MODE = True
DEBUG = DebugPrint(DEBUG_MODE)

TEST_NAME = 'identity_verification'
REL_PT = 'deepface_dataset/identities'
# REL_PT = 'deepface_dataset/test'
IMG_PT = os.path.join(BASE_PT, REL_PT)
NUM_OF_IDENTITIES = len([name for name in os.listdir(IMG_PT)])

# Only "VGG-Face" and "Facenet" are available atm (have weights)
MODEL_NAME = 'VGG-Face'
METRICS = 'cosine'


if MODEL_NAME == 'VGG-Face':
    if METRICS == 'euclidean_l2':
        THRESHOLD = 0.80
    elif METRICS == 'cosine':
        THRESHOLD = 0.43
    else:
        raise ValueError('Wrong Metrics selected. Choose "euclidean_l2" or "cosine".')
elif MODEL_NAME == 'Facenet':
    if METRICS == 'euclidean_l2':
        THRESHOLD = 0.80
    elif METRICS == 'cosine':
        THRESHOLD = 0.40
    else:
        raise ValueError('Wrong Metrics selected. Choose "euclidean_l2" or "cosine".')
else:
    THRESHOLD = 0.40  # I hope you know what you are doing... Can be anything?

MODEL = DeepFace.build_model(MODEL_NAME)

# detector_backend (string): set face detector backend as mtcnn, opencv, ssd or dlib
DETECTOR_BACKEND = 'mtcnn'

# NOTE! Will show both scatter plot and MANY (!!) identity verif images.
DEBUG_SHOW_IMAGES = None  # Has to be 'None' to disable and anything else to enable


class LogWriter():
    def __init__(self, file_pt):
        self.file_pt = file_pt
        try:
            with open(file_pt, 'a') as file:
                file.close
        except Exception as e:THRE
            LOG.write(e)

    def write(self, msg):
        with open(self.file_pt, 'a') as file:
            t = time.localtime()
            current_time = time.strftime("[%H:%M:%S] ", t)
            file.write(current_time + msg + '\n')
            file.close


log_pt = 'test5-6-7_{}_results.txt'.format(TEST_NAME)
results_dir = os.path.join(BASE_PT, 'test_results_{}'.format(TEST_NAME))
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)
full_log_pt = os.path.join(results_dir, log_pt)
LOG = LogWriter(full_log_pt)


def proper_round(num, dec=0):
    num = str(num)[:str(num).index('.') + dec + 2]
    if num[-1] >= '5':
        return float(num[:-2 - (not dec)] + str(int(num[-2 - (not dec)]) + 1))
    return float(num[:-1])


def parse_image(src_img):
    # Parsing input
    if type(src_img) == str:
        if os.path.isfile(src_img):
            _, ext = os.path.splitext(src_img)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                src_img = cv2.imread(src_img)
            return src_img
        elif os.path.isdir(src_img):
            all_src_imgs = []
            for one_img in os.listdir(src_img):
                _, ext = os.path.splitext(one_img)
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    pt = os.path.join(src_img, one_img)
                    img = cv2.imread(pt)
                    all_src_imgs.append(img)
            return all_src_imgs
        else:
            return None
    elif type(src_img) == np.ndarray:
        return src_img
    else:
        raise TypeError("'src_img' must be either '/path/to/img.jpg' or '/path/to/dir' or 'numpy.ndarray' object.")


def compare_two_identities(img1_pt, img2_pt, save_img=None, show_img=None):

    if os.path.splitext(os.path.basename(img1_pt))[0] == 'coronapas' and\
       os.path.splitext(os.path.basename(img2_pt))[0] == 'coronapas':
        return None

    result = DeepFace.verify(img1_pt, img2_pt, model=MODEL, model_name=MODEL_NAME,
                             distance_metric=METRICS,
                             dist_threshold=THRESHOLD,
                             detector_backend=DETECTOR_BACKEND)

    # Shorten & Filter results. Nobody's got time to read through all of this...
    result['thr'] = result.pop('max_threshold_to_verify')
    result.pop('model')
    result.pop('similarity_metric')
    result['distance'] = proper_round(result['distance'], 2)
    path1 = pathlib.PurePath(img1_pt)
    path2 = pathlib.PurePath(img2_pt)
    img1_filename = os.path.splitext(os.path.basename(img1_pt))[0]  # Get a file name without extension from a path.
    img2_filename = os.path.splitext(os.path.basename(img2_pt))[0]
    result['img1'] = os.path.join(path1.parent.name, img1_filename)
    result['img2'] = os.path.join(path2.parent.name, img2_filename)

    if save_img is not None or show_img is not None:
        img1 = parse_image(img1_pt)  # Convert string path to image into numpy.ndarray object.
        img2 = parse_image(img2_pt)
        img1 = img1[..., ::-1].copy()  # Converting openCV image (BGR) to PIL image
        img2 = img2[..., ::-1].copy()
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.imshow(img1)
        ax1.title.set_text(result['img1'])
        ax1.axis('off')
        ax2 = fig.add_subplot(122)
        ax2.imshow(img2)
        ax2.title.set_text(result['img2'])
        ax2.axis('off')
        # fig.tight_layout()
        fig.suptitle('Same identity: {0}\nDistance: {1:1.2f}\nThreshold: {2}'.format(
            result['verified'], result['distance'], result['thr']))
        if save_img is not None:
            plt.savefig('saved_figure.png')  # Saves to root folder of simp. Readress to what's needed.
        if show_img is not None:
            plt.show()
    log_str = '{0} vs {1}\t -> Verified:{2} | Distance:{3:1.2f}'.format(
        result['img1'], result['img2'], result['verified'], result['distance']).expandtabs(10)
    LOG.write(log_str)
    return result


def compare_coronapas_to_identity(coronapas_pt, identity_dir_pt, save_img=None, show_img=None):
    all_id_verif_results = []
    coronapas_img = parse_image(coronapas_pt)
    id_images = []
    id_images_pt = []
    for im in os.listdir(identity_dir_pt):
        _, ext = os.path.splitext(os.path.join(identity_dir_pt, im))
        if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
            abs_img_pt = os.path.join(identity_dir_pt, im)
            result = compare_two_identities(coronapas_pt, abs_img_pt)
            if result is not None:
                identity_img = parse_image(abs_img_pt)
                id_images.append(identity_img)
                id_images_pt.append(abs_img_pt)
                all_id_verif_results.append(result)

    if save_img is not None or show_img is not None:
        coronapas_img = coronapas_img[..., ::-1].copy()  # Converting openCV image (BGR) to PIL image
        id_images = [x[..., ::-1].copy() for x in id_images]
        rows = 1
        cols = len(id_images) + 1  # coronapas_img + 'n' identity images
        fig = plt.figure()
        plt.rc('font', size=8)  # SIZE OF A SUBPLOT TITLE
        title_line_1 = 'Identity Verification: Coronapas vs Scan Triplet\n'
        title_line_2 = "Model: '{0}', Metrics: '{1}', Distance Threshold: {2:1.2f}".format(
            MODEL_NAME, METRICS, THRESHOLD)
        fig.suptitle(title_line_1 + title_line_2, fontsize=12)
        # Plot Coronapas img first
        ax = fig.add_subplot(rows, cols, 1)
        filename_no_ext = os.path.splitext(os.path.basename(coronapas_pt))[0]
        corona_pt = pathlib.PurePath(coronapas_pt)
        corona_short_pt = os.path.join(corona_pt.parent.name, filename_no_ext)
        ax.title.set_text('Comparable:\n{}'.format(corona_short_pt))
        ax.imshow(coronapas_img)
        ax.axis('off')
        # Then plot all other identity images to the right from the Coronapas image.
        for idx, im in enumerate(id_images):
            ax = fig.add_subplot(rows, cols, idx + 2)
            filename_no_ext = os.path.splitext(os.path.basename(id_images_pt[idx]))[0]
            id_pt = pathlib.PurePath(id_images_pt[idx])
            identity_short_pt = os.path.join(id_pt.parent.name, filename_no_ext)
            title = 'Target:\n{0}\nVerified: {1}\nDistance: {2}'.format(
                identity_short_pt, all_id_verif_results[idx]['verified'], all_id_verif_results[idx]['distance'])
            ax.title.set_text(title)
            ax.imshow(im)
            ax.axis('off')
        plt.tight_layout()
        if save_img is not None:
            # 'save_img' variable is a path to a directory. Not best practice but convenient here
            save_dir_pt = save_img  # Example: .../deepface_dataset/test_results_identify_verification
            identity_corona_dir = '{}_coronapas'.format(corona_pt.parent.name)  # Example : 'eugene'
            save_pt = os.path.join(save_dir_pt, identity_corona_dir)
            if not os.path.isdir(save_pt):
                os.mkdir(save_pt)
            img_filename = '{0}_vs_{1}.jpg'.format(
                corona_short_pt.replace('/', '.'), id_pt.parent.name)
            img_full_pt = os.path.join(save_pt, img_filename)

            plt.savefig(img_full_pt, dpi=200)
        if show_img is not None:
            plt.show()

    return all_id_verif_results


def run_DEEPFACE_through_directory(dir_path):
    parent_dir = os.path.dirname(dir_path)
    detections_pt = os.path.join(parent_dir, 'test_results_{}'.format(TEST_NAME))
    if not os.path.isdir(detections_pt):
        os.mkdir(detections_pt)

    LOG.write('Images will be saved to: {}'.format(detections_pt))

    all_results = {}

    LOG.write('')
    LOG.write('Starting identity verification process ================================================================')
    for idx, dir_name in enumerate(os.listdir(dir_path)):
        corona_img_jpg = os.path.join(dir_path, dir_name, 'coronapas.jpg')
        corona_img_png = os.path.join(dir_path, dir_name, 'coronapas.png')
        corona_img_jpeg = os.path.join(dir_path, dir_name, 'coronapas.jpeg')
        if os.path.isfile(corona_img_jpg) or os.path.isfile(corona_img_png) or os.path.isfile(corona_img_jpeg):
            if os.path.isfile(corona_img_jpg):
                corona_img = corona_img_jpg
            elif os.path.isfile(corona_img_png):
                corona_img = corona_img_png
            else:
                corona_img = corona_img_jpeg
            one_coronapas_results = []
            LOG.write('Comparing {} coronapas with all identities.'.format(dir_name))
            for identity in tqdm(os.listdir(dir_path)):
                identity_pt = os.path.join(dir_path, identity)
                results = compare_coronapas_to_identity(corona_img, identity_pt,
                                                        show_img=DEBUG_SHOW_IMAGES, save_img=detections_pt)
                one_coronapas_results.append(results)
            all_results[str(idx)] = one_coronapas_results  # {'1': {data}, '2':{data} ...}
    return all_results


def split_results(results_dict):
    # Renaming 'mask_on_gan' to 'mask_gan' through out all entries
    for key, value in results_dict.items():
        for idx1, identity in enumerate(value):
            for idx2, verification in enumerate(identity):
                if 'mask_on_gan' in verification['img2']:
                    name = results_dict[str(key)][idx1][idx2]['img2']
                    id, _ = name.split('/')
                    new_mask_type_name = 'mask_gan'
                    new_name = '{0}/{1}'.format(id, new_mask_type_name)
                    results_dict[str(key)][idx1][idx2]['img2'] = new_name

    # Making full independant copies of the result dictionary.
    mask_off = copy.deepcopy(results_dict)
    mask_on = copy.deepcopy(results_dict)
    mask_gan = copy.deepcopy(results_dict)

    # Manually splitting 1 result dict with all classes into 3 separate dicts: mask_on, mask_off, mask_gan
    # ---------------------------------------------------------------------------------------------------------
    for key, value in mask_off.items():
        for idx1, identity in enumerate(value):
            for idx2, verification in enumerate(identity):
                if 'mask_gan' in verification['img2']:
                    mask_off[str(key)][idx1].pop(idx2)
    for key, value in mask_off.items():
        for idx1, identity in enumerate(value):
            for idx2, verification in enumerate(identity):
                if 'mask_on' in verification['img2']:
                    mask_off[str(key)][idx1].pop(idx2)

    for key, value in mask_gan.items():
        for idx1, identity in enumerate(value):
            for idx2, verification in enumerate(identity):
                if 'mask_on' in verification['img2']:
                    mask_gan[str(key)][idx1].pop(idx2)
    for key, value in mask_gan.items():
        for idx1, identity in enumerate(value):
            for idx2, verification in enumerate(identity):
                if 'mask_off' in verification['img2']:
                    mask_gan[str(key)][idx1].pop(idx2)

    for key, value in mask_on.items():
        for idx1, identity in enumerate(value):
            for idx2, verification in enumerate(identity):
                if 'mask_gan' in verification['img2']:
                    mask_on[str(key)][idx1].pop(idx2)
    for key, value in mask_on.items():
        for idx1, identity in enumerate(value):
            for idx2, verification in enumerate(identity):
                if 'mask_off' in verification['img2']:
                    mask_on[str(key)][idx1].pop(idx2)

    return mask_on, mask_off, mask_gan


def avg_list_value(lst):
    if len(lst) == 0:
        return 0
    else:
        return sum(lst) / len(lst)


def compute_metrics(mask_dict):
    ''' Take a dictionary object produced by run_DEEPFACE_through_directory() and split by split_results(),
        parse data and compute results.
        Produced results for 3 different tests: Coronapas vs Mask-Off, Coronapas vs Mask-On, Coronapas vs Mask-Off-GAN
        ID - "Identity"

        Result metrics:
        True Positive (TP): ID "A" Coronapas and ID "A" Scan Image -> Verified as same identity
        True Negative (TN): ID "A" Coronapas and ID "B" Scan Image -> Not verified
        False Positive (FP): ID "A" Coronapas and ID "B" Scan Image -> Verified as same identity
        False Negative (FN): ID "A" Coronapas and ID "A" Scan Image -> Not verified
        Accuracy: (TP+TN)/(TP+TN+FP+FN) -> total correct / total
        Precision: (TP)/(TP+FP) -> How precise is the system?
        Recall: (TP)/(TP+FN) -> How many correct positive verifications it made? (e.g. out of 26 identites)
        F1 Score: 2 * (Precision * Recall) / (Precision + Recall) -> Better than Accuracy if TP/TN are not balanced
    '''
    TP_total = 0  # in count. Min 0, max NUM_OF_IDENTITIES
    FN_total = 0
    FP_total = 0
    TN_total = 0

    TP_avg_dist_when_verified = []  # same identity
    FN_avg_dist_when_not_verified = []  # same identity
    FP_avg_dist_when_verified = []  # different identity
    TN_avg_dist_when_not_verified = []  # different identity

    FP_and_TN = 0
    TN_debug = 0
    FP_debug = 0

    for key, value in mask_dict.items():
        for idx1, identity in enumerate(value):
            for idx2, verification in enumerate(identity):
                img1_id, _ = mask_dict[str(key)][idx1][idx2]['img1'].split('/')
                img2_id, _ = mask_dict[str(key)][idx1][idx2]['img2'].split('/')
                if img1_id == img2_id:  # TP and FN
                    if verification['verified'] is True:
                        TP_total += 1
                        TP_avg_dist_when_verified.append(verification['distance'])
                    else:
                        FN_total += 1
                        FN_avg_dist_when_not_verified.append(verification['distance'])
                else:  # FP and TN
                    FP_and_TN += 1
                    if verification['verified'] is False:
                        TN_debug += 1
                        TN_total += 1
                        TN_avg_dist_when_not_verified.append(verification['distance'])
                    else:
                        FP_debug += 1
                        FP_total += 1
                        FP_avg_dist_when_verified.append(verification['distance'])

    TP_avg_dist_when_verified = avg_list_value(TP_avg_dist_when_verified)
    FN_avg_dist_when_not_verified = avg_list_value(FN_avg_dist_when_not_verified)
    FP_avg_dist_when_verified = avg_list_value(FP_avg_dist_when_verified)
    TN_avg_dist_when_not_verified = avg_list_value(TN_avg_dist_when_not_verified)

    TP = (TP_total / NUM_OF_IDENTITIES)
    FN = (FN_total / NUM_OF_IDENTITIES)

    FP = (FP_total / FP_and_TN)
    TN = (TN_total / FP_and_TN)

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = (TP) / (TP + FP)
    Recall = (TP) / (TP + FN)
    F1_score = 2 * (Precision * Recall) / (Precision + Recall)

    LOG.write('True positive (TP): {:1.2f}%'.format(TP * 100))
    LOG.write('False negative (FN): {:1.2f}%'.format(FN * 100))
    LOG.write('False positive (FP): {:1.2f}%'.format(FP * 100))
    LOG.write('True negative (TN): {:1.2f}%'.format(TN * 100))

    LOG.write('Avg. dist. when verified same identity (TP): {:1.2f}'.format(TP_avg_dist_when_verified))
    LOG.write('Avg. dist. when not verified same identity (FN): {:1.2f}'.format(FN_avg_dist_when_not_verified))
    LOG.write('Avg. dist. when verified stranger (FP): {:1.2f}'.format(FP_avg_dist_when_verified))
    LOG.write('Avg. dist. when not verified stranger (TN): {:1.2f}'.format(TN_avg_dist_when_not_verified))

    LOG.write('Accuracy: {:1.2f}%'.format(Accuracy * 100))
    LOG.write('Precision: {:1.2f}%'.format(Precision * 100))
    LOG.write('Recall: {:1.2f}%'.format(Recall * 100))
    LOG.write('F1_score: {:1.2f}'.format(F1_score))


def plot_results(results, save_img=None, show_img=None):
    # Make a plot that:
    # Plot 26 distances of ID "A-Z" coronapas vs ID "A-Z" mask-on
    # Plot 26 distances of ID "A-Z" coronapas vs ID "A-Z" mask-off
    # Plot 26 distances of ID "A-Z" coronapas vs ID "A-Z" mask-gan
    # Plot 26 distances of ID "A-Z" coronapas vs ID "random but not the same" mask-gan
    # plot a line across the plot with the THRESHOLD

    mask_off = []
    mask_on = []
    mask_gan = []
    mask_stranger_gan = []

    for key, value in results.items():
        for idx1, identity in enumerate(value):
            for idx2, verification in enumerate(identity):
                img1_id, _ = results[str(key)][idx1][idx2]['img1'].split('/')
                img2_id, mask_type = results[str(key)][idx1][idx2]['img2'].split('/')
                if img1_id == img2_id:  # if 'hugo' == 'hugo'
                    if mask_type == 'mask_off':  # if 'hugo/coronapas' vs 'hugo/mask_off'
                        mask_off.append(verification['distance'])
                    elif mask_type == 'mask_on':  # if 'hugo/coronapas' vs 'hugo/mask_on'
                        mask_on.append(verification['distance'])
                    elif mask_type == 'mask_on_gan':  # if 'hugo/coronapas' vs 'hugo/mask_on_gan'
                        mask_gan.append(verification['distance'])
                if 'mask_on_gan' in verification['img2']:
                    if img1_id != img2_id:
                        mask_stranger_gan.append(verification['distance'])  # This takes 650 entries instead of 26. Fixed below

    # Reducing 'mask_stranger_gan' list from NUM_OF_IDENTITIES * (NUM_OF_IDENTITIES - 1)
    # to NUM_OF_IDENTITIES entries (Randomized entry per each identity)
    length_to_split = [NUM_OF_IDENTITIES - 1] * NUM_OF_IDENTITIES
    mask_stranger_gan = iter(mask_stranger_gan)
    splitted_list = [list(islice(mask_stranger_gan, elem)) for elem in length_to_split]
    new_list = []
    for sublist in splitted_list:
        one_number = random.choice(sublist)
        new_list.append(one_number)
    mask_stranger_gan = new_list

    cnt_mask_on_good = 0
    cnt_mask_on_bad = 0
    cnt_mask_off_good = 0
    cnt_mask_off_bad = 0
    cnt_mask_on_gan_good = 0
    cnt_mask_on_gan_bad = 0
    cnt_mask_on_stranger_gan_good = 0  # NOTE! This is opposite of others. good = above threshold (not verified)
    cnt_mask_on_stranger_gan_bad = 0  # NOTE! This is opposite of others. bad = below threshold (verified)

    for x in mask_on:
        if x < THRESHOLD:
            cnt_mask_on_good += 1
        else:
            cnt_mask_on_bad += 1
    for x in mask_off:
        if x < THRESHOLD:
            cnt_mask_off_good += 1
        else:
            cnt_mask_off_bad += 1
    for x in mask_gan:
        if x < THRESHOLD:
            cnt_mask_on_gan_good += 1
        else:
            cnt_mask_on_gan_bad += 1
    for x in mask_stranger_gan:
        if x < THRESHOLD:
            cnt_mask_on_stranger_gan_bad += 1
        else:
            cnt_mask_on_stranger_gan_good += 1

    fig = plt.figure()

    yaxis_mask_off = [0] * len(mask_off)
    yaxis_mask_off = [random.uniform(2, 3) for x in yaxis_mask_off]
    yaxis_mask_on = [0] * len(mask_on)
    yaxis_mask_on = [random.uniform(5, 6) for x in yaxis_mask_on]
    yaxis_mask_gan = [0] * len(mask_gan)
    yaxis_mask_gan = [random.uniform(8, 9) for x in yaxis_mask_gan]
    yaxis_mask_strangeg_gan = [0] * len(mask_stranger_gan)
    yaxis_mask_strangeg_gan = [random.uniform(11, 12) for x in yaxis_mask_strangeg_gan]

    point_size = 10
    plt.scatter(mask_off, yaxis_mask_off, c='green', label='mask_off', s=point_size)
    plt.scatter(mask_on, yaxis_mask_on, c='red', label='mask_on', s=point_size)
    plt.scatter(mask_gan, yaxis_mask_gan, c='blue', label='mask_same_gan', s=point_size)
    plt.scatter(mask_stranger_gan, yaxis_mask_strangeg_gan, c='magenta', label='mask_stranger_gan', s=point_size)
    plt.plot([THRESHOLD, THRESHOLD], [0.0, 14], color='k', linestyle='-', linewidth=2, label='verification threshold')

    fontsize = 8
    plt.text(THRESHOLD - 0.1, 3.5, cnt_mask_off_good, fontsize=fontsize, color='green')
    plt.text(THRESHOLD - 0.1, 6.5, cnt_mask_on_good, fontsize=fontsize, color='red')
    plt.text(THRESHOLD - 0.1, 9.5, cnt_mask_on_gan_good, fontsize=fontsize, color='blue')
    plt.text(THRESHOLD - 0.1, 12.2, cnt_mask_on_stranger_gan_bad, fontsize=fontsize, color='magenta')

    plt.text(THRESHOLD + 0.1, 3.5, cnt_mask_off_bad, fontsize=fontsize, color='green')
    plt.text(THRESHOLD + 0.1, 6.5, cnt_mask_on_bad, fontsize=fontsize, color='red')
    plt.text(THRESHOLD + 0.1, 9.5, cnt_mask_on_gan_bad, fontsize=fontsize, color='blue')
    plt.text(THRESHOLD + 0.1, 12.2, cnt_mask_on_stranger_gan_good, fontsize=fontsize, color='magenta')

    all_goods = cnt_mask_off_good + cnt_mask_on_good + cnt_mask_on_gan_good + cnt_mask_on_stranger_gan_good
    total_number = len(mask_on) + len(mask_off) + len(mask_gan) + len(mask_stranger_gan)
    accuracy = round(all_goods / total_number * 100, 2)

    all_goods_without_mask_on = cnt_mask_off_good + cnt_mask_on_gan_good + cnt_mask_on_stranger_gan_good
    total_number_without_mask_on = len(mask_off) + len(mask_gan) + len(mask_stranger_gan)
    accuracy_without_mask_on = round(all_goods_without_mask_on / total_number_without_mask_on * 100, 2)

    title = 'Scatter Plot of Coronapas vs 4 different groups\nAccuracy: {0}/{1} ({2}%)\nAccuracy without mask_on: {3}/{4} ({5}%)'.format(
        all_goods, total_number, accuracy, all_goods_without_mask_on, total_number_without_mask_on, accuracy_without_mask_on)
    fig.suptitle(title, fontsize=10)
    plt.legend(loc='lower right')
    plt.xlabel('Distance Score', fontsize=8)
    plt.ylabel('Random numbers to volumetrize 1D plot', fontsize=8)

    if save_img is not None:
        detections_pt = os.path.join(BASE_PT, 'test_results_{}'.format(TEST_NAME))
        if not os.path.isdir(detections_pt):
            os.mkdir(detections_pt)

        img_full_pt = os.path.join(detections_pt, 'scatter_plot_{0}_{1}_thr-{2}_dataset-{3}_identities-{4}.png'.format(
            MODEL_NAME, METRICS, THRESHOLD, REL_PT.split('/')[-1], NUM_OF_IDENTITIES))

        plt.savefig(img_full_pt, dpi=200)
        LOG.write('Saved a plot in {}'.format(img_full_pt))
    if show_img is not None:
        plt.show()


if __name__ == '__main__':
    try:
        LOG.write('================================================================================================')
        LOG.write('Starting Test: Accuracy of "{}" ============================================='.format(TEST_NAME))
        LOG.write('')
        LOG.write('Test: Accuracy of Really A Robot DeepFace identify verifcator on coronapas vs realsense images.')
        LOG.write('Description: In this test, Coronapas of an identity if compared\
        to mask-off/mask-on/mask-off-GAN images taken from #NoFever.')
        LOG.write('ID Verified model used: {}'.format(MODEL_NAME))
        LOG.write('Path to weights: /home/eugenegalaxy/.deepface/weights/vgg_face_weights.h5')
        LOG.write('Face detector backend: {}'.format(DETECTOR_BACKEND))
        LOG.write('Distance Metrics: {}'.format(METRICS))
        LOG.write('Distance Threshold: {}'.format(THRESHOLD))
        LOG.write('Dataset path: {}'.format(IMG_PT))
        LOG.write('Results written to {}'.format(full_log_pt))

        LOG.write('Identities found in a given Dataset: {}'.format(NUM_OF_IDENTITIES))

        t_start = time.time()

        id_results = run_DEEPFACE_through_directory(IMG_PT)

        results_txt = 'results_{0}_{1}_thr-{2}_dataset-{3}_identities-{4}.txt'.format(
            MODEL_NAME, METRICS, THRESHOLD, REL_PT.split('/')[-1], NUM_OF_IDENTITIES)
        results_dir = os.path.join(BASE_PT, 'test_results_{}'.format(TEST_NAME))
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)
        results_txt_full_pt = os.path.join(results_dir, results_txt)
        with open(results_txt_full_pt, 'w') as file:
            file.write(str(id_results))
            file.close
        LOG.write('Saved results dictionary in {}'.format(results_txt_full_pt))

        LOG.write('Generating a plot...')
        plot_results(id_results, save_img=True, show_img=DEBUG_SHOW_IMAGES)

        t_end = time.time()
        LOG.write('Ending mask detection process ================================================================')
        LOG.write('Elapsed time: {:1.2f} seconds'.format(t_end - t_start))
        LOG.write('')
        LOG.write('Results:')
        LOG.write('')

        mask_on, mask_off, mask_gan = split_results(id_results)
        LOG.write('Computing results of Coronapas vs Mask-on identify verification.')
        compute_metrics(mask_on)
        LOG.write('')
        LOG.write('Computing results of Coronapas vs Mask-off identify verification.')
        compute_metrics(mask_off)
        LOG.write('')
        LOG.write('Computing results of Coronapas vs Mask-GAN identify verification.')
        compute_metrics(mask_gan)
        LOG.write('================================================================================================')
        LOG.write('End of a test.')
    except Exception:
        problem = traceback.format_exc()
        DEBUG(problem)
        LOG.write(problem)
