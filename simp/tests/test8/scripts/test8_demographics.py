import os
import time
import pathlib
import traceback
import copy
import random
from itertools import islice
from pprint import pprint
import json
from json.decoder import JSONDecodeError
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

from simp.submodules.nofever.nofever.utils import DebugPrint
from simp.submodules.deepface.deepface import DeepFace


BASE_PT = os.path.dirname(os.path.abspath(__file__))

DEBUG_MODE = True
DEBUG = DebugPrint(DEBUG_MODE)

TEST_NAME = 'demographics'
REL_PT = 'deepface_dataset/identities'
# REL_PT = 'deepface_dataset/test'
IMG_PT = os.path.join(BASE_PT, REL_PT)
NUM_OF_IDENTITIES = len([name for name in os.listdir(IMG_PT)])


AGE_ERROR_TOLERANCE = 5  # Age prediction must be +/- this number to be counted as correct.

AGE_MODEL = DeepFace.build_model('Age')
GENDER_MODEL = DeepFace.build_model('Gender')

_age_model = DeepFace.build_model('Age')
_gender_model = DeepFace.build_model('Gender')
MODEL = {'age': _age_model, 'gender': _gender_model}  # Parsing models into format expected by DeepFace

# detector_backend (string): set face detector backend as mtcnn, opencv, ssd or dlib
DETECTOR_BACKEND = 'mtcnn'

DEBUG_SHOW_IMAGES = None  # Has to be 'None' to disable and anything else to enable


class LogWriter():
    def __init__(self, file_pt):
        self.file_pt = file_pt
        try:
            with open(file_pt, 'a') as file:
                file.close
        except Exception as e:
            LOG.write(e)

    def write(self, msg):
        with open(self.file_pt, 'a') as file:
            t = time.localtime()
            current_time = time.strftime("[%H:%M:%S] ", t)
            file.write(current_time + msg + '\n')
            file.close


log_pt = 'test8_{}_results.txt'.format(TEST_NAME)
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


def load_data(path):
    ''' Returns contents of the first txt file found in the provided path. If no txt file found -> returns None'''
    for txt_file in os.listdir(path):
        txt_full_path = os.path.join(path, txt_file)
        if os.path.isfile(txt_full_path):
            ext = os.path.splitext(txt_full_path)[1]
            if ext == '.txt':
                with open(txt_full_path, 'r') as file:
                    contents = file.read()
                    if len(contents) == 0:
                        return None
                    else:
                        try:
                            contents = json.loads(contents)
                            return contents
                        except JSONDecodeError as e:
                            print(e)
    return None


def analyse_demographics_one_img(img_pt, save_img=None, show_img=None):
    result = DeepFace.analyze(img_path=img_pt, actions=['age', 'gender'],
                              models=MODEL, detector_backend=DETECTOR_BACKEND)
    # Structure the results.
    result['predicted_gender'] = result.pop('gender')
    result['predicted_age'] = result.pop('age')
    result.pop('region')
    path = pathlib.PurePath(img_pt)
    img_filename = os.path.splitext(os.path.basename(img_pt))[0]  # Get a file name without extension from a path.
    result['img'] = os.path.join(path.parent.name, img_filename)
    return result


def analyse_demographics_identity(identity_dir_pt, save_img=None, show_img=None):
    all_demographics_results = []
    id_images = []  # for image plotting
    id_images_pt = []  # for image plotting
    gender_guess_results = []  # for image plotting
    age_guess_results = []  # for image plotting
    for im in os.listdir(identity_dir_pt):
        _, ext = os.path.splitext(os.path.join(identity_dir_pt, im))
        if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
            abs_img_pt = os.path.join(identity_dir_pt, im)
            result = analyse_demographics_one_img(abs_img_pt)
            if result is not None:
                identity_img = parse_image(abs_img_pt)
                id_images.append(identity_img)
                id_images_pt.append(abs_img_pt)
                all_demographics_results.append(result)

    ground_truth = load_data(identity_dir_pt)
    LOG.write('Ground Truth -> Age: {0}  Gender: {1}'.format(ground_truth['age'], ground_truth['gender']))
    for item in all_demographics_results:
        item['ground_gender'] = ground_truth['gender']
        if ground_truth['gender'] == item['predicted_gender']:
            item['gender_guessed'] = True
        else:
            item['gender_guessed'] = False
        gender_guess_results.append(item['gender_guessed'])

        item['ground_age'] = ground_truth['age']
        if item['predicted_age'] >= (ground_truth['age'] - AGE_ERROR_TOLERANCE) and\
          item['predicted_age'] <= (ground_truth['age'] + AGE_ERROR_TOLERANCE):
            item['age_guessed'] = True
        else:
            item['age_guessed'] = False
        age_guess_results.append(item['age_guessed'])

        log_str = 'Img: {0} -> Predicted Age: {1} (Verified: {2})  Predicted gender {3} (Verified: {4})'.format(
            item['img'], item['predicted_age'], item['age_guessed'],
            item['predicted_gender'], item['gender_guessed']).expandtabs(10)
        LOG.write(log_str)

    if save_img is not None or show_img is not None:
        id_images = [x[..., ::-1].copy() for x in id_images]
        rows = 1
        cols = len(id_images)
        fig = plt.figure()
        plt.rc('font', size=8)  # SIZE OF A SUBPLOT TITLE
        title_line_1 = 'Demographics Analysis: Age and Gender\n'
        title_line_2 = "Ground Truth -> Age: {0}, Gender: '{1}', Age error tolerance: {2} years".format(
            ground_truth['age'], ground_truth['gender'], AGE_ERROR_TOLERANCE)
        fig.suptitle(title_line_1 + title_line_2, fontsize=12)

        # Then plot all other identity images to the right from the Coronapas image.
        for idx, im in enumerate(id_images):
            ax = fig.add_subplot(rows, cols, idx + 1)
            filename_no_ext = os.path.splitext(os.path.basename(id_images_pt[idx]))[0]
            id_pt = pathlib.PurePath(id_images_pt[idx])
            identity_short_pt = os.path.join(id_pt.parent.name, filename_no_ext)
            title = 'Target:\n{0}\n Age: {1} (Verified: {2})\n Gender: {3} (Verified: {4})'.format(
                identity_short_pt, all_demographics_results[idx]['predicted_age'],
                age_guess_results[idx], all_demographics_results[idx]['predicted_gender'],
                gender_guess_results[idx])

            ax.title.set_text(title)
            ax.imshow(im)
            ax.axis('off')
        plt.tight_layout()
        if save_img is not None:
            # 'save_img' variable is a path to a directory. Not best practice but convenient here
            save_dir_pt = save_img  # Example: .../deepface_dataset/test_results_identify_verification
            id_name = os.path.basename(identity_dir_pt)
            identity_corona_dir = '{}_demographics'.format(id_name)  # Example : 'eugene'
            save_pt = os.path.join(save_dir_pt, identity_corona_dir)
            if not os.path.isdir(save_pt):
                os.mkdir(save_pt)
            img_filename = '{}_demographics.jpg'.format(id_name)
            img_full_pt = os.path.join(save_pt, img_filename)

            plt.savefig(img_full_pt, dpi=200)
        if show_img is not None:
            plt.show()

    return all_demographics_results


def run_DEEPFACE_through_directory(dir_path):
    dir_name = os.path.basename(dir_path)
    detections_pt = os.path.join(BASE_PT, 'test_results_{}'.format(TEST_NAME), dir_name)
    if not os.path.isdir(detections_pt):
        os.mkdir(detections_pt)
    LOG.write('Images will be saved to: {}'.format(detections_pt))
    LOG.write('')
    LOG.write('Starting identity verification process ================================================================')
    all_results = {}
    for idx, dir_name in enumerate(os.listdir(dir_path)):
        one_demographics_result = []
        LOG.write('Analysing {} demographics.'.format(dir_name))
        identity_pt = os.path.join(dir_path, dir_name)
        results = analyse_demographics_identity(identity_pt, show_img=DEBUG_SHOW_IMAGES, save_img=detections_pt)
        one_demographics_result.append(results)
        all_results[str(idx)] = one_demographics_result  # {'1': {data}, '2':{data} ...}
    return all_results


def split_results(results_dict):
    # Renaming 'mask_on_gan' to 'mask_gan' through out all entries
    for key, value in results_dict.items():
        for idx1, identity in enumerate(value):
            for idx2, verification in enumerate(identity):
                if 'mask_on_gan' in verification['img']:
                    name = results_dict[str(key)][idx1][idx2]['img']
                    id, _ = name.split('/')
                    new_mask_type_name = 'mask_gan'
                    new_name = '{0}/{1}'.format(id, new_mask_type_name)
                    results_dict[str(key)][idx1][idx2]['img'] = new_name

    # Making full independant copies of the result dictionary.
    mask_off = copy.deepcopy(results_dict)
    mask_on = copy.deepcopy(results_dict)
    mask_gan = copy.deepcopy(results_dict)

    # Manually splitting 1 result dict with all classes into 3 separate dicts: mask_on, mask_off, mask_gan
    # ---------------------------------------------------------------------------------------------------------
    for key, value in mask_off.items():
        for idx1, identity in enumerate(value):
            for idx2, verification in enumerate(identity):
                if 'mask_gan' in verification['img']:
                    mask_off[str(key)][idx1].pop(idx2)
    for key, value in mask_off.items():
        for idx1, identity in enumerate(value):
            for idx2, verification in enumerate(identity):
                if 'mask_on' in verification['img']:
                    mask_off[str(key)][idx1].pop(idx2)

    for key, value in mask_gan.items():
        for idx1, identity in enumerate(value):
            for idx2, verification in enumerate(identity):
                if 'mask_on' in verification['img']:
                    mask_gan[str(key)][idx1].pop(idx2)
    for key, value in mask_gan.items():
        for idx1, identity in enumerate(value):
            for idx2, verification in enumerate(identity):
                if 'mask_off' in verification['img']:
                    mask_gan[str(key)][idx1].pop(idx2)

    for key, value in mask_on.items():
        for idx1, identity in enumerate(value):
            for idx2, verification in enumerate(identity):
                if 'mask_gan' in verification['img']:
                    mask_on[str(key)][idx1].pop(idx2)
    for key, value in mask_on.items():
        for idx1, identity in enumerate(value):
            for idx2, verification in enumerate(identity):
                if 'mask_off' in verification['img']:
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
        Produced results for 3 different tests: Demographics Mask-Off, Demographics Mask-On, Demographics Mask-Off-GAN
        ID - "Identity"

        Result metrics:
            Gender Overall Accuracy - (correct 'Woman' + 'Man' predictions) / Total number of Identities
            Gender Man Accuracy - (correct 'Man' predictions) / Total number of 'Man' Identities
            Gender Woman Accuracy - (correct 'Woman' predictions) / Total number of 'Woman' Identities
            Age Overall Accuracy (with 5 years error tolerance):
                (All 'Age' predictions within +/- 5 years from ground-truth) / Total number of Identities
            Age Absolute Mean Error (AAME) ->  sqrt( Sum of all Age errors / Total number of Identities) ^2 )

    '''
    total_images = 0
    total_woman_identitites = 0
    total_man_identitites = 0

    correct_total_ages = 0
    correct_man_ages = 0
    correct_woman_ages = 0
    AAME = []

    correct_total_genders = 0
    correct_man_genders = 0
    correct_woman_genders = 0

    for key, value in mask_dict.items():
        for idx1, identity in enumerate(value):
            for idx2, verification in enumerate(identity):
                total_images += 1
                if verification['ground_gender'] == 'Woman':
                    total_woman_identitites += 1
                else:
                    total_man_identitites += 1

                if verification['age_guessed'] is True:
                    correct_total_ages += 1
                    if verification['ground_gender'] == 'Woman':
                        correct_woman_ages += 1
                    else:
                        correct_man_ages += 1

                if verification['gender_guessed'] is True:
                    correct_total_genders += 1
                    if verification['ground_gender'] == 'Woman':
                        correct_woman_genders += 1
                    else:
                        correct_man_genders += 1

                age_error = abs(verification['ground_age'] - verification['predicted_age'])
                AAME.append(age_error)

    accuracy_total_gender = correct_total_genders / total_images
    accuracy_man_gender = correct_man_genders / total_man_identitites
    accuracy_woman_gender = correct_woman_genders / total_woman_identitites

    accuracy_total_age = correct_total_ages / total_images
    accuracy_man_age = correct_man_ages / total_man_identitites
    accuracy_woman_age = correct_woman_ages / total_woman_identitites

    AAME = sqrt(pow((sum(AAME) / len(AAME)), 2))

    LOG.write('Accuracy Gender (Total): {:1.2f}%'.format(accuracy_total_gender * 100))
    LOG.write('Accuracy Gender (Man): {:1.2f}%'.format(accuracy_man_gender * 100))
    LOG.write('Accuracy Gender (Woman): {:1.2f}%'.format(accuracy_woman_gender * 100))
    LOG.write('Accuracy Age (Total): {:1.2f}%'.format(accuracy_total_age * 100))
    LOG.write('Accuracy Age (Man): {:1.2f}%'.format(accuracy_man_age * 100))
    LOG.write('Accuracy Age (Woman): {:1.2f}%'.format(accuracy_woman_age * 100))
    LOG.write('Age Absolute Mean Error: {:1.2f} years'.format(AAME))


if __name__ == '__main__':
    try:
        LOG.write('================================================================================================')
        LOG.write('Starting Test: Accuracy of "{}" ============================================='.format(TEST_NAME))
        LOG.write('')
        LOG.write('Test: Accuracy of Really A Robot DeepFace demographics (age and gender) realsense images.')
        LOG.write('Description: In this test, scan images of mask-off/mask-on/mask-off-GAN images taken from #NoFever.')
        LOG.write('Demographics models used: {0} and {1}'.format(AGE_MODEL, GENDER_MODEL))
        LOG.write('Age model weights: /home/eugenegalaxy/.deepface/weights/age_model_weights.h5')
        LOG.write('Gender model weights: /home/eugenegalaxy/.deepface/weights/gender_model_weights.h5')
        LOG.write('Face detector backend: {}'.format(DETECTOR_BACKEND))
        LOG.write('Allowed age error tolerance: {}'.format(AGE_ERROR_TOLERANCE))
        LOG.write('Dataset path: {}'.format(IMG_PT))
        LOG.write('Results written to {}'.format(full_log_pt))
        LOG.write('Identities found in a given Dataset: {}'.format(NUM_OF_IDENTITIES))

        t_start = time.time()

        id_results = run_DEEPFACE_through_directory(IMG_PT)

        results_txt = 'results_{0}_age-tolerance-{1}_dataset-{2}_num-of-ids-{3}.txt'.format(
            TEST_NAME, AGE_ERROR_TOLERANCE, REL_PT.split('/')[-1], NUM_OF_IDENTITIES)
        results_dir = os.path.join(BASE_PT, 'test_results_{}'.format(TEST_NAME))
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)
        results_txt_full_pt = os.path.join(results_dir, results_txt)
        with open(results_txt_full_pt, 'w') as file:
            pprint(id_results, stream=file)
            file.close
        LOG.write('Saved results dictionary in {}'.format(results_txt_full_pt))

        t_end = time.time()
        LOG.write('Ending demographics analysis process ================================================================')
        LOG.write('Elapsed time: {:1.2f} seconds'.format(t_end - t_start))
        LOG.write('')
        LOG.write('Results:')
        LOG.write('')

        mask_on, mask_off, mask_gan = split_results(id_results)

        LOG.write('Computing results of mask-on race and gender demographics analysis.')
        compute_metrics(mask_on)
        LOG.write('')
        LOG.write('Computing results of mask-off race and gender demographics analysis.')
        compute_metrics(mask_off)
        LOG.write('')
        LOG.write('Computing results of mask-GAN race and gender demographics analysis.')
        compute_metrics(mask_gan)
        LOG.write('================================================================================================')
        LOG.write('End of a test.')
        LOG.write('')
    except Exception:
        problem = traceback.format_exc()
        DEBUG(problem)
        LOG.write(problem)
