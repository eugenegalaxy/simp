import traceback
import cv2
import os

from simp.submodules.nofever.nofever.utils import DebugPrint
from simp.data_parser import DataParser
from simp.face_generate import detect_and_generate
from simp.submodules.deepface.deepface import DeepFace


BASE_PT = os.path.dirname(os.path.abspath(__file__))

DEBUG_MODE = True
DEBUG = DebugPrint(DEBUG_MODE)

# yolo_rel_path = 'submodules/nofever/nofever/detection/yolov5'
# full_yolo_path = os.path.join(BASE_PT, yolo_rel_path)
# print(full_yolo_path)
# sys.path.insert(0, full_yolo_path)


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
            file.write(msg + '\n')
            file.close


log_pt = 'outputs/simp_scans.txt'
full_log_pt = os.path.join(BASE_PT, log_pt)
LOG = LogWriter(full_log_pt)


def imshow(img):
    cv2.imshow('Image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def GO(reason):
    DEBUG(' ')
    DEBUG(reason)
    DEBUG('GO | SIMP Scanning succeeded, please come in!')
    LOG.write(' ')
    LOG.write(reason)
    LOG.write('GO | SIMP Scanning succeeded, please come in!')
    LOG.write('=======================================================================================')
    quit()


def NOGO(reason):
    DEBUG(' ')
    DEBUG(reason)
    DEBUG('NO GO | SIMP Scanning failed, please do a secondary inspection by one of the employees.')
    LOG.write(' ')
    LOG.write(reason)
    LOG.write('NO GO | SIMP Scanning failed, please do a secondary inspection by one of the employees.')
    LOG.write('=======================================================================================')
    quit()


def run_simp(qr_code_pt, nofever_scan_pt):
    ''' THIS IS THE MVP OF THE PIPELINE.
        All inputs will be initially stored as files in a directory.
    '''
    DP = DataParser()
    LOG.write('======================================= NEW SCAN ======================================')
    LOG.write('QR-CODE: {}'.format(qr_code_pt))
    LOG.write('NoFever scan: {}'.format(nofever_scan_pt))
    # =======================================================================================================
    # =======================================================================================================
    DEBUG('STEP 1 - Parsing NoFever fresh scan -> Image + Temperature/Mask Status')
    # nofever_scan_pt = 'inputs/nofever_data/eugene'
    scan_img, scan_img_name, scan_img_full_pt, scan_data = DP.parse_nofever_scan(nofever_scan_pt)
    temperature = scan_data['temp']
    mask_status = scan_data['mask_status']
    LOG.write('STEP 1 - OK (Parsing NoFever fresh scan -> Image + Temperature/Mask Status')
    LOG.write(' -> NoFever scan image path: {}'.format(scan_img_full_pt))
    LOG.write(' -> Temperature: {0} | Mask Status: {1}'.format(temperature, mask_status))
    # =======================================================================================================
    # =======================================================================================================
    DEBUG('STEP 2 - Parsing Coronapass QR code -> Image + Name/Age/Vaccination Status. Input can vary.')
    # STEP 2: Parse Corona pass of that user from an image (To be changed to a real nofever scanning)
    #   take a photo(LinkedIn?), QR code -> json/dict with name, age, vaccination
    # qr_code_pt = 'inputs/coronapas_qr/eugene.jpg'

    pass_img  = None
    status = None
    name = None
    given_age = None
    given_gender = None

    pass_img_full_pt, pass_data = DP.parse_coronapas(qr_code_pt)

    LOG.write('STEP 2 - OK (Parsing Coronapass QR code)')

    if pass_img_full_pt is not None:
        pass_img = pass_img_full_pt
        LOG.write(' -> Coronapas img path read from QR: {}'.format(pass_img))
    if pass_data is not None:
        LOG.write(' -> Coronapas data read from QR: {}'.format(pass_data))
        if "coronapas" not in pass_data:
            LOG.write('Coronapas document does not contain COVID-19 status (valid/not valid). Something wrong.')
            LOG.write('=======================================================================================')
            raise SystemError('Coronapas document does not contain COVID-19 status (valid/not valid). Something wrong.')
        else:
            status = pass_data["coronapas"]
        if "name" in pass_data:
            name = pass_data["name"]
        if "age" in pass_data:
            given_age = pass_data["age"]
        if "gender" in pass_data:
            given_gender = pass_data["gender"]
    else:
        LOG.write('No coronapas document found. Something wrong. Quitting')
        LOG.write('=======================================================================================')
        raise SystemError('No coronapas document found. Something wrong.')

    # =======================================================================================================
    # =======================================================================================================
    DEBUG('STEP 3 - Checking Coronapas for VALID / NOT VALID.')
    if status != 'valid':
        NOGO('Coronapas must be valid (GYLDIGT). Yours is not valid (IKKE GYLDIGT)')
    DEBUG('CORONAPAS - Passed (valid)')
    LOG.write('STEP 3 - OK (Checking Coronapas for VALID / NOT VALID.)')
    # =======================================================================================================
    # =======================================================================================================
    DEBUG('STEP 4 - Checking whether face mask is detected.')
    if mask_status == 'mask_off' or mask_status == 'no_detections':
        NOGO('Face mask has not been detected.')

    if mask_status == 'mask_on':
        DEBUG('FACEMASK - Passed (detected)')
        LOG.write('STEP 4 - OK (Checking whether face mask is detected.)')
    elif mask_status == 'mask_wrong':
        DEBUG('FACEMASK - Passed (detected, but make sure it covers your nose and mouth!)')
        LOG.write('STEP 4 - OK (Checking whether face mask is detected.)')

    # =======================================================================================================
    # =======================================================================================================
    DEBUG('STEP 5 - Checking body temperature')
    if temperature <= 20:
        NOGO('Non-human temperature measured (Below 20C). Potential fraud detected.\
        If it is a hat or hair, please remove it and repeat the scan.')
    elif temperature >= 38:
        NOGO('Fever temperature measured (Above 38C). Due to vaccination/tests uncertainty, we ask you to leave.')
    DEBUG('TEMPERATURE - Passed (measured result is between 20C and 38C)')

    LOG.write('STEP 5 - OK (Checking body temperature to be between 20C and 38C)')
    # =======================================================================================================
    # =======================================================================================================
    DEBUG('STEP 6 - Identify validation')
    LOG.write('STEP 6 - In Progerss (Identify validation)')
    # Case 1: Coronapas has only status, no photo or demographics available.
    if pass_img is None and given_age is None and given_gender is None:
        GO('No coronapas secondary data available. Therefore, good to go...')
    # Case 2: Coronapas has a photo available
    elif pass_img is not None:
        if mask_status == 'mask_on' or mask_status == 'mask_wrong':
            DEBUG('Unmasking nofever scan photo')
            LOG.write(' -> Unmasking nofever scan photo')
            gan_scan_img_pt = 'simp/outputs/gan/gan_' + scan_img_name
            LOG.write(' -> Saved GAN img of NoFever scan: {}'.format(gan_scan_img_pt))
            detect_and_generate(scan_img, save_gan_img_to=gan_scan_img_pt)
            GAN_str = '(GAN  generated)'  # for later prints
        else:
            DEBUG('Unmasking nofever scan photo - SKIPPED. Target has no mask detected.')
            LOG.write(' -> Unmasking nofever scan photo - SKIPPED. Target has no mask detected.')
            GAN_str = '(Original photo)'  # for later prints

        DEBUG('Running identity verification on both nofever and coronapass photos')
        LOG.write(' -> Running identity verification on both nofever and coronapass photos')
        pass_vs_gan  = DeepFace.verify(pass_img_full_pt, gan_scan_img_pt)
        pass_vs_facemasked  = DeepFace.verify(pass_img_full_pt, scan_img_full_pt)  # Just curious...
        gan_str = 'Coronapass vs GAN-generated NoFever scan: {0}, {1:1.4f} identity distance'.format(
            pass_vs_gan['verified'], pass_vs_gan['distance'])
        orig_str = 'Coronapass vs Original NoFever scan: {0}, {1:1.4f} identity distance'.format(
            pass_vs_facemasked['verified'], pass_vs_facemasked['distance'])
        DEBUG(gan_str)
        DEBUG(orig_str)
        LOG.write(' -> ' + gan_str)
        LOG.write(' -> ' + orig_str)

        verification_bool = pass_vs_gan['verified']
        verification_distance = pass_vs_gan['distance']

        if verification_bool is True:
            GO('Identity has been verified, distance = {}'.format(verification_distance))
        else:
            if verification_distance >= 0.8:
                NOGO('Identity is not verified by comparing 2 photos.\
                Distance score is too far to run demogrpahics. Distance = {}'.format(verification_distance))
            else:  # The "Not identified but the distance is not too bad" case
                DEBUG('Identity is not verified by comparing 2 photos. Running demographics comparison.')
                LOG.write(' -> Identity is not verified by comparing 2 photos. Running demographics comparison.')
                # List of things to scan from a Coronapas Photo. Starting with race as it is never present in coronapas.
                # Age and Gender might appear in Coronapas. If yes, we won't analyze age/gender from the coronapas photo
                demographics_to_scan = ['race']

                # Will fill with either photo-scanned or given demographics
                coronapas_demographics = {'race': [], 'age': [], 'gender': []}

                if given_age is None:
                    demographics_to_scan.append('age')
                else:
                    coronapas_demographics['age'] = given_age

                if given_gender is None:
                    demographics_to_scan.append('gender')
                else:
                    coronapas_demographics['gender'] = given_gender

                if mask_status == 'mask_on' or mask_status == 'mask_wrong':
                    pass_demographics = DeepFace.analyze(pass_img_full_pt, actions=demographics_to_scan)
                    scan_demographics = DeepFace.analyze(gan_scan_img_pt, actions=['age', 'gender', 'race'])
                    # Can run pass vs scan orig demogrpahics here, if wanted. Skipped due to long processing time.
                else:
                    pass_demographics = DeepFace.analyze(pass_img_full_pt, actions=demographics_to_scan)
                    scan_demographics = DeepFace.analyze(scan_img_full_pt, actions=['age', 'gender', 'race'])

                scan_age = scan_demographics['age']
                scan_gender = scan_demographics['gender']
                scan_race = scan_demographics['dominant_race']

                if 'age' in pass_demographics:
                    coronapas_demographics['age'] = pass_demographics['age']
                if 'gender' in pass_demographics:
                    coronapas_demographics['gender'] = pass_demographics['gender']
                if 'dominant_race' in pass_demographics:  # always True
                    coronapas_demographics['race'] = pass_demographics['dominant_race']

                coronapas_demo_str = 'Coronapass demographics -> Age: {0} | Gender: {1} | Race: {2}'.format(
                    coronapas_demographics['age'], coronapas_demographics['gender'], coronapas_demographics['race'])
                nofever_demo_str = 'NoFever {0} demographics -> Age: {1} | Gender: {2} | Race: {3}'.format(
                    GAN_str, scan_age, scan_gender, scan_race)
                DEBUG(coronapas_demo_str)
                DEBUG(nofever_demo_str)
                LOG.write(' -> ' + coronapas_demo_str)
                LOG.write(' -> ' + nofever_demo_str)

                demographics_status_msg = ''
                demo_status = 0
                if coronapas_demographics['age'] <= (scan_age + 5) and coronapas_demographics['age'] >= (scan_age - 5):
                    demographics_status_msg += 'AGE: Passed | '
                    demo_status += 1
                if coronapas_demographics['gender'] == scan_gender:
                    demographics_status_msg += 'GENDER: Passed | '
                    demo_status += 1
                if coronapas_demographics['race'] == scan_race:
                    demographics_status_msg += 'RACE: Passed'
                    demo_status += 1

                DEBUG(demographics_status_msg)
                LOG.write(' -> ' + demographics_status_msg)

                if demo_status == 3:
                    GO('Demographics comparison has been passed. Please come in.')
                else:
                    NOGO('Demographics comparison has NOT been passed.')

    # Case 3: No Coronapas photo is available, some demographics data is available
    elif pass_img is None and (given_age is not None or given_gender is not None):
        DEBUG('Coronapas has no photo present, but has some demographics. Comparing those.')
        LOG.write(' -> ' + 'Coronapas has no photo present, but has some demographics. Comparing those.')
        DEBUG('Unmasking nofever scan photo')

        if mask_status == 'mask_on' or mask_status == 'mask_wrong':
            DEBUG('Unmasking nofever scan photo')
            LOG.write(' -> ' + 'Unmasking nofever scan photo')
            gan_scan_img_pt = 'simp/outputs/gan/gan_' + scan_img_name
            detect_and_generate(scan_img, save_gan_img_to=gan_scan_img_pt)
            GAN_str = '(GAN  generated)'  # for later prints
        else:
            DEBUG('Unmasking nofever scan photo - SKIPPED. Target has no mask detected.')
            LOG.write(' -> ' + 'Unmasking nofever scan photo - SKIPPED. Target has no mask detected.')
            GAN_str = '(Original photo)'  # for later prints

        coronapas_demographics = {}
        demographics_to_scan = []
        if given_age is not None:
            coronapas_demographics.update({'age': given_age})
            demographics_to_scan.append('age')
        if given_gender is not None:
            coronapas_demographics.update({'gender': given_gender})
            demographics_to_scan.append('gender')

        if mask_status == 'mask_on' or mask_status == 'mask_wrong':
            scan_demographics = DeepFace.analyze(gan_scan_img_pt, actions=demographics_to_scan)
        else:
            scan_demographics = DeepFace.analyze(scan_img_full_pt, actions=demographics_to_scan)

        coronapas_demo_str = 'Coronapass demographics -> {}'.format(coronapas_demographics)
        nofever_demo_str = 'NoFever {0} demographics -> {1}'.format(GAN_str, scan_demographics)
        DEBUG(coronapas_demo_str)
        DEBUG(nofever_demo_str)
        LOG.write(' -> ' + coronapas_demo_str)
        LOG.write(' -> ' + nofever_demo_str)

        demographics_status_msg = ''
        demo_status = []
        if 'age' in coronapas_demographics:
            if coronapas_demographics['age'] <= (scan_demographics['age'] + 5) \
               and coronapas_demographics['age'] >= (scan_demographics['age'] - 5):

                demographics_status_msg += 'AGE: Passed | '
                demo_status.append(True)
            else:
                demo_status.append(False)
        if 'gender' in coronapas_demographics:
            if coronapas_demographics['gender'] == scan_demographics['gender']:
                demographics_status_msg += 'GENDER: Passed'
                demo_status.append(True)
            else:
                demo_status.append(False)

        DEBUG(demographics_status_msg)
        LOG.write(' -> ' + demographics_status_msg)

        if False in demo_status:
            NOGO('Demographics comparison has NOT been passed.')
        else:
            GO('Demographics comparison has been passed. Please come in.')

    if True:
        LOG.write('Case you have not experienced! Do better work.')
        LOG.write('=======================================================================================')
        raise SystemError('Case you have not experienced! Do better work.')


if __name__ == '__main__':
    try:
        qr_code_pt = 'inputs/coronapas_qr/hugo.jpg'
        nofever_scan_pt = 'inputs/nofever_data/stranger_2'
        run_simp(qr_code_pt, nofever_scan_pt)
    except Exception:
        problem = traceback.format_exc()
        DEBUG(problem)
