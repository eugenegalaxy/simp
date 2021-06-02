import numpy as np

from simp.submodules.nofever.nofever.forehead_scanner import ForeheadFinder, BoundingBox, SingleDetection
from simp.submodules.nofever.nofever.utils import LOG, CONFIG_NOFEVER_SETTINGS, DebugPrint

DEBUG_MODE = True
DEBUG = DebugPrint(DEBUG_MODE)

SETTINGS = CONFIG_NOFEVER_SETTINGS['mask_scanner']

LABEL_MASK = SETTINGS['LABEL_MASK']
LABEL_NOMASK = SETTINGS['LABEL_NOMASK']
LABEL_WRONGMASK = SETTINGS['LABEL_WRONGMASK']


class MaskFinder(ForeheadFinder):

    MASK_BB_CENTER_ROI = SETTINGS.getint('MASK_BB_CENTER_ROI')
    MASK_MAX_DISTANCE = SETTINGS.getfloat('MASK_MAX_DISTANCE')
    MASK_DIST_THRESHOLD = SETTINGS.getfloat('MASK_DIST_THRESHOLD')

    def __init__(self):
        # Variables for storing bounding box coordinates for each class
        self.mask_on = []    # id 0
        self.no_mask = []    # id 1
        self.mask_wrong = []  # id 2

    def get_class_list(self):
        return [self.mask_on, self.no_mask, self.mask_wrong]

    def predict_mask(self, findings, print_enable=False):
        self.unpack(findings)

        all_dets = self.get_class_list()
        if self.is_list_empty(all_dets):  # ForeheadFinder method
            return None

        self.compute_3D_coords()

        self.filter_duplicates()

        all_dets = self.get_class_list()
        if self.is_list_empty(all_dets):
            return None

        self.filter_long_range(max_dist=self.MASK_MAX_DISTANCE)

        all_dets = self.get_class_list()
        if self.is_list_empty(all_dets):
            return None

        self.filter_background(dist_thresh=self.MASK_DIST_THRESHOLD)

        all_dets = self.get_class_list()
        if self.is_list_empty(all_dets):
            return None

        if print_enable:
            self.print_detections()  # ForeheadFinder method

        all_dets_flat = [item for sublist in all_dets for item in sublist]
        det_data = [[item.label, item.conf] for item in all_dets_flat]

        if det_data:
            return det_data
        else:
            return None

    def unpack(self, findings_list):
        if len(findings_list) > 0:
            for det in findings_list:
                det_label = det[-1]  # last in the list.
                det[1] = [int(x) for x in det[1]]  # converting BB pixels from floats to ints.
                if det_label == LABEL_MASK:
                    mask_bb = BoundingBox(det[1][0], det[1][1], det[1][2], det[1][3])
                    mask = SingleDetection(det[0], mask_bb, det[2], det[3])
                    self.mask_on.append(mask)
                elif det_label == LABEL_NOMASK:
                    nomask_bb = BoundingBox(det[1][0], det[1][1], det[1][2], det[1][3])
                    nomask = SingleDetection(det[0], nomask_bb, det[2], det[3])
                    self.no_mask.append(nomask)
                elif det_label == LABEL_WRONGMASK:
                    wrongmask_bb = BoundingBox(det[1][0], det[1][1], det[1][2], det[1][3])
                    wrongmask = SingleDetection(det[0], wrongmask_bb, det[2], det[3])
                    self.mask_wrong.append(wrongmask)
                else:
                    pass  # place for future error prevention here.
        else:
            pass

    def compute_3D_coords(self):
        ''' Compute average depth for ALL bounding boxes of all detections,
            existing in Detections class until this moment.'''

        all_dets = self.get_class_list()

        for det_class in all_dets:
            if len(det_class) > 0:
                for unique_det in det_class:
                    xyz = self.get_xyz_bounding_box(unique_det, ROI_size=self.MASK_BB_CENTER_ROI)
                    xyz_has_nan = np.isnan(np.sum(xyz))
                    if not xyz_has_nan and xyz[2] != 0:
                        dist = self.get_3D_dist_two_points(xyz)
                        unique_det.point3D.x, unique_det.point3D.y, unique_det.point3D.z = xyz
                        unique_det.dist = round(dist, 4)
                    else:
                        dist = 0
                        unique_det.point3D.x, unique_det.point3D.y, unique_det.point3D.z = 0, 0, 0

    def filter_duplicates(self):
        all_dets = self.get_class_list()

        for i, det_class in enumerate(all_dets):
            # first filter classes that do not expect more than 1 occurence (everything but eyes and ears)
            if len(det_class) >= 2:
                closest_dist = 10000  # just a big number
                remaining_detection = None
                for unique_det in det_class:
                    if unique_det.dist <= closest_dist:
                        closest_dist = unique_det.dist
                        remaining_detection = unique_det

                if remaining_detection is not None:
                    label = unique_det.label
                    if label == LABEL_MASK:
                        self.mask_on = [remaining_detection]
                    elif label == LABEL_NOMASK:
                        self.no_mask = [remaining_detection]
                    elif label == LABEL_WRONGMASK:
                        self.mask_wrong = [remaining_detection]
                else:
                    if label == LABEL_MASK:
                        self.mask_on = []
                    elif label == LABEL_NOMASK:
                        self.no_mask = []
                    elif label == LABEL_WRONGMASK:
                        self.mask_wrong = []

    def filter_long_range(self, max_dist=0.5):
        """ Remove all detections that are farther from camera than 'max_dist' meters.

        Args:
            max_dist (float): maximum distance in meters where the detected mask status bounding box must
                    be WITHIN max_dist range in order to be accepted.. Defaults to 0.5.
        """
        all_dets = self.get_class_list()
        all_dets_flat = [item for sublist in all_dets for item in sublist]
        detections_to_remove = [i for i in all_dets_flat if i.dist > max_dist]
        for det in detections_to_remove:
            print('Removing long range detection')
            label = det.label
            if label == LABEL_MASK:
                self.mask_on.remove(det)
            elif label == LABEL_NOMASK:
                self.no_mask.remove(det)
            elif label == LABEL_WRONGMASK:
                self.mask_wrong.remove(det)

    def filter_background(self, dist_thresh=0.1):
        """Finds closest detection to camera. Computes distance from this detection to all other detections.
        If other detection is farther than dist_thresh from closest_detection -> remove it from the system.

        Args:
            dist_thresh (float): distance threshold in meters. Detections located closer than dist_thresh to each other
                    are counted as part of the same head.. Defaults to 0.1.
        """
        all_dets = self.get_class_list()
        detections_to_remove = []
        all_dets_flat = [item for sublist in all_dets for item in sublist]
        closest_detection = min(all_dets_flat, key=lambda x: x.dist)

        for det in all_dets_flat:
            dist = self.get_3D_dist_two_points(det.point3D.xyz(), start_point=closest_detection.point3D.xyz())
            if dist >= dist_thresh:
                print('Filtering background mask detection')
                detections_to_remove.append(det)

        for det in detections_to_remove:
            label = det.label
            if label == LABEL_MASK:
                self.mask_on.remove(det)
            elif label == LABEL_NOMASK:
                self.no_mask.remove(det)
            elif label == LABEL_WRONGMASK:
                self.mask_wrong.remove(det)

    # ======================== GARBAGE ===========================
    # Methods inherited from Foreheadfinder parent class that cannot be used in MaskFinder child class.
    # Will raise "AttributeError" if attempted to be used in code.
    @property
    def getForeheadHeight(self, findings, print_enable=False):
        raise AttributeError("'MaskFinder' object has no attribute 'getForeheadHeight'")

    @property
    def show_img_forehead_one_by_one(self):
        raise AttributeError("'MaskFinder' object has no attribute 'show_img_forehead_one_by_one'")

    @property
    def show_img_forehead_averaged(self):
        raise AttributeError("'MaskFinder' object has no attribute 'show_img_forehead_averaged'")

    @property
    def show_img_forehead_final(self, forehead_xyz):
        raise AttributeError("'MaskFinder' object has no attribute 'show_img_forehead_final'")

    @property
    def compute_forehead_coords(self):
        raise AttributeError("'MaskFinder' object has no attribute 'compute_forehead_coords'")

    @property
    def compute_avg_forehead(self):
        raise AttributeError("'MaskFinder' object has no attribute 'compute_avg_forehead'")

    @property
    def get_forehead_from_point3D(self, det):
        raise AttributeError("'MaskFinder' object has no attribute 'get_forehead_from_point3D'")


class MaskSession(object):

    RATIO_MASK_WRONG = SETTINGS.getfloat('RATIO_MASK_WRONG')
    RATIO_MASK_ON = SETTINGS.getfloat('RATIO_MASK_ON')
    RATIO_MASK_OFF = SETTINGS.getfloat('RATIO_MASK_OFF')

    CONF_MASK_WRONG = SETTINGS.getfloat('CONF_MASK_WRONG')
    CONF_MASK_ON = SETTINGS.getfloat('CONF_MASK_ON')
    CONF_MASK_OFF = SETTINGS.getfloat('CONF_MASK_OFF')
    AVG_CONF_RATIO = SETTINGS.getfloat('AVG_CONF_RATIO')
    MAX_CONF_RATIO = SETTINGS.getfloat('MAX_CONF_RATIO')

    def __init__(self):
        # Variables for storing bounding box coordinates for each class
        self.mask_on = []    # id 0
        self.no_mask = []    # id 1
        self.mask_wrong = []  # id 2

    def save_detection(self, detection):
        for single_det in detection:
            if single_det[0] == LABEL_NOMASK:
                self.no_mask.append(single_det)
            elif single_det[0] == LABEL_MASK:
                self.mask_on.append(single_det)
            elif single_det[0] == LABEL_WRONGMASK:
                self.mask_wrong.append(single_det)

    def getEstimate(self):
        """
        Get final estimate of the mask status based on saved detections in this MaskSession.
        Based on class frequency, average confidence, and max confidence.

        return: mask status string. 'mask_on', 'mask_off', 'mask_wrong', 'no_detections'.
        """

        if len(self.mask_on + self.no_mask + self.mask_wrong) == 0:
            return 'no_detections'

        ratio_on, ratio_no, ratio_wrong, total_count = self._compute_count()
        conf_on, conf_no, conf_wrong = self._compute_confidence(avg_ratio=self.AVG_CONF_RATIO,
                                                                max_ratio=self.MAX_CONF_RATIO)

        status_str1 = "Frequency/Confidence: 'mask_on': {0} / {1}, 'mask_off': {2}".format(
            round(ratio_on, 4), round(conf_on, 4), round(ratio_no, 4))
        status_str2 = " / {0}, 'mask_wrong': {1} / {2}. Total: {3}".format(
            round(conf_no, 4), round(ratio_wrong, 4), round(conf_wrong, 4), total_count)
        status_str = status_str1 + status_str2
        LOG.warning(status_str)
        DEBUG(status_str)

        # Decision logics. TODO TUNE IT TO INCREASE ACCURACY!
        if ratio_wrong >= self.RATIO_MASK_WRONG and conf_wrong > self.CONF_MASK_WRONG:
            return LABEL_WRONGMASK
        if ratio_wrong < self.RATIO_MASK_WRONG:
            if conf_wrong > conf_on and conf_wrong > conf_no:
                return LABEL_WRONGMASK

        if ratio_no != 0 and ratio_on != 0:
            if ratio_no >= (ratio_on * 3):
                return LABEL_NOMASK
            else:
                return LABEL_MASK

        if ratio_no >= self.RATIO_MASK_OFF:
            return LABEL_NOMASK

        if ratio_on >= self.RATIO_MASK_ON and conf_on >= self.CONF_MASK_ON:
            return LABEL_MASK
        if ratio_on >= self.RATIO_MASK_ON and conf_on <= self.CONF_MASK_ON:
            return LABEL_NOMASK

        if ratio_on < self.RATIO_MASK_ON:
            if conf_on > conf_wrong and conf_on > conf_no:
                return LABEL_MASK

        if True:
            # DEV NOTE: Using this to catch all cases that are not coded above.
            LOG.warning('Unexpected situation in mask estimation logics.')
            LOG.warning('ratio_wrong = {}'.format(ratio_wrong))
            LOG.warning('ratio_no = {}'.format(ratio_no))
            LOG.warning('ratio_on = {}'.format(ratio_on))
            LOG.warning('conf_wrong = {}'.format(conf_wrong))
            LOG.warning('conf_no = {}'.format(conf_no))
            LOG.warning('conf_on = {}'.format(conf_on))
            LOG.warning('self.RATIO_MASK_WRONG = {}'.format(self.RATIO_MASK_WRONG))
            LOG.warning('self.RATIO_MASK_OFF = {}'.format(self.RATIO_MASK_OFF))
            LOG.warning('self.RATIO_MASK_ON = {}'.format(self.RATIO_MASK_ON))
            LOG.warning('self.CONF_MASK_WRONG = {}'.format(self.CONF_MASK_WRONG))
            LOG.warning('self.CONF_MASK_OFF = {}'.format(self.CONF_MASK_OFF))
            LOG.warning('self.CONF_MASK_ON = {}'.format(self.CONF_MASK_ON))
            return 'no_detections'

    def _compute_count(self):
        """
        Computes ratio for each class in this MaskSession.
        Conf.score is computed as follows:
            number of class detections / total number of all detections

        returns: ratio for each class in this MaskSession and total_count of detections.
        """
        c_mask_on = len(self.mask_on)
        c_no_mask = len(self.no_mask)
        c_mask_wrong = len(self.mask_wrong)
        total_count = c_mask_on + c_no_mask + c_mask_wrong
        ratio_wrong = c_mask_wrong / total_count
        ratio_on = c_mask_on / total_count
        ratio_no = c_no_mask / total_count

        return ratio_on, ratio_no, ratio_wrong, total_count

    def _compute_confidence(self, avg_ratio=0.5, max_ratio=0.5):
        '''
        Computes confidence score for each class in this MaskSession.
        Conf.score is computed as follows:
            (avg_ratio * average_confidence of a class) + (max_ratio * max_confidence of a class)

        returns: confidence scores for each class in this MaskSession.
        '''
        avg_mask_on = 0
        max_mask_on = 0
        avg_no_mask = 0
        max_no_mask = 0
        avg_mask_wrong = 0
        max_mask_wrong = 0

        if self.mask_on:
            all_mask_on = [x[1] for x in self.mask_on]
            avg_mask_on = sum(all_mask_on) / len(all_mask_on)
            max_mask_on = max(all_mask_on)
        if self.no_mask:
            all_no_mask = [x[1] for x in self.no_mask]
            avg_no_mask = sum(all_no_mask) / len(all_no_mask)
            max_no_mask = max(all_no_mask)
        if self.mask_wrong:
            all_mask_wrong = [x[1] for x in self.mask_wrong]
            avg_mask_wrong = sum(all_mask_wrong) / len(all_mask_wrong)
            max_mask_wrong = max(all_mask_wrong)

        # DEBUG("Avrg confidences: {0} 'mask_on',      {1} 'mask_off',      {2} 'mask_wrong'.".format(
        #     round(avg_mask_on, 4), round(avg_no_mask, 4), round(avg_mask_wrong, 4)))
        # DEBUG("Maxi confidences: {0} 'mask_on',      {1} 'mask_off',      {2} 'mask_wrong'.".format(
        #     round(max_mask_on, 4), round(max_no_mask, 4), round(max_mask_wrong, 4)))

        conf_on = (avg_ratio * avg_mask_on) + (max_ratio * max_mask_on)
        conf_no = (avg_ratio * avg_no_mask) + (max_ratio * max_no_mask)
        conf_wrong = (avg_ratio * avg_mask_wrong) + (max_ratio * max_mask_wrong)

        return conf_on, conf_no, conf_wrong

    def clear_detections(self):
        self.mask_on = []    # id 0
        self.no_mask = []    # id 1
        self.mask_wrong = []  # id 2
