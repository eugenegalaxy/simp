import random
import cv2
import numpy as np
import pyrealsense2 as rs

from simp.submodules.nofever.nofever.utils import LOG, CONFIG_NOFEVER_SETTINGS, DebugPrint

DEBUG_MODE = False
DEBUG = DebugPrint(DEBUG_MODE)

SETTINGS = CONFIG_NOFEVER_SETTINGS['forehead_scanner']


# ===================================================================================
# =================================  CONTAINERS  ====================================
class BoundingBox(object):
    ''' Container for bounding box coordinates in YOLOv5 annotation format:
        [X_center, Y_center, Width, Height] '''
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def xywh(self):
        return [self.x, self.y, self.w, self.h]


class Point3D(object):
    ''' Container for coordinates in 3D space: [x, y, z] '''
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def xyz(self):
        ''' return x,y,z as a list'''
        return [self.x, self.y, self.z]


class SingleDetection(object):
    ''' Container for data of a single detection '''
    def __init__(self, idx, bb, conf, label):
        self.id = idx                     # Class ID
        self.bb = bb                      # Bounding box coordinates (x_center, y_center, width, height)
        self.conf = conf                  # Confidence score (probability)
        self.label = label                # Class label
        self.point3D = Point3D(0, 0, 0)   # 3D coordinate of the central point of the bounding box [x, y, z]
        self.dist = 0                     # Distance from camera base to detecton center in 3D coordinate space (meters)
        self.forehead = Point3D(0, 0, 0)  # Relative position of a forehead, in relation to detection.
        self.forehead_dist = 0            # Distance from camera base to forehead center in 3D coordinate space (meters)


class Forehead(object):
    ''' Container for data of a single detection '''
    def __init__(self, idx, label, height, dist, xyz, objects):
        self.id = idx
        self.label = label
        self.height = height
        self.dist = dist
        self.xyz = xyz
        self.objects = objects


class ForeheadFinder(object):
    ''' Class to process and compute detected classes. '''
    LABEL_EYE = SETTINGS['LABEL_EYE']
    LABEL_LEFT_EYE = SETTINGS['LABEL_LEFT_EYE']
    LABEL_RIGHT_EYE = SETTINGS['LABEL_RIGHT_EYE']
    LABEL_HEAD = SETTINGS['LABEL_HEAD']
    LABEL_NOSE = SETTINGS['LABEL_NOSE']
    LABEL_MOUTH = SETTINGS['LABEL_MOUTH']
    LABEL_GLASSES = SETTINGS['LABEL_GLASSES']
    LABEL_SUNGLASSES = SETTINGS['LABEL_SUNGLASSES']

    IGNORE_DIST_LEFT = SETTINGS.getfloat('IGNORE_DIST_LEFT')
    IGNORE_DIST_RIGHT = SETTINGS.getfloat('IGNORE_DIST_RIGHT')
    IGNORE_DIST_FAR = SETTINGS.getfloat('IGNORE_DIST_FAR')
    IGNORE_DIST_CLOSE = SETTINGS.getfloat('IGNORE_DIST_CLOSE')

    FACE_BB_CENTER_ROI = SETTINGS.getint('FACE_BB_CENTER_ROI')

    DIST_EYE_PAIR_MIN = SETTINGS.getfloat('DIST_EYE_PAIR_MIN')
    DIST_EYE_PAIR_MAX = SETTINGS.getfloat('DIST_EYE_PAIR_MAX')

    DIST_SAME_HEAD = SETTINGS.getfloat('DIST_SAME_HEAD')

    INCLUDE_HEAD = SETTINGS.getboolean('INCLUDE_HEAD')
    INCLUDE_NOSE = SETTINGS.getboolean('INCLUDE_NOSE')

    CENTER_ROI_UPPER_STD = SETTINGS.getfloat('CENTER_ROI_UPPER_STD')
    CENTER_ROI_LOWER_STD = SETTINGS.getfloat('CENTER_ROI_LOWER_STD')

    EYE_FOREHEAD_OFFSET = SETTINGS.getfloat('EYE_FOREHEAD_OFFSET')
    HEAD_FOREHEAD_OFFSET = SETTINGS.getfloat('HEAD_FOREHEAD_OFFSET')
    NOSE_FOREHEAD_OFFSET = SETTINGS.getfloat('NOSE_FOREHEAD_OFFSET')
    MOUTH_FOREHEAD_OFFSET = SETTINGS.getfloat('MOUTH_FOREHEAD_OFFSET')
    GLASSES_FOREHEAD_OFFSET = SETTINGS.getfloat('GLASSES_FOREHEAD_OFFSET')
    SUNGLASSES_FOREHEAD_OFFSET = SETTINGS.getfloat('SUNGLASSES_FOREHEAD_OFFSET')

    def __init__(self):
        # Variables for storing bounding box coordinates for each class
        self.eye = []           # id 0 (look for largest 'x' coordinate between two eye BBs)
        self.head = []          # id 2
        self.nose = []          # id 3
        self.mouth = []         # id 4
        self.glasses = []       # id 5
        self.sunglasses = []    # id 6
        self.eyes_pair = []     # no id. Initially empty. Appears only after filter_detections().

    def get_class_list(self):
        return [self.eye, self.head, self.nose, self.mouth, self.glasses, self.sunglasses, self.eyes_pair]

    # ========================== PUBLIC ==========================
    # ============================================================

    def parseImgVars(self, rgb_img, dpt_img, dpt_scale, dpt_intrin):
        self.rgb_img = rgb_img
        self.depth_img = dpt_img
        self.depth_scale = dpt_scale
        self.depth_intrin = dpt_intrin

    def getForeheadHeight(self, findings, print_enable=False):
        self.unpack(findings)

        all_dets = self.get_class_list()
        if self.is_list_empty(all_dets):
            return None

        self.compute_3D_coords()

        all_dets = self.get_class_list()
        if self.is_list_empty(all_dets):
            LOG.debug('None returned after self.compute_3D_coords()')
            DEBUG('None returned after self.compute_3D_coords()')
            return None

        self.filter_detections()

        all_dets = self.get_class_list()
        if self.is_list_empty(all_dets):
            return None

        self.compute_forehead_coords()

        all_dets = self.get_class_list()
        if self.is_list_empty(all_dets):
            return None

        self.remove_background_people()

        all_dets = self.get_class_list()
        if self.is_list_empty(all_dets):
            return None

        if print_enable:
            self.print_detections()

        forehead_xyz, forehead_distance = self.compute_avg_forehead()

        all_dets = self.get_class_list()
        all_dets_flat = [item for sublist in all_dets for item in sublist]
        # det_data = [[item.id , item.label, item.conf, item.point3D.xyz(), item.dist] for item in all_dets_flat]

        # Check forehead xyz coords and ignore detection if it is outside of the defined ranges.
        dist_approved, dist_status = self.directional_distance_filter(forehead_xyz)
        if dist_approved is False:
            LOG.debug(dist_status)
            DEBUG(dist_status)
            return None
        elif forehead_distance == 0:
            return None  # HACK to fix for compute_3D_coords() bug. Seems not be clearing list completely as needed.
        else:
            # Return format:  [[[ fh_x      fh_y   fh_z ], [['class label', conf , [ bb_x,    bb_y,   bb_z ], dist  ]]]]
            # Example return: [[[0.0623, -0.1019, 0.6066], [['human nose', 0.7061, [0.0623, -0.0419, 0.6066], 0.6112]]]]
            return ['forehead', forehead_xyz, forehead_distance, all_dets_flat]

    # ========================= PRIVATE ==========================
    # ============================================================

    # ===================== DETECTION METHODS ====================
    def unpack(self, findings_list):
        if len(findings_list) > 0:
            for det in findings_list:
                det_label = det[-1]  # last in the list.
                det[1] = [int(x) for x in det[1]]  # converting BB pixels from floats to ints.
                if det_label == self.LABEL_EYE:
                    eye_bb = BoundingBox(det[1][0], det[1][1], det[1][2], det[1][3])
                    eye = SingleDetection(det[0], eye_bb, det[2], det[3])
                    self.eye.append(eye)
                elif det_label == self.LABEL_HEAD:
                    head_bb = BoundingBox(det[1][0], det[1][1], det[1][2], det[1][3])
                    head = SingleDetection(det[0], head_bb, det[2], det[3])
                    self.head.append(head)
                elif det_label == self.LABEL_NOSE:
                    nose_bb = BoundingBox(det[1][0], det[1][1], det[1][2], det[1][3])
                    nose = SingleDetection(det[0], nose_bb, det[2], det[3])
                    self.nose.append(nose)
                elif det_label == self.LABEL_GLASSES:
                    glasses_bb = BoundingBox(det[1][0], det[1][1], det[1][2], det[1][3])
                    glasses = SingleDetection(det[0], glasses_bb, det[2], det[3])
                    self.glasses.append(glasses)
                elif det_label == self.LABEL_SUNGLASSES:
                    sunglasses_bb = BoundingBox(det[1][0], det[1][1], det[1][2], det[1][3])
                    sunglasses = SingleDetection(det[0], sunglasses_bb, det[2], det[3])
                    self.sunglasses.append(sunglasses)
                else:
                    pass  # place for future error prevention here.
        else:
            # DEBUG('No classes have been detected in the last frame.')
            pass

    def compute_3D_coords(self):
        ''' Compute average depth for ALL bounding boxes of all detections,
            existing in Detections class until this moment.'''

        all_dets = self.get_class_list()

        for det_class in all_dets:
            if len(det_class) > 0:
                for unique_det in det_class:
                    xyz = self.get_xyz_bounding_box(unique_det, ROI_size=self.FACE_BB_CENTER_ROI)
                    xyz_has_nan = np.isnan(np.sum(xyz))
                    if not xyz_has_nan and xyz[2] != 0:
                        dist = self.get_3D_dist_two_points(xyz)
                        # Rejection/Acceptance
                        unique_det.point3D.x, unique_det.point3D.y, unique_det.point3D.z = xyz
                        unique_det.dist = round(dist, 4)
                    else:
                        label = unique_det.label
                        if label == self.LABEL_EYE:
                            if len(self.eye) == 1:  # HACK FOR SOME REASON, LENGHT IS 2. why 2?
                                self.eye = []
                            else:
                                # self.eye.remove(unique_det)
                                self.eye = []  # HACK might not work correctly if there are more people in scene?
                        if label == self.LABEL_LEFT_EYE or label == self.LABEL_RIGHT_EYE:
                            self.eyes_pair = []
                        elif label == self.LABEL_HEAD:
                            self.head.remove(unique_det)
                        elif label == self.LABEL_NOSE:
                            self.nose.remove(unique_det)
                        elif label == self.LABEL_MOUTH:
                            self.mouth.remove(unique_det)
                        elif label == self.LABEL_GLASSES:
                            self.glasses.remove(unique_det)
                        elif label == self.LABEL_SUNGLASSES:
                            self.sunglasses.remove(unique_det)

    def filter_detections(self):

        all_dets = self.get_class_list()

        for i, det_class in enumerate(all_dets):
            # first filter classes that do not expect more than 1 occurence (everything but eyes and ears)
            if len(det_class) >= 2 and det_class[0].label != self.LABEL_EYE \
               and det_class[0].label != self.LABEL_LEFT_EYE and det_class[0].label != self.LABEL_RIGHT_EYE:
                closest_dist = 10000  # just a big number
                remaining_detection = None
                for unique_det in det_class:
                    if unique_det.dist <= closest_dist:
                        closest_dist = unique_det.dist
                        remaining_detection = unique_det

                label = unique_det.label
                if remaining_detection is not None:
                    if label == self.LABEL_HEAD:
                        self.head = [remaining_detection]
                    elif label == self.LABEL_NOSE:
                        self.nose = [remaining_detection]
                    elif label == self.LABEL_MOUTH:
                        self.mouth = [remaining_detection]
                    elif label == self.LABEL_GLASSES:
                        self.glasses = [remaining_detection]
                    elif label == self.LABEL_SUNGLASSES:
                        self.sunglasses = [remaining_detection]
                else:
                    if label == self.LABEL_HEAD:
                        self.head = []
                    elif label == self.LABEL_NOSE:
                        self.nose = []
                    elif label == self.LABEL_MOUTH:
                        self.mouth = []
                    elif label == self.LABEL_GLASSES:
                        self.glasses = []
                    elif label == self.LABEL_SUNGLASSES:
                        self.sunglasses = []

            elif len(det_class) >= 2 and det_class[0].label == self.LABEL_EYE:  # filtering eyes
                emin = self.DIST_EYE_PAIR_MIN
                emax = self.DIST_EYE_PAIR_MAX

                all_eye_pairs = []
                all_eye_dist = []

                try:
                    for eye_1 in det_class:
                        for eye_2 in det_class:
                            dist = self.get_3D_dist_two_points(eye_1.point3D.xyz(), start_point=eye_2.point3D.xyz())
                            if dist <= emax and dist >= emin and dist not in all_eye_dist:
                                all_eye_dist.append(dist)
                                # larger "x" coordinate in camera axis -> LEFT
                                if eye_1.point3D.x > eye_2.point3D.x:  # labelling eyes as "left" and "right"
                                    eye_1.label = self.LABEL_LEFT_EYE
                                    eye_2.label = self.LABEL_RIGHT_EYE
                                else:
                                    eye_1.label = self.LABEL_RIGHT_EYE
                                    eye_2.label = self.LABEL_LEFT_EYE

                                _pair = [eye_1, eye_2, dist]
                                # print('[filter_detections] eye _pair = [{0}, {1}, {2}]'.format(
                                #    eye_1.point3D.xyz(), eye_2.point3D.xyz(), dist))
                                self.eye.remove(eye_1)
                                self.eye.remove(eye_2)
                                all_eye_pairs.append(_pair)
                except ValueError as err:
                    #  HACK bugfix of "self.eye.remove(eye_1) -> ValueError: list.remove(x): x not in list"
                    LOG.warning(err)
                    DEBUG(err)
                    all_eye_pairs = []

                if all_eye_pairs:
                    closest_eye_pair = []
                    closest_dist_to_camera = 10000  # just a big number to initialize distance comparison
                    for eye_pair in all_eye_pairs:
                        eye_pair_dist_to_camera = (eye_pair[0].dist + eye_pair[1].dist) / 2
                        if eye_pair_dist_to_camera <= closest_dist_to_camera:
                            closest_dist_to_camera = eye_pair_dist_to_camera
                            closest_eye_pair = eye_pair
                    self.eyes_pair.append(closest_eye_pair[0])  # appending one eye to the general detection list
                    self.eyes_pair.append(closest_eye_pair[1])  # appending another eye.
                else:
                    pass

    def compute_forehead_coords(self):

        all_dets = self.get_class_list()

        for idx, cls in enumerate(all_dets):
            # everything, but eyes and ears (there should be only 1 detection of other classes till this point)
            if len(cls) == 1:
                forehead_xyz = self.get_forehead_from_point3D(cls)
                forehead_dist = self.get_3D_dist_two_points(forehead_xyz.xyz())
                cls[0].forehead = forehead_xyz
                cls[0].forehead_dist = forehead_dist
            # if eye_pair is passed (2 detections, left and right eye) -> use same forehead location for both detections
            elif len(cls) == 2 and (cls[0].label == self.LABEL_LEFT_EYE or cls[0].label == self.LABEL_RIGHT_EYE):
                forehead_xyz = self.get_forehead_from_point3D(cls)
                forehead_dist = self.get_3D_dist_two_points(forehead_xyz.xyz())
                cls[0].forehead = forehead_xyz
                cls[0].forehead_dist = forehead_dist
                cls[1].forehead = forehead_xyz
                cls[1].forehead_dist = forehead_dist
            elif len(cls) >= 3:
                if cls[0].label == 'human eye':  # HACK TODO dirty hack. Fix it by optimizing filter_detections
                    self.eye = []
                else:
                    raise ValueError("More than 2 detections per class detection in class '{}'".format(cls[0].label))

    def remove_background_people(self):
        ''' Finds closest detection to camera. Computes distance from this detection to all other detections.
            if other detection is farther than DIST_SAME_HEAD from closest_detection -> remove it from the system.
        '''
        all_dets = self.get_class_list()

        detections_to_remove = []

        all_dets_flat = [item for sublist in all_dets for item in sublist]

        closest_detection = min(all_dets_flat, key=lambda x: x.dist)

        for det in all_dets_flat:
            dist = self.get_3D_dist_two_points(det.point3D.xyz(), start_point=closest_detection.point3D.xyz())
            if dist >= self.DIST_SAME_HEAD:
                detections_to_remove.append(det)

        for det in detections_to_remove:
            label = det.label
            if label == self.LABEL_EYE:
                self.eye.remove(det)
            if label == self.LABEL_LEFT_EYE or label == self.LABEL_RIGHT_EYE:
                self.eyes_pair = []
            elif label == self.LABEL_HEAD:
                self.head.remove(det)
            elif label == self.LABEL_NOSE:
                self.nose.remove(det)
            elif label == self.LABEL_MOUTH:
                self.mouth.remove(det)
            elif label == self.LABEL_GLASSES:
                self.glasses.remove(det)
            elif label == self.LABEL_SUNGLASSES:
                self.sunglasses.remove(det)

    def compute_avg_forehead(self):
        all_dets = self.get_class_list()
        all_dets_flat = [item for sublist in all_dets for item in sublist]
        all_labels = [item.label for item in all_dets_flat]

        if not self.INCLUDE_HEAD:
            for idx, label in enumerate(all_labels):
                # if there are more forehead detections THAN only from 1 human head.
                if label == 'human head' and len(all_labels) > 1:
                    del all_dets_flat[idx]
                    self.head = []

        all_dets = self.get_class_list()
        all_dets_flat = [item for sublist in all_dets for item in sublist]
        all_labels = [item.label for item in all_dets_flat]

        if not self.INCLUDE_NOSE:
            for idx, label in enumerate(all_labels):
                # if there are more forehead detections THAN only from 1 human head.
                if label == 'human nose' and len(all_labels) > 1:
                    del all_dets_flat[idx]
                    self.nose = []

        all_forehead_points = [item.forehead.xyz() for item in all_dets_flat]
        mean_forehead_xyz = [sum(i) / len(i) for i in zip(*all_forehead_points)]
        mean_forehead_xyz = [round(item, 4) for item in mean_forehead_xyz]

        all_forehead_dists = [item.forehead_dist for item in all_dets_flat]
        mean_forehead_dist = round((sum(all_forehead_dists) / len(all_forehead_dists)), 4)

        return mean_forehead_xyz, mean_forehead_dist

    def directional_distance_filter(self, forehead_xyz):
        X, Y, Z = forehead_xyz
        if X > self.IGNORE_DIST_RIGHT:
            status = "Target is too far right. Max: {0}m. Current: {1}m.".format(self.IGNORE_DIST_RIGHT, X)
            return False, status
        elif X < self.IGNORE_DIST_LEFT:
            status = "Target is too far left. Max: {0}m. Current: {1}m.".format(-self.IGNORE_DIST_LEFT, -X)
            return False, status
        elif Z > self.IGNORE_DIST_FAR:
            status = "Target is too far. Max: {0}m. Current: {1}m.".format(self.IGNORE_DIST_FAR, Z)
            return False, status
        elif Z < self.IGNORE_DIST_CLOSE:
            status = "Target is too close. Max: {0}m. Current: {1}m.".format(self.IGNORE_DIST_CLOSE, Z)
            return False, status
        else:
            return True, "OK"

    def get_xyz_bounding_box(self, det, ROI_size=3):
        ''' Compute 3D coordinates of the bounding box to SingleDetection class.
            params:
                det: SingleDetection class object, filled with data.
                square size: grid size around central pixel of the bounding box. For averaging depth.
            returns: [x,y,z] coordinates
        '''
        ROI = self.depth_img[det.bb.y - ROI_size:det.bb.y + ROI_size,
                             det.bb.x - ROI_size:det.bb.x + ROI_size]
        ROI_flatten = ROI.flatten()
        new_ROI = []
        for i in ROI_flatten:
            if i != 0 and i != np.inf and i != np.NaN:
                new_ROI.append(i)

        ROI_flatten = self.trim_list_std(new_ROI, self.CENTER_ROI_LOWER_STD, self.CENTER_ROI_UPPER_STD)

        ROI_mean = np.nanmean(ROI_flatten)
        depth_point = self.get_point_from_pixel([det.bb.x, det.bb.y], ROI_mean)
        return depth_point

    def get_forehead_from_point3D(self, det):
        ''' Find a forehead center point from a single detection
            params:
                det: SingleDetection object filled with data.
        '''

        EYE_PAIR = False  # Special case for eye pair (2 detections instead of 1)
        if len(det) == 2:
            if det[0].label == self.LABEL_LEFT_EYE:
                left_eye = det[0]
                right_eye = det[1]
                EYE_PAIR = True
            elif det[0].label == self.LABEL_RIGHT_EYE:
                left_eye = det[1]
                right_eye = det[0]
                EYE_PAIR = True
            else:
                raise TypeError("Only eye pair is supported for type 'list'.Your list has '{}' label".format(
                                det[0].label))
        elif len(det) == 1:
            det = det[0]  # just removing 'list' around 1 detection
        else:
            raise TypeError("Empty detection is passed. Value of argument 'det' = {0}".format(det))

        forehead_xyz = Point3D(0, 0, 0)
        if not EYE_PAIR:
            x = det.point3D.x
            y = det.point3D.y
            z = det.point3D.z
            label = det.label

            if label == self.LABEL_EYE:
                forehead_xyz = Point3D(x, y - self.EYE_FOREHEAD_OFFSET, z)
            elif label == self.LABEL_HEAD:
                forehead_xyz = Point3D(x, y - self.HEAD_FOREHEAD_OFFSET, z)
            elif label == self.LABEL_NOSE:
                forehead_xyz = Point3D(x, y - self.NOSE_FOREHEAD_OFFSET, z)
            elif label == self.LABEL_MOUTH:
                forehead_xyz = Point3D(x, y - self.MOUTH_FOREHEAD_OFFSET, z)
            elif label == self.LABEL_GLASSES:
                forehead_xyz = Point3D(x, y - self.GLASSES_FOREHEAD_OFFSET, z)
            elif label == self.LABEL_SUNGLASSES:
                forehead_xyz = Point3D(x, y - self.SUNGLASSES_FOREHEAD_OFFSET, z)
            else:
                assert ValueError("Cannot compute forehead, unexpected label '{}'.".format(label))
        else:
            left_x, left_y, left_z = left_eye.point3D.x, left_eye.point3D.y, left_eye.point3D.z
            right_x, right_y, right_z = right_eye.point3D.x, right_eye.point3D.y, right_eye.point3D.z
            fh_x = (left_x + right_x) / 2
            fh_y = ((left_y + right_y) / 2) - (left_x - fh_x)
            fh_z = (left_z + right_z) / 2
            forehead_xyz = Point3D(fh_x, fh_y, fh_z)
        return forehead_xyz

    # ======================== UTILITY ===========================
    def get_point_from_pixel(self, pixel, depth):
        ''' pixel = [x,y]; depth = z'''
        scaled_depth = depth * self.depth_scale
        point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [pixel[0], pixel[1]], scaled_depth)
        point = [round(x, 4) for x in point]
        return point

    def get_pixel_from_point(self, xyz):
        pixel = rs.rs2_project_point_to_pixel(self.depth_intrin, xyz)
        try:
            pixel = [int(x) for x in pixel]
        except ValueError:
            return pixel
        return pixel

    def get_3D_dist_two_points(self, end_point, start_point=[0, 0, 0]):
        ''' The distance between two points in a three dimensional - 3D - coordinate system
            Note: start_point=[0, 0, 0] -> origin of the camera coordinate frame
                  end_point -> some pixel in the scene with obtained x,y,z coordinates.
        '''
        x1, y1, z1 = start_point
        x2, y2, z2 = end_point
        d = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5  # math book equation
        return d

    def print_detections(self):
        ''' Prints all detections in a nicely observable form '''

        all_dets = self.get_class_list()

        det_bool = [True if len(x) != 0 else False for x in all_dets]

        if True in det_bool:
            s0 = '____________________________________________________________________________________________________'
            s1 = '| ID |       Class       | Bounding Box (pixels) |  Conf  |  3D Coordinates (in meters) | Distance |'
            s2 = '|    |                   |  x  |  y  |  w  |  h  |        |    X    |    Y    |    Z    |          |'
            s3 = '----------------------------------------------------------------------------------------------------'
            print(s0)
            print(s1)
            print(s2)
            print(s3)
            for i, cl in enumerate(det_bool):
                if cl is True:
                    for det in all_dets[i]:
                        class_str  = "| %-*s| %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s |" \
                            % (3, det.id, 17, det.label, 3, det.bb.x,
                               3, det.bb.y, 3, det.bb.w, 3, det.bb.h,
                               6, det.conf, 7, det.point3D.x, 7, det.point3D.y, 7, det.point3D.z, 8, det.dist)
                        print(class_str)
                        print(s3)

    def trim_list_std(self, list_input, lower_std, upper_std):
        """ Removes values in a list that are below "lower_std" and above "upper_std".
        Args:
            list_input (list): List of numbers
            lower_std (float): Negative standard deviation coefficient.
            upper_std (float): Positive standard deviation coefficient.
        Returns:
            list: Filtered list of numbers.
        """
        list_input = np.array(list_input)
        mean = np.mean(list_input)
        sd = np.std(list_input)
        final_list = [x for x in list_input if (x > mean - lower_std * sd)]
        final_list = [x for x in final_list if (x < mean + upper_std * sd)]
        return final_list

    def is_list_empty(self, inList):
        if isinstance(inList, list):  # Is a list
            return all(map(self.is_list_empty, inList))
        return False  # Not a list

    # ======================= FOR PLOTTING =======================
    def get_xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = [0, 0, 0, 0]
        y[0] = x[0] - x[2] / 2  # top left x
        y[1] = x[1] - x[3] / 2  # top left y
        y[2] = x[0] + x[2] / 2  # bottom right x
        y[3] = x[1] + x[3] / 2  # bottom right y
        y = [int(x) for x in y]
        return y

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def show_img_forehead_one_by_one(self):
        all_dets = self.get_class_list()

        for cls in all_dets:
            for unique_det in cls:
                color = [random.randint(0, 255) for _ in range(3)]
                xyz = [unique_det.forehead.x, unique_det.forehead.y, unique_det.forehead.z]
                [y, x] = self.get_pixel_from_point(xyz)
                bb_xywh = unique_det.bb.xywh()
                bb_xyxy = self.get_xywh2xyxy(bb_xywh)
                self.plot_one_box(bb_xyxy, self.rgb_img, color=color, label=unique_det.label, line_thickness=1)
                cv2.circle(self.rgb_img, (y, x), 5, color)
                window_name = '{}'.format(unique_det.label)
                cv2.imshow(window_name, self.rgb_img)
                cv2.waitKey()
                cv2.destroyAllWindows()

    def show_img_forehead_averaged(self):
        all_dets = self.get_class_list()

        for cls in all_dets:
            for unique_det in cls:
                try:
                    color = [random.randint(0, 255) for _ in range(3)]
                    xyz = [unique_det.forehead.x, unique_det.forehead.y, unique_det.forehead.z]
                    [y, x] = self.get_pixel_from_point(xyz)
                    bb_xywh = unique_det.bb.xywh()
                    bb_xyxy = self.get_xywh2xyxy(bb_xywh)
                    self.plot_one_box(bb_xyxy, self.rgb_img, label=unique_det.label, line_thickness=1, color=color)
                    cv2.circle(self.rgb_img, (y, x), 5, color)
                except TypeError:
                    pass  # when [y, x] are NaN, comes here
        window_name = '{}'.format('All detections averaged')
        cv2.imshow(window_name, self.rgb_img)
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

    def show_img_forehead_final(self, forehead_xyz):
        forehead_point = Point3D(forehead_xyz[0], forehead_xyz[1], forehead_xyz[2])
        color = (0, 255, 0)
        xyz = [forehead_point.x, forehead_point.y, forehead_point.z]
        [y, x] = self.get_pixel_from_point(xyz)
        cv2.circle(self.rgb_img, (y, x), 5, color)
        window_name = '{}'.format('Final forehead point')
        cv2.imshow(window_name, self.rgb_img)
        cv2.waitKey()
        cv2.destroyAllWindows()


class ForeheadSession(object):
    MIN_VALUES_FINAL_HEIGHT = SETTINGS.getint('MIN_VALUES_FINAL_HEIGHT')
    DIFF_MIN_MAX_HEIGHT = SETTINGS.getfloat('DIFF_MIN_MAX_HEIGHT')
    DIFF_AVG_HEIGHT = SETTINGS.getfloat('DIFF_AVG_HEIGHT')
    CONSECUTIVE_HEIGHT_DIFF = SETTINGS.getfloat('CONSECUTIVE_HEIGHT_DIFF')

    def __init__(self):
        self.avg_height = 0
        self.avg_dist = 0

        self.current_height = None
        self.last_height = None

        self.all_detections = []
        self.all_heights = []
        self.det_counter = 0

        self.final_height = None
        self.previous_avg_height = 100  # just some big number

    def getEstimate(self):
        if len(self.all_heights) >= self.MIN_VALUES_FINAL_HEIGHT:
            last_dets = self.all_heights[-self.MIN_VALUES_FINAL_HEIGHT:]
            avg_height = sum(last_dets) / len(last_dets)
            max_height = max(last_dets)
            min_height = min(last_dets)
            if (max_height >= 0 and min_height >= 0) or (max_height <= 0 and min_height <= 0):
                diff = abs(max_height - min_height)
            else:
                diff = abs(max_height) + abs(min_height)

            if diff < self.DIFF_MIN_MAX_HEIGHT:
                if abs(self.previous_avg_height - avg_height) > self.DIFF_AVG_HEIGHT:
                    self.final_height = avg_height
                    self.previous_avg_height = avg_height
                    return self.final_height, False

        if self.final_height is not None:
            return 0, False
        else:
            if self.last_height is not None:
                if abs(self.current_height - self.last_height) <= self.CONSECUTIVE_HEIGHT_DIFF:
                    two_heights_avg = (self.current_height + self.last_height) / 2
                    return two_heights_avg, True
                else:
                    DEBUG('Returning 0. Threshold works, outside of range!')
                    return 0, True
            else:
                return 0, True

    def save_detection(self, forehead):
        xyz = Point3D(forehead[1][0], forehead[1][1], forehead[1][2])
        sdet = Forehead(self.det_counter, forehead[0], xyz.y, forehead[2], xyz, forehead[3])
        self.all_detections.append(sdet)
        self.all_heights.append(xyz.y)
        self.det_counter += 1

        if self.current_height is not None:
            self.last_height = self.current_height
        self.current_height = xyz.y

    def print_data(self):
        print('===================================================================')
        for item in self.all_detections:
            obj_list = []
            for obj in item.objects:
                obj_list.append(obj.label)
            print("{0} '{1}' xyz: [{2},{3},{4}], dist: {5}, objects: {6}".format(
                  item.id, item.label, item.xyz.x, item.xyz.y, item.xyz.z, item.dist, obj_list))

        print('Average height to forehead: {}'.format(round(self.avg_height, 4)))
        print('Average forehead distance: {}'.format(round(self.avg_dist, 4)))
        print('===================================================================')

    def clear_detections(self):
        self.avg_height = 0
        self.avg_dist = 0
        self.current_height = None
        self.last_height = None
        self.all_detections = []
        self.all_heights = []
        self.det_counter = 0
        self.final_height = None
        self.previous_avg_height = 100  # just some big number
