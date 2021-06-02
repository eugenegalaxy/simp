import os.path
import re
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from utils import CONFIG_NOFEVER_SETTINGS, DebugPrint

DEBUG_MODE = True
DEBUG = DebugPrint(DEBUG_MODE)

SETTINGS = CONFIG_NOFEVER_SETTINGS['realsense']


class RealSense(object):

    RS_IMG_WIDTH = SETTINGS.getint('RS_IMG_WIDTH')
    RS_IMG_HEIGHT = SETTINGS.getint('RS_IMG_HEIGHT')
    RS_FRAMERATE = SETTINGS.getint('RS_FRAMERATE')
    RS_DIST_FILTER_MIN = SETTINGS.getfloat('RS_DIST_FILTER_MIN')
    RS_DIST_FILTER_MAX = SETTINGS.getfloat('RS_DIST_FILTER_MAX')
    RS_GET_FRAME_TIMEOUT = SETTINGS.getint('RS_GET_FRAME_TIMEOUT')

    def __init__(self):
        # Initiliase camera, get pipe
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.RS_IMG_WIDTH, self.RS_IMG_HEIGHT,
                                  rs.format.bgr8, self.RS_FRAMERATE)
        self.config.enable_stream(rs.stream.depth, self.RS_IMG_WIDTH, self.RS_IMG_HEIGHT,
                                  rs.format.z16, self.RS_FRAMERATE)
        self.profile = self.pipeline.start(self.config)

        # Getting the depth sensor's depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Create an align object
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # Get one frame to extract depth intrisics
        frames = self.pipeline.wait_for_frames(timeout_ms=self.RS_GET_FRAME_TIMEOUT)
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        time.sleep(4)

    def getframe(self, save_path=None):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=self.RS_GET_FRAME_TIMEOUT)
            self.pipeline.poll_for_frames()
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not depth_frame or not color_frame:
                pass  # place for error-handling code.

            thr_filter = rs.threshold_filter(min_dist=self.RS_DIST_FILTER_MIN, max_dist=self.RS_DIST_FILTER_MAX)
            depth_frame = thr_filter.process(depth_frame)

            img_color = np.asanyarray(color_frame.get_data())
            img_depth = np.asanyarray(depth_frame.get_data())

            if save_path is not None:
                ext = '.jpg'
                name = 'image_'
                next_nr = generate_number_imgsave(save_path)
                full_name = save_path + name + next_nr + ext
                cv2.imwrite(full_name, img_color)

        except RuntimeError:
            DEBUG('RuntimeError: No connection to Real Sense Camera.')
            return None, None

        return img_color, img_depth

    def save_image_w_number_gen(self, img, save_path, img_name=None, ext='.jpg', number_gen=False):

        MAX_IMAGE_FOLDER_STORAGE_SIZE = SETTINGS.getfloat('MAX_IMAGE_FOLDER_STORAGE_SIZE')

        if img_name is not None:
            name = img_name
        else:
            name = 'image'
        if number_gen is True:
            underscore = "_"
            next_nr = generate_number_imgsave(save_path)
            full_name = save_path + name + underscore + next_nr + ext
        else:
            full_name = save_path + name + ext
        dir_size_guard(save_path, MAX_IMAGE_FOLDER_STORAGE_SIZE)
        cv2.imwrite(full_name, img)

    def ping(self):
        try:
            status = self.pipeline.try_wait_for_frames(timeout_ms=self.RS_GET_FRAME_TIMEOUT)
        except RuntimeError:
            return False

        return status[0]  # True of False, depending if frame is available.

    def getDepthParams(self):
        return self.depth_scale, self.depth_intrin

    def __del__(self):
        if self.pipeline in locals():
            self.pipeline.stop()


def generate_number_imgsave(path):
    """Consecutive number generator for image names. Scans provided directory and determines largest number.
    If no images in the provided directory, will start from "0000"
    Args:
        path (string): Path to a directory
    Returns:
        string: Four digit number from 0000 to 9999
    """
    img_list = sorted(os.listdir(path))
    img_number = 0
    for item in img_list:
        ext = os.path.splitext(item)[1]
        if ext == '.jpg' or ext == '.jpeg':
            img_number += 1
    if img_number == 0:
        return '0000'  # if folder is empty -> give next image '_0000' in name
    else:
        for word in list(img_list):  # iterating on a copy since removing will mess things up
            path_no_ext = os.path.splitext(str(word))[0]  # name without file extension like .jpg
            word_list = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?\s]', str(path_no_ext))
            if len(word_list) != 2 or word_list[0] != 'image':
                img_list.remove(word)

        img_last = img_list[-1]  # -1 -> last item in list
        path_no_ext = os.path.splitext(str(img_last))[0]
        word_list = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?\s]', str(path_no_ext))
        next_number = int(word_list[1]) + 1
        next_number_str = str(next_number).zfill(4)
        return next_number_str


def dir_size_guard(path, limit_in_megabytes):
    """ Check directory size against provided limit and deletes oldest file.
    Args:
        path (string): Path to a directory
        limit_in_megabytes (int): Maximum number of megabytes for the provided "path" directory.
    """
    bytes_in_one_megabyte = 1048576
    while (dir_get_size(path) / bytes_in_one_megabyte) > limit_in_megabytes:
        oldest_file = sorted([os.path.join(path, f) for f in os.listdir(path)], key=os.path.getctime)[0]
        DEBUG('Directory size reached limit of {0} megabytes. Deleting file "{1}".'.format(
            limit_in_megabytes, oldest_file))
        os.remove(oldest_file)


def dir_get_size(path, print_enable=False):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    if print_enable is True:
        size_str = str(round((dir_get_size(path) / 1048576), 4)) + 'MB'
        DEBUG('Size: {0}. Folder: {1}'.format(size_str, path))
    return total_size
