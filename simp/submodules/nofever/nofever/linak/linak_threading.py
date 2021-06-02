import time
import usb1
from linak.linak import LinakController
from utils import ThreadWithReturn, LOG, CONFIG_NOFEVER_SETTINGS, DebugPrint

DEBUG_MODE = False
DEBUG = DebugPrint(DEBUG_MODE)

SETTINGS = CONFIG_NOFEVER_SETTINGS['linak']


class LinakControllerThreaded(object):

    THREAD_linak_stop = False  # This is a bool 'action' to stop linak motion.
    LINAK_TIME_BETWEEN_MOTIONS = 0.4  # Linak needs a break of 300-400millis if it changes direction.
    _CURRENT_TARGET_HEIGHT = 0  # Stores last target height
    WAIT_MOTION_TIMEOUT = SETTINGS.getfloat('WAIT_MOTION_TIMEOUT')

    def __init__(self):
        self.LI = LinakController()

    def move(self, height, wait_motion=False):
        try:
            self._move(height, wait_motion=wait_motion)
            return True
        except Exception as err:
            DEBUG('Error in LinakControllerThreaded: {}'.format(err))
            LOG.critical(err)
            return False

    def _move(self, height, wait_motion=False):

        if height < self.LI.MIN_DISTANCE_rel:
            height = self.LI.MIN_DISTANCE_rel
            err_str = "Target height must be higher than {0} and lower than {1}. Attempted height: {2}".format(
                self.LI.MIN_DISTANCE_rel, self.LI.MAX_DISTANCE_rel, height)
            DEBUG(err_str)

        elif height > self.LI.MAX_DISTANCE_rel:
            height = self.LI.MAX_DISTANCE_rel
            err_str = "Target height must be higher than {0} and lower than {1}. Attempted height: {2}".format(
                self.LI.MIN_DISTANCE_rel, self.LI.MAX_DISTANCE_rel, height)
            DEBUG(err_str)

        if self.LI.LINAK_MOVING:
            move_case = self.compute_motion_decision(self._CURRENT_TARGET_HEIGHT, height, self.LI.LINAK_HEIGHT)
            if move_case == 1:  # -> wait old_motion, do new_motion (pos_delta)
                DEBUG('move_case == 1')
                self._CURRENT_TARGET_HEIGHT = height
                self.wait_motion_finish()
                self.stop()
                time.sleep(self.LINAK_TIME_BETWEEN_MOTIONS)
            elif move_case == 2:  # -> wait old_motion, do new_motion (neg_delta)
                DEBUG('move_case == 2')
                self._CURRENT_TARGET_HEIGHT = height
                self.wait_motion_finish()
                self.stop()
                time.sleep(self.LINAK_TIME_BETWEEN_MOTIONS)
            elif move_case == 3:
                DEBUG('move_case == 3')  # -> stop old_motion, do new_motion (neg_delta)
                self._CURRENT_TARGET_HEIGHT = height
                self.last_height = self.stop()
                time.sleep(self.LINAK_TIME_BETWEEN_MOTIONS)
            elif move_case == 4:  # -> stop old_motion, do new_motion (pos_delta)
                DEBUG('move_case == 4')
                self._CURRENT_TARGET_HEIGHT = height
                self.last_height = self.stop()
                time.sleep(self.LINAK_TIME_BETWEEN_MOTIONS)
            elif move_case == 5:  # -> wait old_motion until x >= new_motion, then stop (neg_delta)
                DEBUG('move_case == 5')
                self._CURRENT_TARGET_HEIGHT = height
                # self.wait_motion_finish() # TODO TEST
                self.last_height = self.stop()
                time.sleep(self.LINAK_TIME_BETWEEN_MOTIONS)
            elif move_case == 6:  # -> wait old_motion until x<= new_motion, then stop (pos_delta)
                DEBUG('move_case == 6')
                self._CURRENT_TARGET_HEIGHT = height
                # self.wait_motion_finish() # TODO TEST
                self.last_height = self.stop()
                time.sleep(self.LINAK_TIME_BETWEEN_MOTIONS)
            elif move_case == 7:  # -> wait old_motion
                DEBUG('move_case == 7')
                self._CURRENT_TARGET_HEIGHT = height
                self.wait_motion_finish()
                self._CURRENT_TARGET_HEIGHT = height
                return 0
            elif move_case == 8:  # -> do nothing
                DEBUG('move_case == 8')
                self._CURRENT_TARGET_HEIGHT = height
                self.wait_motion_finish()
                self._CURRENT_TARGET_HEIGHT = height
                return 0

        self._CURRENT_TARGET_HEIGHT = height
        self.t = ThreadWithReturn(target=self.LI.move_threaded, args=(height, lambda: self.THREAD_linak_stop))
        self.t.start()

        if wait_motion is True:
            self.wait_motion_finish()
            self.stop()
            time.sleep(self.LINAK_TIME_BETWEEN_MOTIONS)

    def active_threads_count(self):
        number_of_threads = self.t.active_count()
        DEBUG('{} threads active.'.format(number_of_threads))
        return number_of_threads

    def move_to_abs_height(self, height, wait_motion=False):
        self._CURRENT_TARGET_HEIGHT = height

        if height < self.LI.MIN_DISTANCE_abs or height > self.LI.MAX_DISTANCE_abs:
            err_str = "Target height must be higher than {0} and lower than {1}. Attempted height: {2}".format(
                self.LI.MIN_DISTANCE_abs, self.LI.MAX_DISTANCE_abs, height)
            DEBUG(err_str)
            return 0

        current_abs_height = self.get_height_absolute()
        abs_diff = height - current_abs_height
        current_height = self.get_height_relative()
        rel_diff = current_height + abs_diff
        status = self.move(rel_diff, wait_motion=wait_motion)
        return status

    def stop(self):
        self.THREAD_linak_stop = True
        height = self.t.join()
        self.THREAD_linak_stop = False
        return height

    def get_height_relative(self):
        return self.LI.LINAK_HEIGHT

    def get_height_absolute(self):
        return self.LI.LINAK_HEIGHT_ABSOLUTE

    def get_min_dist_rel(self):
        return self.LI.MIN_DISTANCE_rel

    def get_max_dist_rel(self):
        return self.LI.MAX_DISTANCE_rel

    def linak_active(self):
        return self.LI.LINAK_MOVING

    def moveDown(self):
        return self.LI._moveDown()

    def wait_motion_finish(self):
        t_start = time.time()
        while self.linak_active():
            t_end = time.time()
            if t_end - t_start >= self.WAIT_MOTION_TIMEOUT:
                raise usb1.USBErrorNoDevice('Waiting motion to finish has timed out. Timeout = {}'.format(
                    self.WAIT_MOTION_TIMEOUT))
            time.sleep(0.01)

    def compute_motion_decision(self, old_target, new_target, curr_height):
        if new_target > old_target and curr_height < old_target and curr_height < new_target:
            return 1  # CASE 1: new > old; x < old,new -> wait old_motion, do new_motion (pos_delta)
        elif new_target < old_target and curr_height > old_target and curr_height > new_target:
            return 2  # CASE 2: new < old; x > old,new -> wait old_motion, do new_motion (neg_delta)
        elif new_target < old_target and curr_height < old_target and curr_height > new_target:
            return 3  # CASE 3: new < old; old > x > new -> stop old_motion, do new_motion (neg_delta)
        elif new_target > old_target and curr_height >= old_target and curr_height < new_target:
            return 4  # CASE 4: new > old; old < x < new -> stop old_motion, do new_motion (pos_delta)
        elif new_target < old_target and curr_height < old_target and curr_height < new_target:
            return 5  # CASE 5: new < old; x < old, new -> wait old_motion until x >= new_motion, then stop (neg_delta)
        elif new_target > old_target and curr_height > old_target and curr_height > new_target:
            return 6  # CASE 6: new > old; x > old, new -> wait old_motion until x<= new_motion, then stop (pos_delta)
        elif new_target == old_target:
            return 7  # CASE 7: new = old; x = whatever -> wait old_motion
        elif curr_height == new_target or curr_height == old_target:
            return 8  # CASE 8: current_height = new_target -> do nothing
        else:
            raise ValueError("[LINAK] Undefined situation calling compute_motion_decision({0}, {1}, {2})"
                             .format(old_target, new_target, curr_height))

    def convert_realsense_to_linak_height(self, height):
        ''' Converts Intel Realsense D435i "y" 3D coord unit to linak unit and adds translation distance.
            Realsense unit: distance in meters, y-axis points down.
            Linak unit: distance in 1/10th milimeter, y-axis points up.
        '''
        height_converted = int(-(height * 10000))
        if height_converted == 0:
            return height_converted
        else:
            height_translated = height_converted - self.LI.HORN_HEIGHT
            return height_translated
