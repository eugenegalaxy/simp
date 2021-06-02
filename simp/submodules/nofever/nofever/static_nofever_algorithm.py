import time
import datetime
from numpy.core.numeric import full
import requests
from serial.serialutil import SerialException
from json.decoder import JSONDecodeError

from detection.realsense import RealSense
from forehead_scanner import ForeheadFinder, ForeheadSession
from mask_scanner import MaskFinder, MaskSession
from detection.yolov5.face_detect import YOLO
from temperature.TemperatureScanner import TemperatureScanner
from utils import ThreadWithReturn, LOG, DebugPrint, DebugCriticalPrint, CONFIG_NOFEVER_SETTINGS

SETTINGS = CONFIG_NOFEVER_SETTINGS['algorithm']

LABEL_PRINTER_ENABLED = SETTINGS.getboolean('LABEL_PRINTER_ENABLED')
if LABEL_PRINTER_ENABLED is True:
    from ticket_printer import LabelPrinter

SCAN_COUNTING_ENABLED = SETTINGS.getboolean('SCAN_COUNTING_ENABLED')

DEBUG_SAVE_IMAGES = True
if DEBUG_SAVE_IMAGES:
    import os

DEBUG_MODE = True
DEBUG = DebugPrint(DEBUG_MODE)
DEBUG_CRITICAL_MODE = True
DEBUG_CRIT = DebugCriticalPrint(DEBUG_CRITICAL_MODE)


class NoFever(object):
    # Module object holders for HW/SW
    YOLO_FACE = None  # YOLO model for Mask detection
    YOLO_MASK = None  # YOLO model for Facial features detection
    RS = None         # Realsense camera
    RS_scale = None   # Realsense camera optics parameter
    RS_intrin = None  # Realsense camera optics parameter
    FS = None         # Forehead Session
    TS = None         # Temperature Scanner
    LP = None         # Brother Label Printer

    REBOOT_TIMER_MAX = SETTINGS.getint('REBOOT_TIMER_MAX')
    timer_reboot = 100000000000  # a number bigger than time.time() time will ever be.
    first_run_reboot = False
    first_run_fail_cnt = 0
    first_run_flag = True
    initialized = False

    # 'outer_loop' variables
    RESET_MAX = SETTINGS.getint('RESET_MAX')
    reset_cnt = 0
    connection_status = False  # used by check_connection()

    # 'init_modules' and 'check_connection' status variables.
    yolo_enabled = False
    temp_sensor_enabled = False
    realsense_enabled = False
    session_enabled = False

    # 'inner_loop' variables
    hardware_status_healthy = True  # jumps out of inner_loop if 'False' -> turns 'False' if no connection to some HW
    detection_active = False  # when 'False' -> inner_loop runs check_connection/detection. If true -> runs algorithm.
    first_detection = False  # Detection from 'while not detection_active' loop, (1 sec sleep-detect cycle)

    TIMER_INACTIVITY_CYCLE = SETTINGS.getfloat('TIMER_INACTIVITY_CYCLE')

    TIMER_NO_NEW_DETECTIONS = SETTINGS.getfloat('STATIC_TIMER_NO_NEW_DETECTIONS')
    reset_no_new_detections = True

    temp_measured = False
    temp_time_substract = False

    ACTIVATION_ZONE = SETTINGS.getfloat('ACTIVATION_ZONE')

    FOREHEAD_DELTA = SETTINGS.getfloat('FOREHEAD_DELTA')
    FOREHEAD_MIN_DIST = SETTINGS.getfloat('FOREHEAD_MIN_DIST')
    FOREHEAD_AFK_TIMER = SETTINGS.getfloat('FOREHEAD_AFK_TIMER')

    MAX_RESET_SCREEN_NO_FINDINGS = SETTINGS.getint('MAX_RESET_SCREEN_NO_FINDINGS')
    cnt_reset_screen_no_findings = 0

    TIMER_RESULT_SCREEN_DELAY = SETTINGS.getfloat('TIMER_RESULT_SCREEN_DELAY')

    OVER_TMP = SETTINGS.getfloat('OVER_TMP')
    RED_TMP = SETTINGS.getfloat('RED_TMP')
    YELLOW_TMP = SETTINGS.getfloat('YELLOW_TMP')
    GREEN_TMP = SETTINGS.getfloat('GREEN_TMP')

    CNT_TEMP_MAX_ATTEMPTS = SETTINGS.getint('CNT_TEMP_MAX_ATTEMPTS')
    cnt_temp_under_min = 0

    abs_dist_fh = 0  # absolute distance from camera. Used to track when to reset "Result" screen.
    prev_abs_dist_fh = 100000  # same as above

    final_dist_cnt = 0
    final_dist_flag = False
    dist_fh = 0
    dist_fh_prev = 100000
    S_FINAL_DIST_THR = SETTINGS.getfloat('S_FINAL_DIST_THR')
    S_FINAL_DIST_CNT_MAX = SETTINGS.getint('S_FINAL_DIST_CNT_MAX')

    final_height_cnt = 0
    final_height_flag = False
    height_fh = 0
    height_fh_prev = 100000  # just a random big starting number
    FINAL_HEIGHT_UP_THR = SETTINGS.getfloat('FINAL_HEIGHT_UP_THR')
    FINAL_HEIGHT_DOWN_THR = SETTINGS.getfloat('FINAL_HEIGHT_DOWN_THR')
    FINAL_HEIGHT_CNT_MAX = SETTINGS.getint('FINAL_HEIGHT_CNT_MAX')

    final_side_cnt = 0
    final_side_flag = False
    side_fh = 0
    side_fh_prev = 100000  # just a random big starting number
    FINAL_SIDE_LEFT_THR = SETTINGS.getfloat('FINAL_SIDE_LEFT_THR')
    FINAL_SIDE_RIGHT_THR = SETTINGS.getfloat('FINAL_SIDE_RIGHT_THR')
    FINAL_SIDE_CNT_MAX = SETTINGS.getint('FINAL_SIDE_CNT_MAX')

    Y_DIST_CAMERA_TO_TEMP = SETTINGS.getfloat('Y_DIST_CAMERA_TO_TEMP')
    X_DIST_CAMERA_TO_TEMP = SETTINGS.getfloat('X_DIST_CAMERA_TO_TEMP')
    Z_DIST_CAMERA_TO_TEMP = SETTINGS.getfloat('Z_DIST_CAMERA_TO_TEMP')

    # How many seconds to scan. Temp speed 10scans/sec, mask speed (jetson) 1 scan per sec
    TIMER_TEMP_AND_MASK_LOOP = SETTINGS.getfloat('TIMER_TEMP_AND_MASK_LOOP')

    # Keywords to keep track of what APP screen is currently active so that it is not attempted to be activated again.
    app_screen_idle = 'detection_idle'
    app_icon_movement = 'movement'
    app_icon_measurement = 'measurement'
    app_screen_idle_on = False
    app_movement = False

    t_scan_cycle_diff = 0  # init variable

    def __init__(self, host_ip):
        self.HOST = host_ip
        DEBUG('Running static NoFever -> Wall-Mount or Tripod version.')
        self.outer_loop()

    def outer_loop(self):
        """Outer loop of NoFever algorithm. Initialized HW/SW and checks status.
           If status is OK, goes through connection check and if ok, launches main algorithm.
        """
        while True:
            if self.first_run_reboot:
                self.timer_reboot = time.time()
                self.first_run_reboot = False

            timer_reboot_diff = time.time() - self.timer_reboot
            if self.initialized is False and timer_reboot_diff > self.REBOOT_TIMER_MAX:
                self.send_app_request(self.app_screen_idle)
                DEBUG('Quitting the program. Initialization failed for {} seconds.'.format(self.REBOOT_TIMER_MAX))
                LOG.critical('Doing system reboot. Initialization failed for {} seconds.'.format(
                    self.REBOOT_TIMER_MAX))
                raise SystemError('Program terminated. Awaiting PID checker to catch it and start a reboot.')
            self.initialized = self.init_modules()  # 'True' if all good.

            if self.initialized is False and self.first_run_flag is True:
                self.first_run_fail_cnt += 1

            if self.first_run_fail_cnt == 1:
                self.first_run_flag = False
                self.first_run_fail_cnt = 0
                self.first_run_reboot = True

            while self.initialized:
                self.first_run_reboot = True
                if self.reset_cnt >= self.RESET_MAX:
                    self.initialized = False
                    self.reset_cnt = 0
                    break
                self.connection_status = self.check_connection()
                self.reset_cnt += 1
                if self.connection_status:
                    self.hardware_status_healthy = True
                    LOG.warning('Program initialized successfully.\n')
                    DEBUG('Program initialized successfully.\n')
                    self.reset_cnt = 0
                    self.clear_session()
                    self.inner_loop()
                else:
                    LOG.critical('Hardware initialization failed.')
                    DEBUG('Hardware initialization failed.')

    def init_modules(self):
        # NOTE add docstring
        LOG.warning('Initialising NoFever...')
        DEBUG_CRIT('Initialising NoFever...')
        time.sleep(0.3)
        t1_init_modules = time.time()

        if not self.yolo_enabled:
            self.YOLO_FACE = YOLO()
            self.YOLO_FACE.load_yolov5()
            self.YOLO_MASK = YOLO(weights='mask')
            self.YOLO_MASK.load_yolov5()
            LOG.warning('Neural Networks initialised.')
            DEBUG_CRIT('Neural Networks initialised.')
            self.yolo_enabled = True

        if not self.temp_sensor_enabled:
            try:
                self.TS = TemperatureScanner()
                _ = self.TS.get_obj_temp_C_raw()
                self.temp_sensor_enabled = True
                LOG.warning('Temperature sensor initialised.')
                DEBUG_CRIT('Temperature sensor initialised.')
            except (OSError, EnvironmentError, SerialException):
                LOG.critical('No MLX temperature sensor detected.')
                DEBUG_CRIT('No MLX temperature sensor detected.')
                time.sleep(1)
                return False

        if not self.realsense_enabled:
            try:
                self.RS = RealSense()
                self.RS_scale, self.RS_intrin = self.RS.getDepthParams()
                LOG.warning('RealSense camera communication established.')
                DEBUG_CRIT('RealSense camera communication established.')
                self.realsense_enabled = True
            except RuntimeError:
                LOG.critical('No RealSense camera connected')
                DEBUG_CRIT('No RealSense camera connected')
                return False

        if not self.session_enabled:
            self.FS = ForeheadSession()
            self.session_enabled = True

        if LABEL_PRINTER_ENABLED is True:
            self.LP = LabelPrinter()

        t2_init_modules = time.time()
        t_init_modules_diff = t2_init_modules - t1_init_modules
        if t_init_modules_diff > 0.1:  # to avoid printing 0.000 when nothing is executed here
            LOG.warning('Initialisation time: {:1.2f} seconds'.format(t_init_modules_diff))
            DEBUG('Initialisation time: {:1.2f} seconds'.format(t_init_modules_diff))

        return True

    def check_connection(self):
        # NOTE add docstring
        try:
            rs_status = self.RS.ping()
            if rs_status is False:
                LOG.critical('[SYSTEM CHECK] -> RealSense camera. Cannot receive frames.')
                DEBUG_CRIT('[SYSTEM CHECK] -> RealSense camera. Cannot receive frames.')
                self.realsense_enabled = False
                return False
        except RuntimeError:
            LOG.critical('[SYSTEM CHECK] -> RealSense camera. No connection.')
            DEBUG_CRIT('[SYSTEM CHECK] -> RealSense camera. No connection.')
            self.realsense_enabled = False
            return False

        try:
            temp = self.TS.get_obj_temp_C_raw()
            if temp > 1000:  # reading value from unconnected pin gives reading of ~1037.
                LOG.critical('[SYSTEM CHECK] -> Temperature reading spike detected! Tmp = {}'.format(temp))
                DEBUG_CRIT('[SYSTEM CHECK] -> Temperature reading spike detected! Tmp = {}'.format(temp))
                temp_list = []
                for x in range(30):  # Just a safety thing. Look at 30 measurements isntead of 1.
                    temp_list.append(self.TS.get_obj_temp_C_raw())
                    time.sleep(0.1)
                avg_temp = (sum(temp_list) / len(temp_list))
                if avg_temp > 1000:
                    LOG.critical('[SYSTEM CHECK] -> MLX temperature sensor. No connection.')
                    raise SystemError('[SYSTEM CHECK] -> MLX temperature sensor. No connection.')
            elif temp < -10:  # corrupted value (maybe because of tinkering with Emissivity value?) gives -30/-60C.
                LOG.critical('[SYSTEM CHECK] -> Temperature reading corruption detected! Tmp = {}'.format(temp))
                DEBUG_CRIT('[SYSTEM CHECK] -> Temperature reading corruption detected! Tmp = {}'.format(temp))
                temp_list = []
                for x in range(30):  # Just a safety thing. Look at 30 measurements isntead of 1.
                    temp_list.append(self.TS.get_obj_temp_C_raw())
                    time.sleep(0.1)
                avg_temp = (sum(temp_list) / len(temp_list))
                if avg_temp < -10:
                    LOG.critical('[SYSTEM CHECK] -> MLX temperature sensor. Reading error.')
                    self.AS.arduinoReboot()
                    time.sleep(3)
                    temp = self.TS.get_obj_temp_C_raw()
                    for x in range(30):  # Just a safety thing. Look at 30 measurements isntead of 1.
                        temp_list.append(self.TS.get_obj_temp_C_raw())
                        time.sleep(0.1)
                    avg_temp = (sum(temp_list) / len(temp_list))
                    if avg_temp < -10:
                        raise SystemError('[SYSTEM CHECK] -> MLX temperature sensor. Reading Value error.')
        except TypeError:  # happens if connects to wrong port or to arduino that doesnt have code flashed into it.
            LOG.critical('[SYSTEM CHECK] -> Arduino Nano. Connected to wrong port or no program flashed in it.')
            DEBUG_CRIT('[SYSTEM CHECK] -> Arduino Nano. Connected to wrong port or no program flashed in it.')
            self.temp_sensor_enabled = False
            return False
        except (SerialException, JSONDecodeError, OSError):
            LOG.critical('[SYSTEM CHECK] -> Arduino Nano. No connection.')
            DEBUG_CRIT('[SYSTEM CHECK] -> Arduino Nano. No connection.')
            self.temp_sensor_enabled = False
            return False

        # t2 = time.time()
        # DEBUG('Connection check done in {:1.2f} seconds'.format(t2 - t1))

        return True

    def get_temp_and_mask(self, timer_seconds):
        '''
        Computes temperature and mask status simultaneously, using threading and precise timers.

        param:
            timer_seconds [float]: Period to scan temperature and run mask detector
        returns:
            object temperature (temp), mask status estimate (mask_status)
        '''
        all_mask_det_times = []
        avg_mask_det_time = 0
        mask_iterations = 0
        t_loop_end = 0

        MS = MaskSession()

        Temp_Thread = ThreadWithReturn(target=self.TS.get_temperature, args=(timer_seconds,))
        Temp_Thread.start()
        t_loop_start = time.time()
        t_loop_diff = t_loop_end - t_loop_start
        # loop while remaining time from timer_seconds (90% of it) is enough to do one more mask detection
        while timer_seconds - t_loop_diff > avg_mask_det_time:
            t1_mask = time.time()

            img_color, img_depth = self.RS.getframe()
            findings = self.YOLO_MASK.predict(img_color)
            if findings:
                MF = MaskFinder()
                MF.parseImgVars(img_color, img_depth, self.RS_scale, self.RS_intrin)
                mask_status = MF.predict_mask(findings)
                if mask_status:
                    MS.save_detection(mask_status)
            t2_mask = time.time()
            t_mask_diff = t2_mask - t1_mask
            mask_iterations += 1
            all_mask_det_times.append(t_mask_diff)
            avg_mask_det_time = sum(all_mask_det_times) / len(all_mask_det_times)
            t_loop_end = time.time()
            t_loop_diff = t_loop_end - t_loop_start
        temp = Temp_Thread.join()
        mask_status = MS.getEstimate()

        # # DEV NOTE: image saving on measurements. USE FOR DEBUGGING PURPOSES ONLY.
        if DEBUG_SAVE_IMAGES:
            base_path = os.path.dirname(os.path.abspath(__file__))
            img_rel_path = 'log/images/'
            full_img_path = os.path.join(base_path, img_rel_path)
            date_time = time.strftime("%Y.%m.%d_%H:%M:%S", time.localtime())
            file_name = 'temp{0}_{1}_{2}'.format(
                round(temp, 2), mask_status, date_time)
            self.RS.save_image_w_number_gen(img_color, full_img_path, img_name=file_name)

        return round(temp, 2), mask_status

    def clear_session(self):
        """Reset all variables to initial state. For example, final height, final distance, prev_dist, some booleans.
           Used after inactivity timer goes off, temp.measurement ir GREEN or RED.
        """
        self.cnt_temp_under_min = 0
        self.temp_measured = False
        self.first_detection = False
        self.detection_active = False
        self.cnt_reset_screen_no_findings = 0
        self.FS.clear_detections()
        self.reset_no_new_detections = True

        self.scan_cycle = False

        self.abs_dist_fh = 0  # absolute distance from camera. Used to track when to reset "Result" screen.
        self.prev_abs_dist_fh = 100000  # same as above

        self.final_dist_cnt = 0
        self.final_dist_flag = False
        self.dist_fh = 0
        self.dist_fh_prev = 100000

        self.final_height_cnt = 0
        self.final_height_flag = False
        self.height_fh = 0
        self.height_fh_prev = 100000  # just a random big starting number

        self.final_side_cnt = 0
        self.final_side_flag = False
        self.side_fh = 0
        self.side_fh_prev = 100000  # just a random big starting number
        DEBUG('Clearing session.')

    def clear_only_temp_vars(self):
        """ Clear only the variables used to initiate temperature mesaurement:
                distance, height, side  of the forehead from the sensor position.
        """
        self.final_dist_cnt = 0
        self.final_dist_flag = False
        self.dist_fh = 0
        self.dist_fh_prev = 100000

        self.final_height_cnt = 0
        self.final_height_flag = False
        self.height_fh = 0
        self.height_fh_prev = 100000  # just a random big starting number

        self.final_side_cnt = 0
        self.final_side_flag = False
        self.side_fh = 0
        self.side_fh_prev = 100000  # just a random big starting number

    def get_temperature_decision(self, tmp):
        """Given a temperature value in Celsius, decides which temperature category to return.
           Designed to be used with human skin temperature ranges (not inner!).

        Args:
            tmp (float): temperature value in Celsius.

        Returns:
            emit_str (string): string to socketio.emit() to NoFever App.
            color_status (string): Color code of the temperature category (red, green, yellow)
        """
        if tmp <= self.GREEN_TMP:
            if self.cnt_temp_under_min < self.CNT_TEMP_MAX_ATTEMPTS:
                self.temp_measured = False
                emit_str = 'temperature_measured_wrong'
                color_status = 'Blue'
                self.cnt_temp_under_min += 1
                self.FS.clear_detections()
                self.dist_fh_prev = 0
                self.final_dist_flag = False
                self.final_dist_cnt = 0
                self.final_height_cnt = 0
            else:
                self.temp_measured = True
                emit_str = 'temperature_measured_under_twice'
                color_status = 'Blue Twice'
                self.cnt_temp_under_min = 0
        elif tmp <= self.YELLOW_TMP and tmp > self.GREEN_TMP:
            self.temp_measured = True
            emit_str = 'temperature_measured_good'
            color_status = 'Green'
            self.cnt_temp_under_min = 0
        elif tmp < self.RED_TMP and tmp > self.YELLOW_TMP:
            self.temp_measured = True
            emit_str = 'temperature_measured_slightly_high'
            color_status = 'Yellow'
            self.cnt_temp_under_min = 0
        elif tmp >= self.RED_TMP and tmp < self.OVER_TMP:
            self.temp_measured = True
            emit_str = 'temperature_measured_bad'
            color_status = 'Red'
            self.cnt_temp_under_min = 0
        elif tmp >= self.OVER_TMP:
            self.temp_measured = True
            emit_str = 'temperature_measured_over_twice'
            color_status = 'Super Hot'
            self.cnt_temp_under_min = 0
        return emit_str, color_status

    def send_app_request(self, emit_signal):
        try:
            # DEBUG("[TABLET] -> {} ".format(emit_signal))
            requests.get(self.HOST + emit_signal)
        except requests.exceptions.ConnectionError as err:
            err_string = 'Error: {}.'.format(err)
            err_string2 = 'No connection to App Server. Terminating the program.'
            LOG.critical(err_string)
            LOG.critical(err_string2)
            raise SystemError(err_string2)

    def emit_app_screen(self, screen_name, second_emit=None):
        if screen_name == self.app_screen_idle and self.app_screen_idle_on is False:
            self.send_app_request(screen_name)
            self.app_movement = False
            self.app_screen_idle_on = True
        elif 'movement' in screen_name and self.app_movement is False:
            self.send_app_request(screen_name)
            self.app_movement = True
            self.app_screen_idle_on = False
        elif 'measurement' in screen_name:
            self.send_app_request(screen_name)
            self.app_movement = False
            self.app_screen_idle_on = False
        elif 'temperature' in screen_name:
            if second_emit is not None:
                self.send_app_request(second_emit)
            self.send_app_request(screen_name)
            self.app_movement = False
            self.app_screen_idle_on = False

    def count_scan(self, temp):
        base_path = os.path.dirname(__file__)
        dir_name = os.uname()[1]
        rel_path = 'log/{}'.format(dir_name)
        dir_path = os.path.join(base_path, rel_path)
        file_name = '{}_scans.txt'.format(dir_name)
        full_path = os.path.join(base_path, dir_path, file_name)

        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

        if not os.path.isfile(full_path):
            DEBUG('File does not exist at {}'.format(full_path))
            DEBUG('Creating new file...')
            default_txt = {
                'nofever_id': os.uname()[1],  # hostname
                'last_updated': str(datetime.datetime.now()),
                'total_scans': 0,
                'blue_scans': 0,
                'green_scans': 0,
                'yellow_scans': 0,
                'red_scans': 0,
                'too_hot_scans': 0
            }
            with open(full_path, 'w') as f:
                for key, value in default_txt.items():
                    f.write('%s:%s\n' % (key, value))

        scan_info = {}
        try:
            with open(full_path, 'r') as f:
                for line in f:
                    (key, val) = line.strip().split(':', 1)
                    scan_info[key] = val

            for key, value in scan_info.items():
                if value.isdecimal() is True:
                    scan_info[key] = int(value)
        except ValueError as e:
            DEBUG(e)
            LOG.warning(e)
            LOG.warning('Content of scan count txt is wrong. Here is what was removed:')
            with open(full_path, 'r') as f:
                for line in f:
                    LOG.warning(line)
            open(full_path, 'w').close()
            DEBUG('Creating new file...')
            default_txt = {
                'nofever_id': os.uname()[1],  # hostname
                'last_updated': str(datetime.datetime.now()),
                'total_scans': 0,
                'blue_scans': 0,
                'green_scans': 0,
                'yellow_scans': 0,
                'red_scans': 0,
                'too_hot_scans': 0
            }
            with open(full_path, 'w') as f:
                for key, value in default_txt.items():
                    f.write('%s:%s\n' % (key, value))
            with open(full_path, 'r') as f:
                for line in f:
                    (key, val) = line.strip().split(':', 1)
                    scan_info[key] = val
            for key, value in scan_info.items():
                if value.isdecimal() is True:
                    scan_info[key] = int(value)

        scan_info['nofever_id'] = os.uname()[1]  # get hostname (NoFever unique ID)
        scan_info['last_updated'] = str(datetime.datetime.now())  # get current timedate
        scan_info['total_scans'] += 1  # Iterate total # of scans by 1
        if temp == 'Blue' or temp == 'Blue Twice':
            scan_info['blue_scans'] += 1
        elif temp == 'Green':
            scan_info['green_scans'] += 1
        elif temp == 'Yellow':
            scan_info['yellow_scans'] += 1
        elif temp == 'Red':
            scan_info['red_scans'] += 1
        elif temp == 'Super Hot':
            scan_info['too_hot_scans'] += 1
        else:
            DEBUG('Error in count_scans() -> Wrong "temp" string: {}'.format(temp))

        with open(full_path, 'w') as f:
            for key, value in scan_info.items():
                f.write('%s:%s\n' % (key, value))

    def inner_loop(self):
        while self.hardware_status_healthy:
            self.detection_active = False
            self.emit_app_screen('detection_idle')
            self.dist_fh_prev = 0

            while not self.detection_active:  # Standby loop
                t1_inactivity = time.time()
                hw_status_ok = self.check_connection()
                if not hw_status_ok:
                    self.hardware_status_healthy = False
                    break
                img_color, img_depth = self.RS.getframe()
                findings = self.YOLO_FACE.predict(img_color)
                forehead = []
                if findings is not None:
                    FF = ForeheadFinder()
                    FF.parseImgVars(img_color, img_depth, self.RS_scale, self.RS_intrin)
                    forehead = FF.getForeheadHeight(findings)
                t2_inactivity = time.time()
                time_diff_inactivity = t2_inactivity - t1_inactivity
                if forehead:
                    self.detection_active = True
                    self.first_detection = True
                else:
                    if time_diff_inactivity < self.TIMER_INACTIVITY_CYCLE:
                        DEBUG('Sleeping 1 sec')
                        time.sleep(self.TIMER_INACTIVITY_CYCLE - time_diff_inactivity)

            while self.detection_active:
                if self.reset_no_new_detections:
                    t1_det_active = time.time()
                    self.reset_no_new_detections = False

                t_detection_cycle_start = time.time()
                if self.first_detection is True:
                    self.first_detection = False  # don't take new detection, but use one from 'Standby loop'
                else:
                    img_color, img_depth = self.RS.getframe()
                    findings = self.YOLO_FACE.predict(img_color)
                t_detection_cycle_diff = time.time() - t_detection_cycle_start

                if findings and self.temp_measured is False:
                    t1_scan_cycle = time.time()

                    FF = ForeheadFinder()
                    FF.parseImgVars(img_color, img_depth, self.RS_scale, self.RS_intrin)
                    forehead = FF.getForeheadHeight(findings)  # ['forehead', forehead_xyz, forehead_distance, det_data]
                    # FF.show_img_forehead_averaged()
                    if forehead:
                        self.reset_no_new_detections = True
                        self.side_fh, self.height_fh, self.dist_fh = forehead[1]
                        self.abs_dist_fh = forehead[2]
                        # DEBUG('DIST {0:1.4f}  HEIGHT {1:1.4}  SIDE {2:1.4}'.format(
                        #     self.dist_fh, self.height_fh, self.side_fh))

                        # Transformation of x,y,z distances from camera to temp center.
                        # Forehead must be aligned with temperature horn, not camera!
                        DIST_THR = self.S_FINAL_DIST_THR + self.Z_DIST_CAMERA_TO_TEMP
                        HEIGHT_UP_THR = self.FINAL_HEIGHT_UP_THR + self.Y_DIST_CAMERA_TO_TEMP
                        HEIGHT_DOWN_THR = self.FINAL_HEIGHT_DOWN_THR + self.Y_DIST_CAMERA_TO_TEMP
                        SIDE_LEFT_THR = self.FINAL_SIDE_LEFT_THR + self.X_DIST_CAMERA_TO_TEMP
                        SIDE_RIGHT_THR = self.FINAL_SIDE_RIGHT_THR + self.X_DIST_CAMERA_TO_TEMP

                        #  TEST PRINTS TO BE REMOVED ==============
                        if self.dist_fh <= DIST_THR:
                            dist_str = 'GOOD'
                        else:
                            missing_distance = self.dist_fh - DIST_THR
                            dist_str = '{:1.4f}'.format(missing_distance)

                        if self.height_fh >= HEIGHT_UP_THR and self.height_fh <= HEIGHT_DOWN_THR:
                            height_str = 'GOOD'
                        elif self.height_fh <= HEIGHT_UP_THR:
                            missing_height = HEIGHT_UP_THR - self.dist_fh
                            height_str = '{:1.4f}'.format(missing_height)
                        elif self.height_fh >= HEIGHT_DOWN_THR:
                            missing_height = HEIGHT_DOWN_THR - self.dist_fh
                            height_str = '{:1.4f}'.format(missing_height)

                        if (self.side_fh <= SIDE_RIGHT_THR) and (self.side_fh >= SIDE_LEFT_THR):
                            side_str = 'GOOD'
                        elif self.side_fh >= SIDE_RIGHT_THR:
                            missing_side = self.side_fh - SIDE_RIGHT_THR
                            side_str = '{:1.4f}'.format(missing_side)
                        elif self.side_fh <= SIDE_LEFT_THR:
                            missing_side = self.side_fh - SIDE_LEFT_THR
                            side_str = '{:1.4f}'.format(missing_side)

                        # DEBUG('SIDE: {0}  | HEIGHT: {1}  |  DIST: {2}'.format(side_str, height_str, dist_str))
                        # ===========================================

                        if self.dist_fh < self.ACTIVATION_ZONE:
                            self.emit_app_screen('movement_begin')

                        if self.dist_fh <= DIST_THR and self.dist_fh_prev <= DIST_THR:
                            self.final_dist_cnt += 1
                            if self.final_dist_cnt == self.S_FINAL_DIST_CNT_MAX:
                                self.final_dist_flag = True
                        else:
                            self.final_dist_cnt = 0  # This will reset counter if final_dists are not consecutive
                            self.final_dist_flag = False

                        if (self.height_fh >= HEIGHT_UP_THR) and \
                          (self.height_fh <= HEIGHT_DOWN_THR) and \
                          (self.height_fh_prev >= HEIGHT_UP_THR) and \
                          (self.height_fh_prev <= HEIGHT_DOWN_THR):

                            self.final_height_cnt += 1
                            if self.final_height_cnt == self.FINAL_HEIGHT_CNT_MAX:
                                self.final_height_flag = True
                        else:
                            self.final_height_cnt = 0  # This will reset counter if final_dists are not consecutive
                            self.final_height_flag = False

                        if (self.side_fh <= SIDE_RIGHT_THR) and \
                          (self.side_fh >= SIDE_LEFT_THR) and \
                          (self.side_fh_prev <= SIDE_RIGHT_THR) and \
                          (self.side_fh_prev >= SIDE_LEFT_THR):

                            self.final_side_cnt += 1
                            if self.final_side_cnt == self.FINAL_SIDE_CNT_MAX:
                                self.final_side_flag = True
                        else:
                            self.final_side_cnt = 0  # This will reset counter if final_dists are not consecutive
                            self.final_side_flag = False

                        if self.final_dist_flag is True and self.final_height_flag is True and \
                          self.final_side_flag is True:
                            self.clear_only_temp_vars()
                            self.emit_app_screen('measurement_begin')
                            temp, mask = self.get_temp_and_mask(self.TIMER_TEMP_AND_MASK_LOOP)
                            self.temp_time_substract = True
                            emit_screen, temp_color = self.get_temperature_decision(temp)

                            if mask == 'no_detections':
                                # mask = 'mask_off'
                                self.emit_app_screen(emit_screen)
                            else:
                                self.emit_app_screen(emit_screen, second_emit=mask)

                            if LABEL_PRINTER_ENABLED is True:
                                self.LP.ticket_handling(emit_screen, mask)

                            if SCAN_COUNTING_ENABLED is True:
                                self.count_scan(temp_color)

                            LOG.warning('Temperature: {0}. Status: {1}. Mask: {2}.'.format(
                                temp + 3, temp_color, mask))
                            DEBUG_CRIT('Temperature: {0}. Status: {1}. Mask: {2}.'.format(
                                temp + 3, temp_color, mask))
                            if temp_color == 'Red':
                                time.sleep(self.TIMER_RESULT_SCREEN_DELAY + 3)
                            else:
                                time.sleep(self.TIMER_RESULT_SCREEN_DELAY)
                            t1_temp = time.time()
                    t_scan_cycle_diff = time.time() - t1_scan_cycle
                    self.scan_cycle = True

                elif findings and self.temp_measured is True:
                    t2_temp = time.time()
                    FF = ForeheadFinder()
                    FF.parseImgVars(img_color, img_depth, self.RS_scale, self.RS_intrin)
                    forehead = FF.getForeheadHeight(findings)
                    if forehead:
                        self.reset_no_new_detections = True  # NOTE This has to stay here
                        self.abs_dist_fh = forehead[2]
                        if (self.abs_dist_fh - self.prev_abs_dist_fh) <= self.FOREHEAD_DELTA and \
                           (t2_temp - t1_temp) < self.FOREHEAD_AFK_TIMER and \
                           self.abs_dist_fh <= self.FOREHEAD_MIN_DIST:
                            pass
                            # DEBUG('Time left to reset screen: {}'.format(
                            #     round((self.FOREHEAD_AFK_TIMER - (t2_temp - t1_temp)), 2)))
                        else:
                            self.clear_session()

                elif not findings and self.temp_measured is True:
                    self.cnt_reset_screen_no_findings += 1
                    if self.cnt_reset_screen_no_findings >= self.MAX_RESET_SCREEN_NO_FINDINGS:
                        self.clear_session()

                self.prev_abs_dist_fh = self.abs_dist_fh
                self.height_fh_prev = self.height_fh
                self.dist_fh_prev = self.dist_fh
                self.side_fh_prev = self.side_fh

                t2_det_active_diff = time.time() - t1_det_active
                if self.scan_cycle is True:
                    det_active_diff = t2_det_active_diff - t_detection_cycle_diff - t_scan_cycle_diff
                    self.scan_cycle = False
                else:
                    det_active_diff = t2_det_active_diff - t_detection_cycle_diff

                #  Case that goes back to welcome screen if no new detections happen in TIMER_NO_NEW_DETECTIONS seconds.
                if det_active_diff > self.TIMER_NO_NEW_DETECTIONS:
                    self.clear_session()
                    DEBUG('No new detections in last {} seconds. Enabling Welcome screen.'.format(
                        self.TIMER_NO_NEW_DETECTIONS))

                status = self.check_connection()
                if not status:
                    self.emit_app_screen('detection_idle')
                    self.detection_active = False
                    self.hardware_status_healthy = False
                    self.clear_session()
