from threading import Thread, active_count
import os
import time
import logging
from logging.handlers import RotatingFileHandler
import configparser


class Internet():
    def __init__(self, device_name, log_handle):
        self.device_name = device_name

    def connect(self):
        print('Connecting to internet via "{}" device'.format(self.device_name))
        LOG.warning('Connecting to internet via "{}" device'.format(self.device_name))
        os.system('nmcli dev connect {}'.format(self.device_name))
        time.sleep(1)

    def disconnect(self):
        print('Disconnecting from internet via "{}" device'.format(self.device_name))
        LOG.warning('Disconnecting from internet via "{}" device'.format(self.device_name))
        os.system('nmcli dev disconnect {}'.format(self.device_name))


class DebugPrint(object):
    def __init__(self, enabled):
        self.enabled = enabled

    def __call__(self, msg):
        if self.enabled:
            t = time.localtime()
            current_time = time.strftime("[%H:%M:%S] ", t)
            print(current_time + str(msg))


class DebugCriticalPrint(object):
    def __init__(self, enabled):
        self.enabled = enabled

    def __call__(self, msg):
        if self.enabled:
            t = time.localtime()
            current_time = time.strftime("[%H:%M:%S] ", t)
            print(current_time + str(msg))


class ThreadWithReturn(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        self.exc = None
        try:
            if self._target is not None:
                self._return = self._target(*self._args, **self._kwargs)
        except Exception as e:
            self.exc = e

    def join(self, *args):
        # Thread.join(self, *args)
        super(ThreadWithReturn, self).join()
        if self.exc:
            raise self.exc
        return self._return

    def getName(self):
        return Thread.getName(self)

    def how_many_threads_active(self):
        return active_count()


class Logger():
    def __init__(self, log_path):
        self.log_path = log_path

    def init_logger(self):
        """Logger for NoFever. Saves log history in one file, and ongoingly writes log to the other file.
        'LOG_FILE_MAX_SIZE_MB' variable controls how many megabytes of log data to store.
        Uses 'utf-8' encoding to allow for special characters: rød grød med fløde!

        Returns:
        app_log: logger handle to use for logging. Example: app_log.critical('Critical message!')
        """
        log_formatter = logging.Formatter('%(asctime)s :: %(message)s')
        # log_formatter = logging.Formatter('%(message)s')
        path = os.path.dirname(os.path.abspath(__file__))
        logFile = os.path.join(path, self.log_path)

        my_handler = RotatingFileHandler(logFile,
                                         mode='a',
                                         maxBytes=LOG_FILE_MAX_SIZE_MB * 1024 * 1024,
                                         backupCount=1,
                                         encoding='utf-8',
                                         delay=0)
        my_handler.setFormatter(log_formatter)
        my_handler.setLevel(logging.WARNING)

        app_log = logging.getLogger('root')
        app_log.setLevel(logging.WARNING)
        app_log.addHandler(my_handler)
        return app_log


class NofeverConfig():
    def __init__(self, cfg_relative_path='config/nofever_settings.ini'):
        self.cfg_path = cfg_relative_path
        self.root_path = os.path.dirname(os.path.abspath(__file__))
        self.full_path = os.path.join(self.root_path, self.cfg_path)
        self.default_rel_path = 'config/default_settings.ini'
        self.full_default_path = os.path.join(self.root_path, self.default_rel_path)

    # READ CONFIG VALUES AND STORE THEM
    def init_config(self):
        # Check if config file exists. If not -> create new file.
        if not os.path.exists(self.full_path):
            print(self.full_path)
            os.mknod(self.full_path)

        # Check if config file is empty. If empty -> Set default settings.
        if os.stat(self.full_path).st_size == 0:
            self.cfg_set_default_settings()

        config = configparser.ConfigParser()
        config.read(self.full_path)
        return config

    def cfg_set_default_settings(self):
        with open(self.full_default_path, 'r') as default_settings, open(self.full_path, 'a') as new_file:
            for line in default_settings:
                new_file.write(line)


NOC = NofeverConfig()
CONFIG_NOFEVER_SETTINGS = NOC.init_config()
SETTINGS = CONFIG_NOFEVER_SETTINGS['utils']
LOG_FILE_MAX_SIZE_MB = SETTINGS.getint('LOG_FILE_MAX_SIZE_MB')
INTERNET_DEVICE_NAME = SETTINGS['INTERNET_DEVICE_NAME']


LOG_PATH = SETTINGS['LOG_PATH']

LOGGER = Logger(LOG_PATH)
LOG = LOGGER.init_logger()

INT = Internet(INTERNET_DEVICE_NAME, LOG)
