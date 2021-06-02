import warnings
import sys
import os
import traceback
import requests
import datetime
from utils import LOG, CONFIG_NOFEVER_SETTINGS, DebugPrint, INT

DEBUG_MODE = True
DEBUG = DebugPrint(DEBUG_MODE)

warnings.filterwarnings("ignore")  # To disable RuntimeWarning popped by numpy (harmless and accounted for)
SETTINGS = CONFIG_NOFEVER_SETTINGS['main']

NOFEVER_WALL_MOUNT = SETTINGS.getboolean('WALL_MOUNT_VERSION')
if NOFEVER_WALL_MOUNT is True:
    from static_nofever_algorithm import NoFever
else:
    from nofever_algorithm import NoFever


def save_pid():
    pid = os.getpid()
    DEBUG('PID is {}'.format(pid))
    LOG.warning('PID is {}'.format(pid))
    f = open(pid_path, "w+")
    f.write(str(pid))
    f.close()


def start_NoFever(HOST_IP):
    """Run NoFever app.

    Args:
        HOST_IP (string): IP address of the HOST that holds the NoFever app server.
    """
    NoFever(HOST_IP)


base_path = os.path.dirname(os.path.abspath(__file__))
yolo_rel_path = 'detection/yolov5'
full_yolo_path = os.path.join(base_path, yolo_rel_path)
sys.path.insert(0, full_yolo_path)
pid_path = os.path.join(base_path, 'config/pid.txt')

IP_ADDRESS = SETTINGS['IP_ADDRESS']
PORT = SETTINGS['PORT']
HOST = 'http://{0}:{1}/'.format(IP_ADDRESS, PORT)


if __name__ == '__main__':
    try:
        save_pid()
        INT.connect()
        DEBUG('Current date & time: {}'.format(str(datetime.datetime.now())))
        LOG.warning('Current date & time: {}'.format(str(datetime.datetime.now())))
        INT.disconnect()
        start_NoFever(HOST)
    except Exception:
        requests.get(HOST + 'detection_idle')
        exc_type = sys.exc_info()[0]
        DEBUG(traceback.format_exc())
        DEBUG('Shutting down...')
        LOG.critical(traceback.format_exc())
        LOG.critical('Shutting down...')
