import os
import requests
from time import sleep

from utils import LOG, CONFIG_NOFEVER_SETTINGS
SETTINGS = CONFIG_NOFEVER_SETTINGS['main']

PID_CHECK_FREQUENCE_SECONDS = SETTINGS.getint('PID_CHECK_FREQUENCE_SECONDS')
PID_PATH = SETTINGS['PID_PATH']

base_path = os.path.dirname(os.path.abspath(__file__))
pid_path = os.path.join(base_path, PID_PATH)

IP_ADDRESS = SETTINGS['IP_ADDRESS']
PORT = SETTINGS['PORT']
HOST = 'http://{0}:{1}/'.format(IP_ADDRESS, PORT)


def send_app_request(host, signal):
    try:
        requests.get(host + signal)
    except requests.exceptions.ConnectionError as err:
        err_string = 'Error: {}.'.format(err)
        err_string2 = 'No connection to App Server. Terminating the program.'
        LOG.critical(err_string)
        LOG.critical(err_string2)
        raise SystemError(err_string2)


def check_pid(pid):
    """ Check For the existence of a unix pid. """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def check_nofever_running(timer):
    while True:
        f = open(pid_path, "r")
        pid = f.read()
        pid = int(pid)
        print('PID is {}'.format(pid))
        result = check_pid(pid)
        print('PID is alive: {}'.format(result))
        if result is False:
            send_app_request(HOST, 'detection_idle')
            print('PID {} is not found. Doing system reboot.'.format(pid))
            LOG.warning('PID {} is not found. Doing system reboot.'.format(pid))
            os.system('sudo reboot')
        sleep(timer)


check_nofever_running(PID_CHECK_FREQUENCE_SECONDS)
