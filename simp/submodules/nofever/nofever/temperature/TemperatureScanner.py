import numpy as np
import time
from itertools import islice
import matplotlib.pyplot as plt

from utils import CONFIG_NOFEVER_SETTINGS, DebugPrint

DEBUG_MODE = False
DEBUG = DebugPrint(DEBUG_MODE)

SETTINGS = CONFIG_NOFEVER_SETTINGS['temperature_scanner']
MLX_JETSON_ENABLED = SETTINGS.getboolean('MLX_JETSON_ENABLED')

if MLX_JETSON_ENABLED:
    from temperature.MLX90614 import MLX90614
    from temperature.kalman_filter import KalmanFilter
else:
    from arduino_serial import ArduinoSerial


class TemperatureScanner(object):

    TMP_SLIDING_WINDOW_RANGE = SETTINGS.getint('TMP_SLIDING_WINDOW_RANGE')

    def __init__(self):
        if MLX_JETSON_ENABLED is True:
            self.MLX = MLX90614(voltage=3.33)
            # Default Q=0.001, R=0.04, init_x=33.3, init_p=10000
            self.KalmanObject = KalmanFilter(init_x=32)
        else:
            self.AS = ArduinoSerial()
            time.sleep(2)  # giving enough time for arduino to boot

    if MLX_JETSON_ENABLED is True:
        def get_temperature(self, timer):
            time_start = time.time()
            all_temps = []
            while (time.time() - time_start <= timer):
                temp_obj = self.MLX.get_obj_temp_C()
                self.KalmanObject.add_measurement(temp_obj)
                kalmaned = self.KalmanObject.get_estimate()
                all_temps.append(kalmaned)  # NOTE not using ambient here.
                time.sleep(0.1)
            time_end = time.time()
            temperature = self._sliding_average(all_temps)
            max_temperature = max(temperature)
            DEBUG('Temperature recorded is: {}'.format(max_temperature))
            DEBUG('Time for temperature measurement is: {:1.4f} seconds'.format(time_end - time_start))
            self.KalmanObject.reset()
            return max_temperature

        def get_obj_temp_C_raw(self):
            return self.MLX.get_obj_temp_C()

        def get_amb_temp_C_raw(self):
            return self.MLX.get_amb_temp_C()

        def get_obj_temp_F_raw(self):
            return self.MLX.get_obj_temp_F()

        def get_amb_temp_F_raw(self):
            return self.MLX.get_amb_temp_F()
    else:
        def get_temperature(self, timer):
            time_start = time.time()
            temp = []
            # ambient = []
            while (time.time() - time_start <= timer):
                obj, amb = self.get_one_temp_C()
                temp.append(obj)  # NOTE not using ambient here.
                time.sleep(0.1)
            time_end = time.time()
            temperature = self._sliding_average(temp)
            max_temperature = max(temperature)
            DEBUG('Temperature recorded is: {}'.format(max_temperature))
            # ambient = self.AS.getTemp()[1]
            # DEBUG('Ambient temperature is : {}'.format(ambient))
            DEBUG('Time for temperature measurement is: {:1.4f} seconds'.format(time_end - time_start))
            #  TODO temperature_ambient_combo
            return max_temperature

        def get_obj_temp_C_kalman(self):
            return self.AS.getTemp(unit='C_kalman')[0]

        def get_amb_temp_C_kalman(self):
            return self.AS.getTemp(unit='C_kalman')[1]

        def get_obj_temp_C_raw(self):
            return self.AS.getTemp(unit='C_raw')[0]

        def get_amb_temp_C_raw(self):
            return self.AS.getTemp(unit='C_raw')[1]

        def get_obj_temp_F_raw(self):
            return self.AS.getTemp(unit='F_raw')[0]

        def get_amb_temp_F_raw(self):
            return self.AS.getTemp(unit='F_raw')[1]

    def plot_measurements(self, list1, list2, save_path,
                          ylabel='Object temp', xlabel='Measurements', title='Something'):
        plt.plot(list1, color='red')
        plt.plot(list2, color='blue')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.savefig(save_path, bbox_inches='tight')

    def _sliding_window(self, temps):
        itterations = iter(temps)
        result = tuple(islice(itterations, self.TMP_SLIDING_WINDOW_RANGE))
        if len(result) == self.TMP_SLIDING_WINDOW_RANGE:
            yield result
        for elem in itterations:
            result = result[1:] + (elem,)
            yield sum(result) / len(result)

    def _sliding_average(self, temp):
        Temperatures = []
        floatTemperatures = []
        for value in self._sliding_window(temp):
            floatTemperatures = np.array(value, dtype=np.float32)
            Temperatures = np.append(Temperatures, floatTemperatures)
        return Temperatures
