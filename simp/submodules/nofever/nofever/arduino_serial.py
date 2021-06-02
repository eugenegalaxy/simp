import serial
import glob
import json
import time

from utils import LOG, CONFIG_NOFEVER_SETTINGS, DebugPrint

DEBUG_MODE = True
DEBUG = DebugPrint(DEBUG_MODE)

SETTINGS = CONFIG_NOFEVER_SETTINGS['arduino_serial']


class ArduinoSerial:
    '''Serial interface that communicates with Arduino via JSON strings over serial port'''
    BAUDRATE = SETTINGS.getint('BAUDRATE')
    READ_TIMEOUT = SETTINGS.getfloat('READ_TIMEOUT')
    WRITE_TIMEOUT = SETTINGS.getfloat('WRITE_TIMEOUT')
    LOOP_TIMEOUT = SETTINGS.getfloat('LOOP_TIMEOUT')

    def __init__(self):
        self.ser = self.openSerialPort()
        if self.ser.is_open:
            DEBUG('Serial port is open')
            LOG.critical('Serial port is open')
            self.ser.flushInput()
            self.ser.flushOutput()
        else:
            DEBUG('Cannot open Serial Port')
            LOG.critical('Cannot open Serial Port')

        def __del__(self):
            if self.ser != 0:
                self.ser.close()

    def openSerialPort(self):
        port = self.serialPorts()
        ports_str = "Available ports: {0} Connecting to: {1}".format(port, port[0])
        DEBUG(ports_str)
        LOG.critical(ports_str)
        ser = serial.Serial(port[0], self.BAUDRATE,
                            timeout=self.READ_TIMEOUT, write_timeout=self.WRITE_TIMEOUT)
        return ser

    def serialPorts(self):
        """ Function: lists serial port names
            Raise:  EnvironmentError: On unsupported or unknown platforms
            Return: A list of the serial ports available on the system
        """
        # if glob.glob('/dev/ttyASM*'):
        #     # this excludes your current terminal "/dev/tty"
        #     ports = glob.glob('/dev/ttyASM*')
        # elif glob.glob('/dev/ttyUSB*'):
        #     ports = glob.glob('/dev/ttyUSB*')
        # elif glob.glob('/dev/tty[A-Za-z]*'):
        #     ports = glob.glob('/dev/tty[A-Za-z]*')
        # else:
        #     raise EnvironmentError('Unsupported platform')

        if glob.glob('/dev/ttyUSB*'):
            ports = glob.glob('/dev/ttyUSB*')
        else:
            raise EnvironmentError('Unsupported platform')

        result = []
        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                result.append(port)
            except (OSError, serial.SerialException):
                pass
        return result

    def sendSerial(self, data_package):
        msg = json.dumps(data_package)
        msg = msg + '\n'  # Newline is required on Arduino side for readStringUntil('/n')
        msg = msg.encode()
        self.ser.write(msg)

    def readSerial(self):
        t_start = time.time()
        while (time.time() - t_start) < self.LOOP_TIMEOUT:
            while self.ser.in_waiting > 0:
                in_data = self.ser.readline()
                if in_data:
                    return in_data
            time.sleep(0.001)
        return None

    def arduinoReboot(self):
        data_to_send = dict(reset=1)
        self.sendSerial(data_to_send)

    def getTemp(self, unit='C_kalman'):
        if unit == 'C_kalman':  # Celsius temperature reading adjusted by Kalman Filter (always running)
            data_to_send = dict(C_kalman=1)  # 1 is just a dummy value, can be anything...
        elif unit == 'C_raw':  # Celsius reading from sensor (no Kalman)
            data_to_send = dict(C_raw=1)
        elif unit == 'F_raw':  # Fahrenheit reading from sensor (no Kalman)
            data_to_send = dict(F_raw=1)
        else:
            raise ValueError("'unit' must be a string: C_kalman, C_Raw, or F_raw.")

        self.sendSerial(data_to_send)
        temp = self.readSerial()

        if temp is not None:
            MLX_read = json.loads(temp).get('data')
            if MLX_read is not None:
                return MLX_read
            else:
                return None
        else:
            return None  # do nothing, not a valid json
