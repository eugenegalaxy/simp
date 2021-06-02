# Marie Hildebrand Grevil
# Last changes: 2021-03-08

import smbus # System Management Bus - used for I2C.
from time import sleep


# MLX90614 class code is provided by user Stephen2615 on WordPress (https://stephen2615.wordpress.com/)
# Modified to include voltage compensation.
class MLX90614():
    # Datasheet: http://www.haoyuelectronics.com/Attachment/GY-906/MLX90614.pdf
    # RAM addresses:
    __MLX90614_RAWIR1 = 0x04
    __MLX90614_RAWIR2 = 0x05
    __MLX90614_TA = 0x06
    __MLX90614_TOBJ1 = 0x07
    __MLX90614_TOBJ2 = 0x08
    __MLX90614_TOMAX = 0x20
    __MLX90614_TOMIN = 0x21
    __MLX90614_PWMCTRL = 0x22
    __MLX90614_TARANGE = 0x23
    __MLX90614_EMISS = 0x24
    __MLX90614_CONFIG = 0x25
    __MLX90614_ADDR = 0x0E
    __MLX90614_ID1 = 0x3C
    __MLX90614_ID2 = 0x3D
    __MLX90614_ID3 = 0x3E
    __MLX90614_ID4 = 0x3F

    __comm_retries = 5
    __comm_sleep_amount = 0.1

    def __init__(self, address=0x5A, bus_num=1, voltage=3.0):
        self.bus_num = bus_num
        self.address = address
        self.bus = smbus.SMBus(bus=bus_num)
        self.volt_ideal = 3.0
        self.volt_supply = voltage

    def __read_reg(self, reg_addr):
        err = None
        for i in range(self.__comm_retries):
            try:
                return self.bus.read_word_data(self.address, reg_addr)
            except IOError as e:
                err = e
                sleep(self.__comm_sleep_amount)
                raise err

    def __write_reg(self, reg_addr, value):
        err = None
        for i in range(self.__comm_retries):
            try:
                print('writing {0} to {1} address'.format(value, reg_addr))
                return self.bus.write_word_data(self.address, reg_addr, value)
            except IOError as e:
                err = e
                sleep(self.__comm_sleep_amount)
                raise err

    # Converts raw data to Celsius.
    def __data_to_temp(self, data):
        temp = (data * 0.02) - 273.15
        temp = temp - 0.6 * (self.volt_supply - self.volt_ideal)
        return temp

    def get_amb_temp_C(self):
        data = self.__read_reg(self.__MLX90614_TA)
        celsius = self.__data_to_temp(data)
        return celsius

    def get_obj_temp_C(self):
        data = self.__read_reg(self.__MLX90614_TOBJ1)
        celsius = self.__data_to_temp(data)
        return celsius

    def get_obj_temp_F(self):
        data = self.__read_reg(self.__MLX90614_TOBJ1)
        temp = self.__data_to_temp(data)
        fahrenheit = (temp * 9 / 5) + 32
        return fahrenheit

    def get_amb_temp_F(self):
        data = self.__read_reg(self.__MLX90614_TA)
        temp = self.__data_to_temp(data)
        fahrenheit = (temp * 9 / 5) + 32
        return fahrenheit

    def read_emissivity(self):
        emis = self.__read_reg(self.__MLX90614_EMISS)
        scaled_emis = emis / 65535.0
        print('Emissivity: {0}  | Scaled Emis: {1}'.format(emis, scaled_emis))
        return scaled_emis

    def write_emissivity(self, emis):
        unscaled_emis = int(0xffff * emis)
        print(unscaled_emis)
        # self.__write_reg(self.__MLX90614_EMISS, 0)
        # sleep(0.01)
        self.__write_reg(self.__MLX90614_EMISS, unscaled_emis)
        sleep(0.01)
