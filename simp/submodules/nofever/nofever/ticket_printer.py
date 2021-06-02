import os
from time import time
from datetime import datetime
import usb

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from brother_ql.raster import BrotherQLRaster
from brother_ql.backends import backend_factory
from brother_ql import create_label
from utils import LOG, DebugPrint, CONFIG_NOFEVER_SETTINGS

SETTINGS = CONFIG_NOFEVER_SETTINGS['label_printer']

DEBUG_MODE = True
DEBUG = DebugPrint(DEBUG_MODE)

DEBUG_SHOW_TEMPLATE = False
DEBUG_SHOW_GENERATED_IMG = False


class LabelPrinter(object):
    CHOSEN_LANGUAGE = SETTINGS['CHOSEN_LANGUAGE']
    DEVICE_NAME = SETTINGS['DEVICE_NAME']
    PRINTER_MODEL = SETTINGS['PRINTER_MODEL']
    LABEL_SIZE = SETTINGS['LABEL_SIZE']
    BACKEND = SETTINGS['BACKEND']
    TICKET_LENGTH = SETTINGS.getint('TICKET_LENGTH')
    TICKET_WIDTH = SETTINGS.getint('TICKET_WIDTH')

    PRINT_TIMEOUT = SETTINGS.getint('PRINT_TIMEOUT')

    MAX_IMG_LENGTH_IN_TICKET = SETTINGS.getint('MAX_IMG_LENGTH_IN_TICKET')
    MAX_IMG_WIDTH_IN_TICKET = SETTINGS.getint('MAX_IMG_WIDTH_IN_TICKET')

    backend_factory = backend_factory(BACKEND)

    def __init__(self):
        if self.CHOSEN_LANGUAGE == 'dk':
            self.contents = self.get_contents_danish()
        elif self.CHOSEN_LANGUAGE == 'en':
            self.contents = self.get_contents_english()
        else:
            self.contents = self.get_contents_danish()

        self.check_usb_connection()

    def check_usb_connection(self):
        list_available_devices = self.backend_factory['list_available_devices']()
        # assert (len(list_available_devices) > 0), 'No available Brother Label printer. Check if printer is turned on.'
        if len(list_available_devices) < 1:
            DEBUG('Brother Label printer is not connected or is not turned on.')
            LOG.warning('Brother Label printer is not connected or is not turned on.')
            return False
        else:
            return True

    def ticket_handling(self, temp, mask):
        '''
            Main function that will check if the result needs to be printed,
            will generate an empty template, fill result in, and print it.
        '''
        enabled = self.check_usb_connection()
        if enabled is False:
            return None

        temp_contents, mask_contents = self.filter_logics(temp, mask)
        if temp_contents is None:
            return None

        empty_ticket = self.create_ticket_template()
        final_ticket = self.generate_result(empty_ticket, temp_contents, mask_contents)
        self.print_brother_label(final_ticket)

    def filter_logics(self, temp, mask):
        # This part prevents label print when temperature is BLUE/COLD.
        if temp == 'temperature_measured_wrong' or temp == 'temperature_measured_under_twice':
            return None, None

        if temp == 'temperature_measured_good':
            temp_contents = self.contents['temp_normal']
        elif temp == 'temperature_measured_slightly_high':
            temp_contents = self.contents['temp_slightly_elevated']
        elif temp == 'temperature_measured_bad':
            temp_contents = self.contents['temp_elevated']
        elif temp == 'temperature_measured_over_twice':
            temp_contents = self.contents['temp_coffee']
        else:
            DEBUG('some fuck up in ticket_print logics...')

        if mask == 'mask_on':
            mask_contents = self.contents['mask_on']
        elif mask == 'mask_off':
            mask_contents = self.contents['mask_off']
        elif mask == 'mask_wrong':
            mask_contents = self.contents['mask_wrong']
        elif mask == 'no_detections':
            mask_contents = self.contents['mask_off']
        else:
            DEBUG('some fuck up in ticket_print logics...')
        return temp_contents, mask_contents

    def create_ticket_template(self):
        # Create empty white background image with 696x300 resolution. # "1" -> 1bit BW mode. color 1 -> white
        background = Image.new(mode='1',
                               size=(self.TICKET_LENGTH, self.TICKET_WIDTH),
                               color=1)

        underqr = self.contents['underqr']
        headline1 = self.contents['headline1']
        headline2 = self.contents['headline2']
        temp = self.contents['temp']
        mask = self.contents['mask']
        time_date = self.contents['time_date']
        time_date['text'] = str(datetime.now().strftime("%H:%M:%S %d.%m.%Y"))

        all_strings = [underqr, headline1, headline2, temp, mask, time_date]  # to loop it below (less code)

        # Create drawing handle on background image
        draw = ImageDraw.Draw(background)

        qrcode_img = self.open_image(self.contents['qrcode_img']['rel_path'])
        if qrcode_img is not None:
            background.paste(qrcode_img, self.contents['qrcode_img']['pos'])

        # Fill background with objects at specificed (x,y) pixel locations.
        for x in all_strings:
            draw.text(x['pos'], x['text'], font=ImageFont.truetype(x['font'], x['size']), fill='black')

        ticket_template = background
        if DEBUG_SHOW_TEMPLATE is True:
            ticket_template.show()
        return ticket_template

    def print_brother_label(self, img):
        qlr = BrotherQLRaster(self.PRINTER_MODEL)
        qlr.exception_on_warning = True
        create_label(qlr, img, self.LABEL_SIZE)  # resizes image?
        BrotherQLBackend = self.backend_factory['backend_class']
        printer = BrotherQLBackend(self.DEVICE_NAME)
        printer.write_timeout = self.PRINT_TIMEOUT

        t1 = time()
        # usb.core.USBError <---- crashed printer.write with this error
        try:
            printer.write(qlr.data, )
            r = ">"
            while r:
                r = printer.read()
        except usb.core.USBError as e:
            DEBUG(e, "<- Brother Printer has been disconnected/disabled during the printing process.")
            LOG.warning(e, "<- Brother Printer has been disconnected/disabled during the printing process.")
            return None
        t2 = time()
        # if printing took more time than "write timeout" -> its not working. We take 90% of write_timeout for safety..
        if t2 - t1 >= (self.PRINT_TIMEOUT / 1000) - ((self.PRINT_TIMEOUT / 1000) * 0.1):
            err = 'Brother Label printer cannot print. Check if lid is closed, roll is mounted in and pushed to feeder.'
            DEBUG(err)
            LOG.warning(err)

    def generate_result(self, ticket_template, temp, mask):
        # Create drawing handle on background image and insert text contents
        draw = ImageDraw.Draw(ticket_template)
        draw.text(temp['pos'], temp['text'], font=ImageFont.truetype(temp['font'], temp['size']), fill='black')
        draw.text(mask['pos'], mask['text'], font=ImageFont.truetype(mask['font'], mask['size']), fill='black')

        ready_ticket = ticket_template
        if DEBUG_SHOW_GENERATED_IMG is True:
            ready_ticket.show()
        return ready_ticket

    def open_image(self, rel_path):
        base_path = os.path.dirname(__file__)
        full_path = os.path.join(base_path, rel_path)

        if os.path.isfile(full_path):
            if '.jpg' in full_path or '.jpeg' in full_path or '.png' in full_path:
                try:
                    img = Image.open(full_path)
                    return img
                except (FileNotFoundError, UnidentifiedImageError) as e:
                    print(e)
        for file_name in sorted(os.listdir(full_path)):
            ext = os.path.splitext(file_name)[1]
            if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                img_path = os.path.join(full_path, file_name)
                try:
                    img = Image.open(img_path)
                    len, wid = img.size
                    if len > self.MAX_IMG_LENGTH_IN_TICKET:
                        new_len = self.MAX_IMG_LENGTH_IN_TICKET
                        len_diff_ratio = (len - self.MAX_IMG_LENGTH_IN_TICKET) / len
                        new_wid = wid - (wid * (len_diff_ratio))
                        if new_wid > self.MAX_IMG_WIDTH_IN_TICKET:
                            new_wid = self.MAX_IMG_WIDTH_IN_TICKET
                        new_size = (int(new_len), int(new_wid))
                        img = img.resize(new_size, Image.NEAREST)
                    return img
                except UnidentifiedImageError as e:
                    print(e)
                    break
        return None

    def get_contents_danish(self):
        danish_ticket = {
            # QR CODE IMAGE
            'qrcode_img': {'rel_path': 'labels/', 'pos': (430, 10)},
            # STATIC TEXT
            'underqr': {'text': 'Følg os', 'pos': (505, 257), 'font': 'Ubuntu-RI.ttf', 'size': 30},
            'headline1': {'text': 'Tak fordi du scannede', 'pos': (20, 20), 'font': 'Ubuntu-R.ttf', 'size': 38},
            'headline2': {'text': 'med #NoFever!', 'pos': (20, 55), 'font': 'Ubuntu-R.ttf', 'size': 38},
            'temp': {'text': 'Temperatur: ', 'pos': (20, 130), 'font': 'Ubuntu-R.ttf', 'size': 30},
            'mask': {'text': 'Mundbind:', 'pos': (20, 175), 'font': 'Ubuntu-R.ttf', 'size': 30},
            'time_date': {'text': 'will be initialized later', 'pos': (20, 257), 'font': 'Ubuntu-R.ttf', 'size': 25},
            # RESULT TEXT
            'temp_normal': {'text': 'NORMAL', 'pos': (200, 130), 'font': 'Ubuntu-B.ttf', 'size': 30},
            'temp_slightly_elevated': {'text': 'LIDT FORHØJET', 'pos': (200, 130), 'font': 'Ubuntu-B.ttf', 'size': 30},
            'temp_elevated': {'text': 'FORHØJET', 'pos': (200, 130), 'font': 'Ubuntu-B.ttf', 'size': 30},
            'temp_coffee': {'text': 'KAFFE KOP?', 'pos': (200, 130), 'font': 'Ubuntu-B.ttf', 'size': 30},
            'mask_on': {'text': 'PÅ', 'pos': (200, 175), 'font': 'Ubuntu-B.ttf', 'size': 30},
            'mask_off': {'text': 'IKKE PÅ', 'pos': (200, 175), 'font': 'Ubuntu-B.ttf', 'size': 30},
            'mask_wrong': {'text': 'FORKERT PÅ', 'pos': (200, 175), 'font': 'Ubuntu-B.ttf', 'size': 30},
        }
        return danish_ticket

    def get_contents_english(self):
        english_ticket = {
            # QR CODE IMAGE
            'qrcode_img': {'rel_path': 'labels/', 'pos': (430, 10)},
            # STATIC TEXT
            'underqr': {'text': 'Follow us', 'pos': (490, 257), 'font': 'Ubuntu-RI.ttf', 'size': 30},
            'headline1': {'text': 'Thank you for scanning', 'pos': (20, 20), 'font': 'Ubuntu-R.ttf', 'size': 38},
            'headline2': {'text': 'with #NoFever!', 'pos': (20, 55), 'font': 'Ubuntu-R.ttf', 'size': 38},
            'temp': {'text': 'Temperature : ', 'pos': (20, 130), 'font': 'Ubuntu-R.ttf', 'size': 27},
            'mask': {'text': 'Mask:', 'pos': (20, 175), 'font': 'Ubuntu-R.ttf', 'size': 27},
            'time_date': {'text': 'will be initialized later', 'pos': (20, 257), 'font': 'Ubuntu-R.ttf', 'size': 25},
            # RESULT TEXT
            'temp_normal': {'text': 'NORMAL', 'pos': (200, 130), 'font': 'Ubuntu-B.ttf', 'size': 27},
            'temp_slightly_elevated': {'text': 'ELEVATED A BIT', 'pos': (200, 130), 'font': 'Ubuntu-B.ttf', 'size': 27},
            'temp_elevated': {'text': 'ELEVATED', 'pos': (200, 130), 'font': 'Ubuntu-B.ttf', 'size': 27},
            'temp_coffee': {'text': 'COFFEE CUP?', 'pos': (200, 130), 'font': 'Ubuntu-B.ttf', 'size': 27},
            'mask_on': {'text': 'ON', 'pos': (200, 175), 'font': 'Ubuntu-B.ttf', 'size': 27},
            'mask_off': {'text': 'NOT ON', 'pos': (200, 175), 'font': 'Ubuntu-B.ttf', 'size': 27},
            'mask_wrong': {'text': 'ON BUT WRONG', 'pos': (200, 175), 'font': 'Ubuntu-B.ttf', 'size': 27},
        }
        return english_ticket

# export BROTHER_QL_PRINTER=usb://0x04f9:0x2042
# export BROTHER_QL_MODEL=QL-700
