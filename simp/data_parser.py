from json.decoder import JSONDecodeError
import os
import json
import cv2

from qr_parser import qr_decode_img


class DataParser():
    def __init__(self):
        self.base_pt = os.path.dirname(os.path.abspath(__file__))

    def parse_coronapas(self, qr_code_pt):
        abs_qr_code_pt = os.path.join(self.base_pt, qr_code_pt)
        qr_data_path = self.decode_QR(abs_qr_code_pt)
        _, _, img_full_path = self.load_img(qr_data_path)
        data = self.load_data(qr_data_path)
        return img_full_path, data

    def parse_nofever_scan(self, nofever_scan_pt):
        abs_nofever_scan_pt = os.path.join(self.base_pt, nofever_scan_pt)
        img, img_name, img_full_path = self.load_img(abs_nofever_scan_pt)
        data = self.load_data(abs_nofever_scan_pt)
        return img, img_name, img_full_path, data

    def decode_QR(self, path):
        qr_data_pt, qr_type = qr_decode_img(path)
        assert(qr_type == 'QRCODE'), "Provided code was not QR, but '{}' instead".format(qr_type)
        abs_qr_data_pt = os.path.join(self.base_pt, qr_data_pt)
        assert(os.path.isdir(abs_qr_data_pt)), "QR Code path is not a directory or does not exist. \nProvided\
            path: {}".format(abs_qr_data_pt)
        return abs_qr_data_pt

    def load_img(self, path):
        ''' Returns first found image in the provided path. If no images found -> returns None'''
        for img_file in os.listdir(path):
            img_full_path = os.path.join(path, img_file)
            img_filename = os.path.basename(img_full_path)
            if os.path.isfile(img_full_path):
                ext = os.path.splitext(img_full_path)[1]
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    img_obj = cv2.imread(img_full_path, cv2.IMREAD_UNCHANGED)
                    return img_obj, img_filename, img_full_path
        return None, None, None

    def load_data(self, path):
        ''' Returns contents of the first txt file found in the provided path. If no txt file found -> returns None'''
        for txt_file in os.listdir(path):
            txt_full_path = os.path.join(path, txt_file)
            if os.path.isfile(txt_full_path):
                ext = os.path.splitext(txt_full_path)[1]
                if ext == '.txt':
                    with open(txt_full_path, 'r') as file:
                        contents = file.read()
                        if len(contents) == 0:
                            return None
                        else:
                            try:
                                contents = json.loads(contents)
                                return contents
                            except JSONDecodeError as e:
                                print(e)
        return None
