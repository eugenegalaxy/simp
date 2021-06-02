# Installation:
# pip install pyzbar
# sudo apt-get install zbar-tools

import cv2
from pyzbar import pyzbar


def qr_decode_video(video_input=4):
    cam = cv2.VideoCapture(video_input)

    while True:
        _, img = cam.read()
        barcodes = pyzbar.decode(img)
        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type
            text = "{} ({})".format(barcodeData, barcodeType)
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print("[INFO] found {} barcode {}".format(barcodeType, barcodeData))
        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord("Q"):
            break
    cam.release()
    cv2.destroyAllWindows()


def qr_decode_img(img, save_img=False):
    barcodes = pyzbar.decode(img)
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        text = "{} ({})".format(barcodeData, barcodeType)
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        print("Found {} barcode: {}".format(barcodeType, barcodeData))
        cv2.imshow("img", img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        if save_img:
            cv2.imwrite("new_img.jpg", img)


# qr_decode_video(video_input=4)
path = '/home/eugenegalaxy/Documents/tmp_images/images/1.jpg'
img = cv2.imread(path)
qr_decode_img(img, save_img=True)
