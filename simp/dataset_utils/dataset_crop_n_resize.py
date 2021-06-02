from PIL import Image
import os
import cv2

# dir_path1 = "/home/eugenegalaxy/Desktop/joligan_480x480/trainA/binary"
# dir_path2 = "/home/eugenegalaxy/Desktop/joligan_480x480/trainA/masked"
# dir_path3 = "/home/eugenegalaxy/Desktop/joligan_480x480/trainB/nomask"
# dir_path4 = "/home/eugenegalaxy/Desktop/joligan_480x480/testA/binary"
# dir_path5 = "/home/eugenegalaxy/Desktop/joligan_480x480/testA/masked"
# dir_path6 = "/home/eugenegalaxy/Desktop/joligan_480x480/testB/nomask"

# dir_list = [dir_path1, dir_path2, dir_path3, dir_path4, dir_path5, dir_path6]
# img_path = "/home/eugenegalaxy/Desktop/joligan_resized_480/masked/2021-01-03-160044.jpg"

dir_path7 = "/home/eugenegalaxy/Desktop/image_inpaint-RR_all_670img_256x256/images/binary"

def dir_img_resizer(dir_path):
    width = 256
    height = 256
    resized_pt = os.path.join(dir_path, 'resized')
    if not os.path.isdir(resized_pt):
        os.mkdir(resized_pt)
    for img_file in os.listdir(dir_path):
        print(img_file)
        img_full_path = os.path.join(dir_path, img_file)
        if os.path.isfile(img_full_path):
            img_obj = cv2.imread(img_full_path, cv2.IMREAD_UNCHANGED)
            resized = cv2.resize(img_obj, (width, height), interpolation=cv2.INTER_AREA)
            img_file_resized = os.path.join(resized_pt, img_file)
            cv2.imwrite(img_file_resized, resized)


def dir_img_cropped(dir_path):
    left = 160
    top = 0
    right = 640
    bottom = 480
    resized_pt = os.path.join(dir_path, 'cropped')
    if not os.path.isdir(resized_pt):
        os.mkdir(resized_pt)
    for img_file in os.listdir(dir_path):
        print(img_file)
        img_full_path = os.path.join(dir_path, img_file)
        if os.path.isfile(img_full_path):
            img_obj = Image.open(img_full_path)
            img = img_obj.crop((left, top, right, bottom))
            img_file_resized = os.path.join(resized_pt, img_file)
            img.save(img_file_resized)

# print('First cropping images')
# dir_img_cropped(dir_path7)
# print(' ')
print('Then resizing to 256x256 images')
dir_ae = "/home/eugenegalaxy/Documents/projects/master_thesis/simp/inputs/coronapas_data/eugene"
dir_img_resizer(dir_ae)