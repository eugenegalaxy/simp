import os

from PIL import Image
from tqdm import tqdm

# dataset of 600 images
# train - first 480
# valid - last 120
train_valid_threshold = 200

REL_PT_MASKED = 'datasets/RR_masking/images/masked/'
REL_PT_BINARY = 'datasets/RR_masking/images/binary/'
REL_PT_ANNOTATIONS_TRAIN = 'datasets/RR_masking/annotations/train.csv'
REL_PT_ANNOTATIONS_VALID = 'datasets/RR_masking/annotations/val.csv'


def generate_segm_annotations():
    '''
        Generate annotations in .csv files for dataset used in UNet Segmentation model training.
        DESIRED OUTPUT FORMAT IN .CSV:
        ,img_name,mask_name
        0,masked\image1_masked.jpg,binary\image1_binary.jpg
        1,masked\image2_masked.jpg,binary\image2_binary.jpg
        2,masked\image3_masked.jpg,binary\image3_binary.jpg

        More doc TBA.
    '''
    base_pt = os.path.dirname(os.path.abspath(__file__))
    masked_pt = os.path.join(base_pt, REL_PT_MASKED)
    masked_files = os.listdir(masked_pt)
    masked_files.sort()

    binary_pt = os.path.join(base_pt, REL_PT_BINARY)
    binary_files = os.listdir(binary_pt)
    binary_files.sort()

    anno_train_pt = os.path.join(base_pt, REL_PT_ANNOTATIONS_TRAIN)
    anno_val_pt = os.path.join(base_pt, REL_PT_ANNOTATIONS_VALID)

    for idx, (mask_file, binary_file) in enumerate(zip(masked_files, binary_files)):
        entry_str = '{0},masked/{1},binary/{2}\n'.format(idx, mask_file, binary_file)

        if idx < train_valid_threshold:
            with open(anno_train_pt, 'a') as csv_file:
                csv_file.writelines(entry_str)
                csv_file.close()
        else:
            with open(anno_val_pt, 'a') as csv_file:
                csv_file.writelines(entry_str)
                csv_file.close()


def resize_images_in_directory(path, size_w, size_h):
    base_pt = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(base_pt, path)
    dirs = os.listdir(dir_path)
    for item in tqdm(dirs):
        img_path = os.path.join(path + item)
        print(img_path)
        if os.path.isfile(img_path):
            print('resizing')
            im = Image.open(img_path)
            f, e = os.path.splitext(img_path)
            # left = 140
            # top = 0
            # right = 620
            # bottom = 480
            # imResize = im.crop((left, top, right, bottom))
            imResize = im.resize((size_w, size_h), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=90)


# Resize images to a specific size if needed:
# resize_images_in_directory(REL_PT_MASKED, 256, 256)

generate_segm_annotations()
