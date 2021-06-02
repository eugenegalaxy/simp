# This python script will move files from src to dst directory AND merge text contents of .txt files
# Purpose: Collecting OpenImages v6 multiple class images and label into common folders ('test', 'train', 'valid')
# while merging .txt label of images used in multiple classes #

import os
import time


def darknet_label_index_changer(src_pt, old_idx_list, new_idx_list):
    if len(old_idx_list) != len(new_idx_list):
        raise ValueError('Old and new id number does not match.')
    base_pt = os.path.dirname(__file__)
    full_src_pt = os.path.join(base_pt, src_pt)
    for item in os.listdir(full_src_pt):
        src_file_pt = os.path.join(full_src_pt, item)
        if os.path.splitext(item)[1] == '.txt':
            with open(src_file_pt, 'r') as fin:
                lines = fin.readlines()
                for idx, line in enumerate(lines):
                    for n in range(len(old_idx_list)):
                        if line[0] == old_idx_list[n]:
                            lines[idx] = line.replace(old_idx_list[n], new_idx_list[n], 1)
            with open(src_file_pt, 'w') as fout:
                fout.writelines(lines)


def img_and_label_mover(src_pt, dst_pt, number_img):
    base_pt = os.path.dirname(__file__)
    full_src_pt = os.path.join(base_pt, src_pt)
    full_dst_pt = os.path.join(base_pt, dst_pt)

    filenames_list = []
    file_dmod_list = []

    for filename in os.listdir(full_src_pt):
        path_to_file = os.path.join(full_src_pt, filename)
        filenames_list.append(filename)
        file_dmod_list.append(os.path.getmtime(path_to_file))

    sorted_dmods, sorted_filenames = zip(*sorted(zip(file_dmod_list, filenames_list)))
    sorted_filenames = sorted_filenames[::-1]

    for item in os.listdir(full_src_pt):
        if item in sorted_filenames[0:number_img]:
            src_file_pt = os.path.join(full_src_pt, item)
            dst_file_pt = os.path.join(full_dst_pt, item)
            if os.path.exists(dst_file_pt) and (os.path.splitext(item)[1] == '.jpg' or os.path.splitext(item)[1] == 'txt'):
                if os.path.splitext(item)[1] == '.txt':
                    with open(src_file_pt, 'r') as fout:
                        lines = fout.readlines()
                    with open(dst_file_pt, 'a') as fin:
                        for line in lines:
                            fin.write(line)
                    os.remove(src_file_pt)
                elif os.path.splitext(item)[1] == '.jpg':
                    os.rename(src_file_pt, dst_file_pt)
                else:
                    print('File {0} is neither .jpg or .txt. Doing nothing with it...\n Full source path: {1}'.format(item, src_file_pt))
            else:
                os.rename(src_file_pt, dst_file_pt)


def mask_data_mover():
    src1 = 'mask_dataset/images/valid'
    dst1 = 'oi_dataset/images/valid'
    nmb1 = 1000000
    img_and_label_mover(src1, dst1, nmb1)
    time.sleep(1)

    src2 = 'mask_dataset/images/test'
    dst2 = 'oi_dataset/images/test'
    nmb2 = 1000000
    img_and_label_mover(src2, dst2, nmb2)
    time.sleep(1)

    src3 = 'mask_dataset/images/train'
    dst3 = 'oi_dataset/images/train'
    nmb3 = 1000000
    img_and_label_mover(src3, dst3, nmb3)
    time.sleep(1)

    src6 = 'mask_dataset/labels/valid'
    dst6 = 'oi_dataset/labels/valid'
    nmb6 = 1000000
    img_and_label_mover(src6, dst6, nmb6)
    time.sleep(1)

    src7 = 'mask_dataset/labels/test'
    dst7 = 'oi_dataset/labels/test'
    nmb7 = 1000000
    img_and_label_mover(src7, dst7, nmb7)
    time.sleep(1)

    src8 = 'mask_dataset/labels/train'
    dst8 = 'oi_dataset/labels/train'
    nmb8 = 1000000
    img_and_label_mover(src8, dst8, nmb8)
    time.sleep(1)


def oi_dataset_mover():
    #  ============= VALID LABEL DATA ==================
    src1 = 'oi_dataset/human mouth/darknet'
    dst1 = 'oi_dataset/labels/valid'
    nmb1 = 50
    img_and_label_mover(src1, dst1, nmb1)
    time.sleep(1)

    src2 = 'oi_dataset/human eye/darknet'
    dst2 = 'oi_dataset/labels/valid'
    nmb2 = 50
    img_and_label_mover(src2, dst2, nmb2)
    time.sleep(1)

    src3 = 'oi_dataset/human ear/darknet'
    dst3 = 'oi_dataset/labels/valid'
    nmb3 = 50
    img_and_label_mover(src3, dst3, nmb3)
    time.sleep(1)

    src4 = 'oi_dataset/human nose/darknet'
    dst4 = 'oi_dataset/labels/valid'
    nmb4 = 50
    img_and_label_mover(src4, dst4, nmb4)
    time.sleep(1)

    src5 = 'oi_dataset/human head/darknet'
    dst5 = 'oi_dataset/labels/valid'
    nmb5 = 50
    img_and_label_mover(src5, dst5, nmb5)
    time.sleep(1)
    #  ==========================================

    #  =============  TEST LABEL DATA ==================
    src6 = 'oi_dataset/human mouth/darknet'
    dst6 = 'oi_dataset/labels/test'
    nmb6 = 50
    img_and_label_mover(src6, dst6, nmb6)
    time.sleep(1)

    src7 = 'oi_dataset/human eye/darknet'
    dst7 = 'oi_dataset/labels/test'
    nmb7 = 50
    img_and_label_mover(src7, dst7, nmb7)
    time.sleep(1)

    src8 = 'oi_dataset/human ear/darknet'
    dst8 = 'oi_dataset/labels/test'
    nmb8 = 50
    img_and_label_mover(src8, dst8, nmb8)
    time.sleep(1)

    src9 = 'oi_dataset/human nose/darknet'
    dst9 = 'oi_dataset/labels/test'
    nmb9 = 50
    img_and_label_mover(src9, dst9, nmb9)
    time.sleep(1)

    src10 = 'oi_dataset/human head/darknet'
    dst10 = 'oi_dataset/labels/test'
    nmb10 = 50
    img_and_label_mover(src10, dst10, nmb10)
    time.sleep(1)
    #  ==========================================

    #  =============  TRAIN LABEL DATA ============
    src11 = 'oi_dataset/human mouth/darknet'
    dst11 = 'oi_dataset/labels/train'
    nmb11 = 50
    img_and_label_mover(src11, dst11, nmb11)
    time.sleep(1)

    src12 = 'oi_dataset/human eye/darknet'
    dst12 = 'oi_dataset/labels/train'
    nmb12 = 50
    img_and_label_mover(src12, dst12, nmb12)
    time.sleep(1)

    src13 = 'oi_dataset/human ear/darknet'
    dst13 = 'oi_dataset/labels/train'
    nmb13 = 50
    img_and_label_mover(src13, dst13, nmb13)
    time.sleep(1)

    src14 = 'oi_dataset/human nose/darknet'
    dst14 = 'oi_dataset/labels/train'
    nmb14 = 50
    img_and_label_mover(src14, dst14, nmb14)
    time.sleep(1)

    src15 = 'oi_dataset/human head/darknet'
    dst15 = 'oi_dataset/labels/train'
    nmb15 = 50
    img_and_label_mover(src15, dst15, nmb15)
    time.sleep(1)
    #  ==========================================


## FOR MASK DATASET!
# old_idx = ['0', '1', '2']
# new_idx = ['5', '6', '7']
# src_path = 'x'
# darknet_label_index_changer(src_path, old_idx, new_idx)
