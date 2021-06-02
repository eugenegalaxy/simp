import os

os.chdir('/home/eugenegalaxy/Documents/projects/simp/simp/tests/yolo_dataset/mask_incorrect_detections')
COUNT = 0


def increment():
    global COUNT
    COUNT = COUNT + 1


def renamer():
    for f in os.listdir():
        f_name, f_ext = os.path.splitext(f)
        f_name = 'nofever_{0:04}'.format(COUNT)
        increment()

        new_name = '{}{}'.format(f_name, f_ext)
        os.rename(f, new_name)


def renamer_parts():
    for f in os.listdir():
        f_name, f_ext = os.path.splitext(f)
        # f_name = 'something'
        parts = f_name.split('height')
        f_name = parts[0]
        parts2 = f_name.split('_')
        f_name = '{0:04}'.format(COUNT) + '_old_' + parts2[1] + '_' + parts2[2]
        new_name = '{}{}'.format(f_name, f_ext)
        os.rename(f, new_name)
        increment()


renamer_parts()
