
import os

import numpy as np
import cv2
import torch
from torchvision.utils import save_image

from simp.submodules.face_generator.face_generator.networks import GatedGenerator
from simp.submodules.face_generator.face_generator.configs import Config

abs_pt = '/home/eugenegalaxy/Documents/projects/simp/simp/submodules/face_generator/face_generator/weights/'

GAN_FACE_GEN_WEIGHT_PT = abs_pt + 'ImIn_GAN_PDE600+RR280_78e_70000it.pth'


CFG = '/home/eugenegalaxy/Documents/projects/simp/simp/submodules/face_generator/face_generator/configs/facemask.yaml'


class Predictor():
    def __init__(self, cfg=Config(CFG)):
        self.cfg = cfg
        self.device = torch.device('cuda:0' if cfg.cuda else 'cpu')

        self.inpaint = GatedGenerator().to(self.device)
        self.inpaint.load_state_dict(torch.load(GAN_FACE_GEN_WEIGHT_PT, map_location='cpu')['G'])
        self.inpaint.eval()

    def save_image(self, img_list, save_img_path, nrow):
        img_list  = [i.clone().cpu() for i in img_list]
        imgs = torch.stack(img_list, dim=1)
        imgs = imgs.view(-1, *list(imgs.size())[2:])
        save_image(imgs, save_img_path, nrow=nrow)
        print(f"Save image to {save_img_path}")

    def generate_face(self, image, mask, no_img_save=True):
        ''' NOFEVER VERSION!
            image = path to the same image that 'mask' binary sensor was made of.
            mask = torch.Tensor of mask binary map. Dimension = ([1, 1, image_width, image_height])
        '''

        if type(image) == str:
            if os.path.isfile(image):
                _, ext = os.path.splitext(image)
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    img = cv2.imread(image)
            elif os.path.isdir(image):
                raise TypeError("'image' argument must be either a path to img or a numpy.ndarray object.")
        elif type(image) == np.ndarray:
            img = image
        else:
            raise("'image' must be either '/path/to/img.jpg' 'numpy.ndarray' object (e.g. from 'cv2.imread'). ")

        assert(type(mask) == torch.Tensor), "Argument 'mask' must be a class object of torch.Tensor"

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.astype(np.float32) / 255.0)
        img = img.permute(2, 0, 1).contiguous()
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, out = self.inpaint(img, mask)
            inpaint = img * (1 - mask) + out * mask
        masks = img * (1 - mask) + mask  # torch.cat([outputs, outputs, outputs], dim=1)

        if no_img_save is False:
            dirname = os.path.dirname(image)
            weights_filename = os.path.basename(GAN_FACE_GEN_WEIGHT_PT).split('.')[0]
            filename, extension = os.path.splitext(os.path.basename(image))
            output_filename = filename + '_' + weights_filename + extension
            out_file_pt = os.path.join(dirname, output_filename)
            self.save_image([img, masks, inpaint], out_file_pt, nrow=3)
        return inpaint, img, masks
