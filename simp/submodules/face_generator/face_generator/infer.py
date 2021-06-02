
import os
import numpy as np
import cv2
import argparse
import sys

import torch
from torchvision.utils import save_image

from networks import UNetSemantic, GatedGenerator
from configs import Config

SEGMENTATION_WEIGHT_PT = 'weights/OLD_model_segm_100_12000.pth'


GAN_FACE_GEN_WEIGHT_PT = 'weights/RR_GAN_model_e100_it33000.pth'
# GAN_FACE_GEN_WEIGHT_PT = 'weights/RR_GAN_model_e199_it46000.pth'
# GAN_FACE_GEN_WEIGHT_PT = 'weights/RR_GAN_model_e299_it69000.pth'

class Predictor():
    def __init__(self, cfg=Config('./configs/facemask.yaml')):
        self.cfg = cfg
        self.device = torch.device('cuda:0' if cfg.cuda else 'cpu')
        self.masking = UNetSemantic().to(self.device)
        self.masking.load_state_dict(torch.load(SEGMENTATION_WEIGHT_PT, map_location='cpu'))

        self.inpaint = GatedGenerator().to(self.device)
        self.inpaint.load_state_dict(torch.load(GAN_FACE_GEN_WEIGHT_PT, map_location='cpu')['G'])
        self.inpaint.eval()

    def save_image(self, img_list, save_img_path, nrow):
        img_list  = [i.clone().cpu() for i in img_list]
        imgs = torch.stack(img_list, dim=1)
        imgs = imgs.view(-1, *list(imgs.size())[2:])
        save_image(imgs, save_img_path, nrow=nrow)
        print(f"Save image to {save_img_path}")

    def predict(self, image):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.cfg.img_size, self.cfg.img_size))
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.masking(img)
            print(outputs.size())
            _, out = self.inpaint(img, outputs)
            inpaint = img * (1 - outputs) + out * outputs
        masks = img * (1 - outputs) + outputs  # torch.cat([outputs, outputs, outputs], dim=1)

        dirname = os.path.dirname(image)
        weights_filename = os.path.basename(GAN_FACE_GEN_WEIGHT_PT).split('.')[0]

        filename, extension = os.path.splitext(os.path.basename(image))
        output_filename = filename + '_' + weights_filename + extension
        out_file_pt = os.path.join(dirname, output_filename)
        self.save_image([img, masks, inpaint], out_file_pt, nrow=3)

    def predict_mask_segm(self, image):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.cfg.img_size, self.cfg.img_size))
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.masking(img)
        masks = img * (1 - outputs) + outputs  # torch.cat([outputs, outputs, outputs], dim=1)

        dirname = os.path.dirname(image)
        filename, extension = os.path.splitext(os.path.basename(image))
        output_filename = filename + '_mask_segm' + extension
        out_file_pt = os.path.join(dirname, output_filename)
        self.save_image([img, masks], out_file_pt, nrow=2)

    def generate_face(self, image, mask):
        ''' NOFEVER VERSION!
            image = path to the same image that 'mask' binary sensor was made of.
            mask = torch.Tensor of mask binary map. Dimension = ([1, 1, image_width, image_height])
        '''
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img.shape[0], img.shape[1]))
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, out = self.inpaint(img, mask)
            inpaint = img * (1 - mask) + out * mask
        masks = img * (1 - mask) + mask  # torch.cat([outputs, outputs, outputs], dim=1)

        dirname = os.path.dirname(image)
        filename, extension = os.path.splitext(os.path.basename(image))
        output_filename = filename + '_GAN_generated' + extension
        out_file_pt = os.path.join(dirname, output_filename)
        self.save_image([img, masks, inpaint], out_file_pt, nrow=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training custom model')
    parser.add_argument('--image', default=None, type=str, help='Path to image for inference')
    parser.add_argument('--mode', default='gan', type=str, help='Which network model to run. "segm" for\
         Mask segmentation, "gan" for Segmentation+Face generation.')
    # parser.add_argument('config', default='config', type=str, help='config training')
    args = parser.parse_args()
    config = Config('./configs/facemask.yaml')
    model = Predictor(cfg=config)

    assert(args.image is not None), 'Provide path to a target image as "--image <path>".'

    if args.mode == 'segm':
        model.predict_mask_segm(args.image)
    elif args.mode == 'gan':
        model.predict(args.image)
    else:
        raise ValueError('Wrong mode selected. Choose "segm" for Mask segmentation\
            or "gan" for Segmentation+Face generation.')
