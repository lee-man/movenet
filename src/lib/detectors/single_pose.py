from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

try:
    from external.nms import soft_nms_39
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import single_pose_decode
from models.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform
from utils.post_process import multi_pose_post_process, single_pose_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector


class SinglePoseDetector(BaseDetector):
    def __init__(self, opt):
        super(SinglePoseDetector, self).__init__(opt)
        self.flip_idx = opt.flip_idx
        self.vis_thresh = opt.vis_thresh

    def process(self, images,name, return_time=False):
        with torch.no_grad():
            # torch.cuda.synchronize()
            # images(3,256,256)
            # output: dict:4 {hm,hos,hm_hp,hp_offset}
            # images = (1,3,256,256)
            output = self.model(images)[0]
            dets = self.model.decode(output,name)
            # torch.cuda.synchronize()
            forward_time = time.time()

        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta):
        dets = dets[0, 0, :, :]
        dets = dets.cpu().numpy()
        dets = single_pose_post_process(
            dets.copy(),
            meta['in_height'], meta['in_width'])
        return dets

    def merge_outputs(self, detections):
        results = {}
        results[1] = np.concatenate(
            [detection[1] for detection in detections], axis=0).astype(np.float32)
        if self.opt.nms or len(self.opt.test_scales) > 1:
            soft_nms_39(results[1], Nt=0.5, method=2)
        results[1] = results[1].tolist()
        return results

    def debug(self, debugger, images, dets, output):
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((
            img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(torch.sigmoid(output['hm'][0]).detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        # torch.sigmoid(output['hm_hp'][0][13].resize(1,64,64)).detach().cpu().numpy())
        pred = debugger.gen_colormap_hp(
            torch.sigmoid(output['hm_hp'][0]).detach().cpu().numpy())
        #pred = debugger.gen_colormap_hp(
        #    torch.sigmoid(output['hm_hp'][0][15]).detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hmhp')
    def show_34_results(self, debugger, img_name, image, results):
        # image is ori image
        debugger.add_img(image, img_id='single_pose')
        debugger.add_coco_hp(results, img_id='single_pose', vis_thresh=self.vis_thresh)
        debugger.show_all_imgs(img_name, pause=self.pause)

    def show_results(self, debugger, img_name, image, results):
        # image is ori image
        debugger.add_img(image, img_id='single_pose')
        debugger.add_coco_hp(results, img_id='single_pose', vis_thresh=self.vis_thresh)
        debugger.show_all_imgs(img_name,pause=self.pause)
