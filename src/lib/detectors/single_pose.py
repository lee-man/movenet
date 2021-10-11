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
from utils.post_process import multi_pose_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector


class SinglePoseDetector(BaseDetector):
    def __init__(self, opt):
        super(SinglePoseDetector, self).__init__(opt)
        self.flip_idx = opt.flip_idx

    def process(self, images, return_time=False):
        with torch.no_grad():
            # torch.cuda.synchronize()
            dets = self.model.decode(images)
            output = self.model(images)[-1]
            dets = self.model.decode(output)
            # torch.cuda.synchronize()
            forward_time = time.time()

        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets = multi_pose_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'])
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 39)
            # import pdb; pdb.set_trace()
            dets[0][j][:, :4] /= scale
            dets[0][j][:, 5:] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        results[1] = np.concatenate(
            [detection[1] for detection in detections], axis=0).astype(np.float32)
        if self.opt.nms or len(self.opt.test_scales) > 1:
            soft_nms_39(results[1], Nt=0.5, method=2)
        results[1] = results[1].tolist()
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        dets = dets.detach().cpu().numpy().copy()
        dets[:, :, :4] *= self.opt.down_ratio
        dets[:, :, 5:39] *= self.opt.down_ratio
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((
            img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        if self.opt.hm_hp:
            pred = debugger.gen_colormap_hp(
                output['hm_hp'][0].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hmhp')

    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='multi_pose')
        for bbox in results[1]:
            if bbox[4] > self.opt.vis_thresh:
                # mli: not draw the bounding box.
                # debugger.add_coco_bbox(bbox[:4], 0, bbox[4], img_id='multi_pose')
                debugger.add_coco_hp(bbox[5:39], img_id='multi_pose')
        debugger.show_all_imgs(pause=self.pause)
