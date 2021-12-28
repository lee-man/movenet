from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import ipdb
import os
import sys
sys.path.append("..")

from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math
from utils.debugger import Debugger

PART_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]

NUM_KEYPOINTS = len(PART_NAMES)

PART_IDS = {pn: pid for pid, pn in enumerate(PART_NAMES)}
#


CONNECTED_PART_NAMES = [
    ("leftHip", "leftShoulder"), ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("rightHip", "rightShoulder"),
    ("rightElbow", "rightShoulder"), ("rightElbow", "rightWrist"),
    ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"), ("leftHip", "rightHip")
]

CONNECTED_PART_INDICES = [(PART_IDS[a], PART_IDS[b]) for a, b in CONNECTED_PART_NAMES]


class SinglePoseDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)

        keypoint_coords = np.array(anns[0]['keypoints']).reshape(17, 3)
        results = []
        adjacent_keypoints = []
        for left, right in CONNECTED_PART_INDICES:
            results.append(
                np.array([keypoint_coords[left][0:2],
                          keypoint_coords[right][0:2]]).astype(np.int32),
            )
        adjacent_keypoints.extend(results)
        cv_keypoints = []
        for kc in keypoint_coords:
            cv_keypoints.append(cv2.KeyPoint(kc[0], kc[1], 5))  ###################################################
        M = [0]
        L = [1, 3, 5, 7, 9, 11, 13, 15]
        R = [2, 4, 6, 8, 10, 12, 14, 16]
        out_img = cv2.drawKeypoints(
            img, [cv_keypoints[i] for i in M], outImage=np.array([]), color=(255, 255, 255),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        out_img = cv2.drawKeypoints(
            out_img, [cv_keypoints[i] for i in L], outImage=np.array([]), color=(255, 0, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        out_img = cv2.drawKeypoints(
            out_img, [cv_keypoints[i] for i in R], outImage=np.array([]), color=(0, 0, 255),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        out_img = cv2.polylines(out_img, adjacent_keypoints[0:5], isClosed=False, color=(255, 0, 0))
        out_img = cv2.polylines(out_img, adjacent_keypoints[5:10], isClosed=False, color=(0, 0, 255))
        out_img = cv2.polylines(out_img, adjacent_keypoints[10:12], isClosed=False, color=(0, 255, 0))
        # out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
        box = anns[0]['bbox']
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.int32)
        out_img = cv2.rectangle(
            out_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.imshow('img', out_img)

        cv2.waitKey()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        rot = 0

        flipped = False
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(
                    low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(
                    low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.aug_rot:
                rf = self.opt.rotate
                rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)

            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, rot, [self.opt.input_res, self.opt.input_res])
        inp = cv2.warpAffine(img, trans_input,
                             (self.opt.input_res, self.opt.input_res),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 127.5)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_res = self.opt.output_res
        num_joints = self.num_joints
        trans_output_rot = get_affine_transform(
            c, s, rot, [output_res, output_res])  # 经过下采样后,得到原图与下采样特征图之间的转换矩阵
        trans_output = get_affine_transform(c, s, 0, [output_res, output_res])

        hm = np.zeros((self.num_classes, output_res,
                       output_res), dtype=np.float32)  # 1, 64 64
        hm_hp = np.zeros((num_joints, output_res, output_res),
                         dtype=np.float32)  # 27 64 64
        kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)  # 2 54
        ind = np.zeros((self.max_objs), dtype=np.int64)
        kps_mask = np.zeros(
            (self.max_objs, self.num_joints * 2), dtype=np.uint8)
        hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)
        hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)
        hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        debugger = Debugger(
            dataset='pair', ipynb=False, theme='white')

        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(ann['category_id']) - 1
            pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3)

            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                pts[:, 0] = width - pts[:, 0] - 1
                for e in self.flip_idx:
                    pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox = np.clip(bbox, 0, output_res - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if (h > 0 and w > 0) or (rot != 0):
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = self.opt.hm_gauss if self.opt.mse_loss else max(
                    0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                ind[k] = ct_int[1] * output_res + ct_int[0]
                num_kpts = pts[:, 2].sum()
                if num_kpts == 0:
                    hm[cls_id, ct_int[1], ct_int[0]] = 0.9999

                hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                hp_radius = self.opt.hm_gauss \
                    if self.opt.mse_loss else max(0, int(hp_radius))
                for j in range(num_joints):
                    temp = pts[j]
                    if pts[j, 2] > 0:
                        pts[j, :2] = affine_transform(
                            pts[j, :2], trans_output_rot)
                        temp_1 = pts[j, :]

                        if pts[j, 0] >= 0 and pts[j, 0] < output_res and \
                                pts[j, 1] >= 0 and pts[j, 1] < output_res:
                            kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                            kps_mask[k, j * 2: j * 2 + 2] = 1
                            pt_int = pts[j, :2].astype(np.int32)
                            hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
                            hp_ind[k * num_joints + j] = pt_int[1] * \
                                                         output_res + pt_int[0]
                            hp_mask[k * num_joints + j] = 1
                            draw_gaussian(hm_hp[j], pt_int, hp_radius)
                            temp3 = hm_hp[j]
                            a = 0

                y = draw_gaussian(hm[cls_id], ct_int, radius)
                ipdb.set_trace()
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1] +
                              pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])
        if rot != 0:
            hm = hm * 0 + 0.9999
            # reg_mask *= 0
            kps_mask *= 0
        # ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
        #        'hps': kps, 'hps_mask': kps_mask}
        # ret = {'input': inp, 'hm': hm, 'ind': ind,
        #        'hps': kps, 'hps_mask': kps_mask}
        # if self.opt.hm_hp:
        #     ret.update({'hm_hp': hm_hp})
        # if self.opt.reg_hp_offset:
        #     ret.update({'hp_offset': hp_offset,
        #                'hp_ind': hp_ind, 'hp_mask': hp_mask})
        # new_img = img.resize((64, 64), Image.BILINEAR)

        ret = {'input': inp, 'hm': hm, 'ind': ind,
               'hps': kps, 'hps_mask': kps_mask,
               'hm_hp': hm_hp, 'hp_offset': hp_offset,
               'hp_ind': hp_ind, 'hp_mask': hp_mask}
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 40), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta
        return ret
