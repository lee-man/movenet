'''
Split `Active` dataset into training and test sets.
Move this file to `{$movenet}/data/active` and run it.
Author: Min LI

TODO: Check whether keypoint mapping from MPII to COCO is correct.
'''
from PIL import Image
import os
import os.path as osp
import numpy as np
import json
import shutil
import random

db_type = 'train' # train, test
train_percentage = 0.9
annot_path = "annotations/active.json"
train_save_path = "annotations/active_train.json"
val_save_path = "annotations/active_val.json"

if not osp.isdir('train'):
    os.makedirs('train')
if not osp.isdir('val'):
    os.makedirs('val')


print("Loading Acitve dataset...")
with open(annot_path) as json_file:
    active = json.load(json_file)
'''
MPII: 0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist

COCO_PERSON_KEYPOINT_NAMES = [
    'nose', 0
    'left_eye', 1
    'right_eye', 2
    'left_ear', 3
    'right_ear', 4
    'left_shoulder', 5
    'right_shoulder', 6
    'left_elbow', 7
    'right_elbow', 8
    'left_wrist', 9
    'right_wrist', 10
    'left_hip', 11
    'right_hip', 12
    'left_knee', 13
    'right_knee', 14
    'left_ankle', 15
    'right_ankle' 16
]
'''
joint_mapping = {'0': 16, '1': 14, '2': 12, '3': 11, '4': 13, '5': 15, '6': -1, '7': -1, '8': -1, '9': 0, '10': 10, '11': 8, '12': 6, '13': 5, '14': 7, '15': 9}
joint_num = 17
img_num = len(active)
random_index = list(range(img_num))
random.shuffle(random_index)
train_index = random_index[:int(img_num * train_percentage) + 1]
val_index = random_index[int(img_num * train_percentage) + 1:]

print("image size: ", img_num)
print("train size: ", int(img_num * train_percentage))
print("val size: ", img_num -int(img_num * train_percentage))

aid = 0
coco_train = {'images': [], 'categories': [], 'annotations': []}

for img_id in train_index:
    
    filename = 'images/' + str(active[img_id]['image'])#filename
    filename_target = 'train/' + str(active[img_id]['image'])
    shutil.copy(filename, filename_target)
    img = Image.open(osp.join('.', filename))
    w,h = img.size
    img_dict = {'id': aid, 'file_name': str(active[img_id]['image']), 'width': w, 'height': h}
    coco_train['images'].append(img_dict)
    
    bbox = np.zeros((4)) # xmin, ymin, w, h
    kps = np.zeros((joint_num, 3)) # xcoord, ycoord, vis
    ori_kps = []

    #kps
    for jid in range(16):
        if (joint_mapping[str(jid)] == -1): continue
        kps[joint_mapping[str(jid)]][0] = active[img_id]["joints"][jid][0]
        kps[joint_mapping[str(jid)]][1] = active[img_id]["joints"][jid][1]
        kps[joint_mapping[str(jid)]][2] = active[img_id]["joint_vis"][jid] + 1
        ori_kps.append([active[img_id]["joints"][jid][0],active[img_id]["joints"][jid][1]])
    kps[1:5] = np.zeros((4, 3))
    ori_kps = np.asarray(ori_kps)

    #bbox extract from annotated kps
    
    xmin = np.min(ori_kps[:,0])
    ymin = np.min(ori_kps[:,1])
    xmax = np.max(ori_kps[:,0])
    ymax = np.max(ori_kps[:,1])
    width = xmax - xmin - 1
    height = ymax - ymin - 1
    
    # corrupted bounding box
    if width <= 0 or height <= 0:
        continue
    # 20% extend    
    else:
        bbox[0] = ((xmin + xmax)/2. - width/2*1.2) if(((xmin + xmax)/2. - width/2*1.2)>0) else 0 
        bbox[1] = ((ymin + ymax)/2. - height/2*1.2) if(((ymin + ymax)/2. - height/2*1.2)>0) else 0
        bbox[2] = width*1.2 if ((bbox[0]+width*1.2)<w) else (w-bbox[0])
        bbox[3] = height*1.2 if ((bbox[1]+height*1.2)<w) else (h-bbox[1])

    person_dict = {'id': aid, 'image_id': aid, 'category_id': 1, 'area': bbox[2]*bbox[3],'bbox':bbox.tolist(), 'iscrowd': 0, 'keypoints': kps.reshape(-1).tolist(), 'num_keypoints':int(np.sum(kps[:,2]==2))}
    coco_train['annotations'].append(person_dict)
    aid += 1

category = {"supercategory": "person","id": 1,"name": "person","keypoints": ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"],"skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]}

coco_train['categories'] = [category]

with open(train_save_path, 'w') as f:
    json.dump(coco_train, f)

coco_val = {'images': [], 'categories': [], 'annotations': []}

for img_id in val_index:
    
    filename = 'images/' + str(active[img_id]['image'])#filename
    filename_target = 'val/' + str(active[img_id]['image'])
    shutil.copy(filename, filename_target)
    img = Image.open(osp.join('.', filename))
    w,h = img.size
    img_dict = {'id': aid, 'file_name': str(active[img_id]['image']), 'width': w, 'height': h}
    coco_val['images'].append(img_dict)
    
    bbox = np.zeros((4)) # xmin, ymin, w, h
    kps = np.zeros((joint_num, 3)) # xcoord, ycoord, vis
    ori_kps = []
    #kps
    for jid in range(16):
        if (joint_mapping[str(jid)] == -1): continue
        kps[joint_mapping[str(jid)]][0] = active[img_id]["joints"][jid][0]
        kps[joint_mapping[str(jid)]][1] = active[img_id]["joints"][jid][1]
        kps[joint_mapping[str(jid)]][2] = active[img_id]["joint_vis"][jid] + 1
        ori_kps.append([active[img_id]["joints"][jid][0],active[img_id]["joints"][jid][1]])
    kps[1:5] = np.zeros((4, 3))


    #bbox extract from annotated kps
    ori_kps = np.asarray(ori_kps)
    xmin = np.min(ori_kps[:,0])
    ymin = np.min(ori_kps[:,1])
    xmax = np.max(ori_kps[:,0])
    ymax = np.max(ori_kps[:,1])
    width = xmax - xmin - 1
    height = ymax - ymin - 1
    
    # corrupted bounding box
    if width <= 0 or height <= 0:
        continue
    # 20% extend    
    else:
        bbox[0] = ((xmin + xmax)/2. - width/2*1.2) if(((xmin + xmax)/2. - width/2*1.2)>0) else 0 
        bbox[1] = ((ymin + ymax)/2. - height/2*1.2) if(((ymin + ymax)/2. - height/2*1.2)>0) else 0
        bbox[2] = width*1.2 if ((bbox[0]+width*1.2)<w) else (w-bbox[0])
        bbox[3] = height*1.2 if ((bbox[1]+height*1.2)<w) else (h-bbox[1])

    person_dict = {'id': aid, 'image_id': aid, 'category_id': 1, 'area': bbox[2]*bbox[3],'bbox':bbox.tolist(), 'iscrowd': 0, 'keypoints': kps.reshape(-1).tolist(), 'num_keypoints':int(np.sum(kps[:,2]==2))}
    coco_val['annotations'].append(person_dict)
    aid += 1

category = {"supercategory": "person","id": 1,"name": "person","keypoints": ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"],"skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]}

coco_val['categories'] = [category]

with open(val_save_path, 'w') as f:
    json.dump(coco_val, f)