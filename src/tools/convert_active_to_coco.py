from PIL import Image
import os
import os.path as osp
import numpy as np
import json

db_type = 'train' # train, test
annot_path = "active.json"
save_path = "active_coco.json"

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
joint_num = 15
img_num = len(active)
print("image size: ", img_num)

aid = 100000
coco = {'images': [], 'categories': [], 'annotations': []}
for img_id in range(img_num):
    
    filename = 'images/' + str(active[img_id]['image'])#filename
    img = Image.open(osp.join('..', filename))
    w,h = img.size
    img_dict = {'id': aid, 'file_name': str(active[img_id]['image']), 'width': w, 'height': h}
    coco['images'].append(img_dict)
    
    bbox = np.zeros((4)) # xmin, ymin, w, h
    kps = np.zeros((joint_num, 3)) # xcoord, ycoord, vis
    #kps
    for jid in range(16):
        if (joint_mapping[str(jid)] == -1): continue
        kps[jid][0] = active[img_id]["joints"][jid][0]
        kps[jid][1] = active[img_id]["joints"][jid][1]
        kps[jid][2] = active[img_id]["joint_vis"][jid] + 1
    kps[1:5] = np.zeros((4, 3))


    #bbox extract from annotated kps
    annot_kps = kps[kps[:,2]==1,:].reshape(-1,3)
    xmin = np.min(annot_kps[:,0])
    ymin = np.min(annot_kps[:,1])
    xmax = np.max(annot_kps[:,0])
    ymax = np.max(annot_kps[:,1])
    width = xmax - xmin - 1
    height = ymax - ymin - 1
    
    # corrupted bounding box
    if width <= 0 or height <= 0:
        continue
    # 20% extend    
    else:
        bbox[0] = (xmin + xmax)/2. - width/2*1.2
        bbox[1] = (ymin + ymax)/2. - height/2*1.2
        bbox[2] = width*1.2
        bbox[3] = height*1.2
    person_dict = {'id': aid, 'image_id': aid, 'category_id': 1, 'area': bbox[2]*bbox[3],'bbox':bbox.tolist(), 'iscrowd': 0, 'keypoints': kps.reshape(-1).tolist(), 'num_keypoints':int(np.su(kps[:,2]==1))}
    coco['annotations'].append(person_dict)
    aid += 1

category = {
    "supercategory": "person",
    "id": 1,  # to be same as COCO, not using 0
    "name": "person",
    "skeleton": [[0,1],
        [1,2], 
        [2,6], 
        [7,12], 
        [12,11], 
        [11,10], 
        [5,4], 
        [4,3], 
        [3,6], 
        [7,13], 
        [13,14], 
        [14,15], 
        [6,7], 
        [7,8], 
        [8,9]] ,
    "keypoints": ["r_ankle", "r_knee","r_hip", 
                    "l_hip", "l_knee", "l_ankle",
                  "pelvis", "throax",
                  "upper_neck", "head_top",
                  "r_wrist", "r_elbow", "r_shoulder",
                  "l_shoulder", "l_elbow", "l_wrist"]}

coco['categories'] = [category]

with open(save_path, 'w') as f:
    json.dump(coco, f)