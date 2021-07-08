from PIL import Image
import os
import os.path as osp
import numpy as np
import json

coco_path = "person_keypoints_train2017_filtered.json"
active_path = "active_coco.json"

save_path = "person_keypoints_train2017_filtered_merged.json"

print("Loading Acitve and COCO dataset...")
with open(coco_path) as json_file:
    coco = json.load(json_file)
with open(active_path) as json_file:
    active = json.load(json_file)

images = []
for coco_image in coco['images']:
  images.append(coco_image)
for active_image in active['images']:
  images.append(active_image)
annotations = []
for coco_annotation in coco['annotations']:
  annotations.append(coco_annotation)
for active_annotation in active['annotations']:
  annotations.append(active_annotation)
new_master_json = {
            'info': coco['info'],
            'licenses': coco['licenses'],
            'images': images,
            'annotations': annotations,
            'categories': coco['categories']
        }

img_num = len(images)
print("image size: ", img_num)
annotations_num = len(annotations)
print("annotation size: ", annotations_num)

with open(save_path, 'w') as f:
    json.dump(new_master_json, f)