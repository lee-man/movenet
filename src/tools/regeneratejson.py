import json
import os
p1 = '/Users/rachel/PycharmProjects/movenet/data/failure_case/annotations/add_aist_val.json'
p2 = '/Users/rachel/PycharmProjects/movenet/data/failure_case/val'
p3 = '/Users/rachel/PycharmProjects/movenet/data/failure_case/annotations/failure_case_val.json'

filelist = []
def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield f

for i in findAllFile(p2):
    filelist.append(i)
with open(p1) as f1:
    files = json.load(f1)
print(1)
images = []
annotations = []
categories = [{'supercategory': 'person', 'id': 1, 'name': 'person', 'keypoints': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'], 'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]}]
for i in range(len(files['images'])):
    if(files['images'][i]['file_name'] in filelist):
        annotations.append(files['annotations'][i])
        images.append(files['images'][i])
new_anno = {}
new_anno['images'] = images
new_anno['annotations'] = annotations
new_anno['categories'] = categories
with open(p3, 'w') as f:
    json.dump(new_anno, f)
#for i in filelist:


