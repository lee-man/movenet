
import json
from shutil import copyfile
file1 = '../../data/LiXi_Pair/pair/pair_train.json'
with open(file1) as json_file1:
    pair_train = json.load(json_file1)

file2 = '../../data/LiXi_Pair/pair/pair_val.json'
with open(file2) as json_file2:
    pair_val = json.load(json_file2)

file3 = '../../data/active/annotations/active_train.json'
with open(file3) as json_file3:
    active_train = json.load(json_file3)

file4 = '../../data/active/annotations/active_val.json'
with open(file4) as json_file4:
    active_val = json.load(json_file4)

new_images = active_train['images']+pair_train['images']
new_annotations = active_train['annotations'] + pair_train['annotations']
new_categories = active_train['categories']
combined_anno = {}
combined_anno['images'] = new_images
combined_anno['categories'] = new_categories
combined_anno['annotations'] = new_annotations

with open('../../data/combined_active_pair_train.json','w') as f:
    json.dump(combined_anno,f)

with open('../../data/combined_active_pair_train.json') as f:
    combined_val = json.load(f)
filenames = [i['file_name'] for i in combined_val['images']]
print(len(filenames))

'''for i in range(len(pair_val['images'])):
    name = pair_val['images'][i]['file_name']
    new_name = name.split('_')[1]+'.jpg'
    pair_val['images'][i]['file_name'] = new_name'''







'''file2 = '../../data/active/annotations/active_train.json'
with open(file2) as json_file2:
    active = json.load(json_file2)

file3 = '../../data/coco/person_keypoints_val2017.json'
with open(file3) as json_file3:
    coco = json.load(json_file3)
print('1')'''



