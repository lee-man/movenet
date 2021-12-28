import json
import numpy as np
import math
import xlwt
p1 = '/Users/rachel/PycharmProjects/movenet/data/failure_case/annotations/failure_case_val.json'
p2 = '/Users/rachel/PycharmProjects/movenet/exp/single_pose/1213add_aist_aug_0.6_90_0.005/top_finetune_results.json'
p3 = '/Users/rachel/PycharmProjects/movenet/exp/single_pose/1213add_aist_aug_0.6_90_0.005/top2_finetune_results.json'
file_path = '/Users/rachel/PycharmProjects/movenet/experiments/1220test/test1.csv'
with open(p1) as f1:
    gt = json.load(f1)
with open(p2) as f2:
    predori = json.load(f2)
with open(p3) as f3:
    prednew = json.load(f3)

#def calc_ori_distance():

def calc_distance(pred):
    dislist = []
    for i in range(len(gt['images'])):
        cur_dict = {}
        kpt = gt['annotations'][i]['keypoints']
        id = gt['images'][i]['id']
        for j in pred:
            if(j['image_id'] == id):
                pred_kpt = j['keypoints']
        pred_kpt = np.array(pred_kpt).reshape(-1,3)
        kpt = np.array(kpt).reshape(-1,3)[5:]
        #dis = 1000
        totdis1 = 0
        totdis2 = 0
        ind = 0
        for k in range(len(kpt)):
            dis = 1000
            a = kpt[k]
            b = pred_kpt[k+5]
            cur_dis = math.sqrt((a[0] - b[1]) * (a[0] - b[1]) \
                                    + (a[1] - b[0]) * (a[1] - b[0]))
            totdis1+= cur_dis
        totdis1/=12
        cur_dict['image_id'] = id
        cur_dict['distance'] = totdis1
        cur_dict['file_name'] = gt['images'][i]['file_name']
        dislist.append(cur_dict)
    return dislist

def calc_distance1(pred):
    dislist = []
    for i in range(len(gt['images'])):
        cur_dict = {}
        kpt = gt['annotations'][i]['keypoints']
        id = gt['images'][i]['id']
        for j in pred:
            if(j['image_id'] == id):
                pred_kpt = j['keypoints']
        pred_kpt = np.array(pred_kpt).reshape(-1,3)
        kpt = np.array(kpt).reshape(-1,3)[5:]
        dis = 1000
        totdis1 = 0
        ind = 0
        for k in range(len(kpt)):
            a = kpt[k]
            # index: ori 0-16; 5 - 0 - 10
            for b in pred_kpt[10:]:
                cur_dis = math.sqrt((a[0] - b[1]) * (a[0] - b[1]) \
                                     + (a[1] - b[0]) * (a[1] - b[0]))
                if(cur_dis<dis):
                    dis = cur_dis
            '''b1 = pred_kpt[k + 5]
            b2 = pred_kpt[k + 5+17]
            cur_dis1 = math.sqrt((a[0] - b1[1]) * (a[0] - b1[1]) \
                                + (a[1] - b1[0]) * (a[1] - b1[0]))
            cur_dis2 = math.sqrt((a[0] - b2[1]) * (a[0] - b2[1]) \
                                 + (a[1] - b2[0]) * (a[1] - b2[]))'''
            totdis1+= dis
            ind+=1
        totdis1/=12
        cur_dict['image_id'] = id
        cur_dict['distance'] = totdis1
        cur_dict['file_name'] = gt['images'][i]['file_name']
        dislist.append(cur_dict)
    return dislist
dislist1 = calc_distance(predori)
dislist2 = calc_distance1(prednew)

f = xlwt.Workbook()
sheet1 = f.add_sheet(u'sheet1',cell_overwrite_ok=True) #创建sheet
a = 0
row1 = []
sheet1.write(0,1,'image_name')
sheet1.write(0,2,'top1_mean')
sheet1.write(0,2,'top2_mean')

for b in range(len(dislist1)):
    data1 = dislist1[b]
    data2 = dislist2[b]

    #filename =

    sheet1.write(b+1,0,data1['file_name'])
    sheet1.write(b + 1, 1, data1['distance'])
    sheet1.write(b + 1, 2, data2['distance'])


f.save(file_path)
print(1)