import os.path as osp
import sys
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, '../lib')
add_path(lib_path)
#from datasets.dataset.active import run_eval
def run_eval():
    coco1 = coco.COCO('/Users/rachel/PycharmProjects/movenet/data/active_pair/annotations/active_pair_val.json')
    coco_dets = coco1.loadRes('/Users/rachel/PycharmProjects/movenet/exp/single_pose/default/results.json')
    # some are better
    coco_eval = COCOeval(coco1, coco_dets, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

run_eval()
# process
# evaluate
#
# params = Params(iouType=iouType)
# self.setKpParams()
#