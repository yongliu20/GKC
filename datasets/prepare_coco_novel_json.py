import argparse
import os
import os.path as osp
import shutil
from functools import partial
from glob import glob

import mmcv
import numpy as np
from PIL import Image

import json
from pycocotools.coco import COCO
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata, COCO_CATEGORIES

COCO_LEN = 123287

novel_clsID = [2, 33, 21, 25, 95, 76, 53, 57, 70, 145, 87, 46, 13, 100, 34, 41, 175, 148]
base_clsID = [k["id"] for k in COCO_CATEGORIES if k["id"] not in novel_clsID]
# NOVEL_CATEGORIES = [k for k in COCO_CATEGORIES if k["id"] in novel_clsID]
# BASE_CATEGORIES = [k for k in COCO_CATEGORIES if k["id"] not in novel_clsID]

# novel_clsID2trainID = {}
# for i, cat in enumerate(NOVEL_CATEGORIES):
#         novel_clsID2trainID[cat["id"]] = i

# base_clsID2trainID = {}
# for i, cat in enumerate(BASE_CATEGORIES):
#         base_clsID2trainID[cat["id"]] = i

# novel_clsID = [20, 24, 32, 33, 40, 56, 86, 99, 105, 123, 144, 147, 148, 168, 171]
# base_clsID = [k for k in full_clsID_to_trID.keys() if k not in novel_clsID + [255]]
# novel_clsID_to_trID = {k: i for i, k in enumerate(novel_clsID)}
# base_clsID_to_trID = {k: i for i, k in enumerate(base_clsID)}

def convert_ignore_json(json_path, out_json_path, keepID, remain_img):
    with open(json_path) as f:
        coco_json = json.load(f)
    # target_json = {}
    annotations = []
    keep_count = 0
    ann_len = len(coco_json["annotations"])
    for i, ann in enumerate(coco_json["annotations"]):
        new_ann = {'file_name' : ann['file_name'], 
            'image_id' : ann['image_id']}
        seg_info = []
        has_keep, has_remove = False, False
        for seg in ann["segments_info"]:
            if seg['category_id'] in keepID:
                seg_info.append(seg)
                has_keep = True
            else:
                has_remove = True
        if has_keep and (remain_img or not has_remove):
            new_ann['segments_info'] = seg_info
            annotations.append(new_ann)
            keep_count += 1
        print("[{}]/[{}], image count: {}".format(i, ann_len, keep_count), end='\r')

    with open(out_json_path, "w") as f:
        json.dump({"annotations" : annotations}, f)


if __name__ == "__main__":
    convert_ignore_json("data/coco/annotations/panoptic_train2017.json", 
        "data/coco/annotations/panoptic_train2017_base_ignore.json", base_clsID, True)
    print("Train base ignore DONE.")
    convert_ignore_json("data/coco/annotations/panoptic_train2017.json", 
        "data/coco/annotations/panoptic_train2017_base_delete.json", base_clsID, False)
    print("Train base delete DONE.")
    convert_ignore_json("data/coco/annotations/panoptic_val2017.json", 
        "data/coco/annotations/panoptic_val2017_base.json", base_clsID, True)
    print("Val base DONE.")
    convert_ignore_json("data/coco/annotations/panoptic_val2017.json", 
        "data/coco/annotations/panoptic_val2017_novel.json", novel_clsID, True)
    print("Val novel DONE.")