 1/1: from nltk.corpus import wordnet as wn
 1/2: wn.synsets('dog')
 1/3: nltk.download('wordnet')
 1/4: import nltk
 1/5: nltk.download('wordnet')
 1/6: wn.synsets('dog')
 1/7: nltk.download('omw-1.4')
 1/8: wn.synsets('dog')
 1/9:
def get_syn(word):
    syn = []
    for s in wn.synsets(word):
        for l in s.lemmas():
            syn.appned(l.name())
1/10:
def get_syn(word):
    syn = []
    for s in wn.synsets(word):
        for l in s.lemmas():
            syn.appned(l.name())
    return syn
1/11: print(get_syn('person'))
1/12:
def get_syn(word):
    syn = []
    for s in wn.synsets(word):
        for l in s.lemmas():
            syn.append(l.name())
    return syn
1/13: print(get_syn('person'))
1/14:
def get_syn(word):
    syn = set()
    for s in wn.synsets(word):
        for l in s.lemmas():
            syn.add(l.name())
    return list(syn)
1/15: print(get_syn('person'))
1/16: from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES, CITYSCAPES_CATEGORIES, _get_builtin_metadata
1/17: coco_meta = _get_builtin_metadata("coco_panoptic_standard")
1/18: coco_name = []
1/19:
for coco_log in COCO_CATEGORIES:
    coco_name.append(coco_log['name'])
1/20: coco_syn = {}
1/21:
for name in coco_name:
    coco_syn[name] = get_syn(name)
1/22: coco_syn
1/23: import json
1/24:
with open("coco_name_syn.json", "w") as f:
    json.dump(coco_syn, f)
1/25:
def get_syn(word):
    syn = set()
    for s in wn.synsets(word, pos=wn.NOUN):
        for l in s.lemmas():
            syn.add(l.name())
    return list(syn)
1/26: coco_syn = {}
1/27:
for name in coco_name:
    coco_syn[name] = get_syn(name)
1/28:
with open("coco_name_syn.json", "w") as f:
    json.dump(coco_syn, f)
1/29: wn.synsets("wall-other-merged")
1/30: wn.synsets("wall_other_merged")
1/31: wn.synsets("wall_other")
1/32: wn.synsets("wall")
1/33: from ...modeling.utils.misc import process_coco_cat
1/34:
def process_coco_cat(cls_name):
    if isinstance(cls_name, str):
        cls_name = [cls_name]
    cls_name = [n.replace("-merged", "") for n in cls_name]
    cls_name = [n.replace("-other", "") for n in cls_name]
    cls_name = [n.replace("-", " ") for n in cls_name]

    return cls_name
1/35: wn.synsets("wall_brick")
1/36: wn.synsets("wall-brick")
1/37: wn.synsets("wall brick")
1/38: wn.synsets("brick")
1/39: history
1/40: coco_name_dic = {}
1/41:
for coco_log in COCO_CATEGORIES:
    name = coco_log['name']
    coco_name_dic[name] = process_coco_cat(name)
1/42:
for name in coco_name:
    coco_syn[name] = get_syn(coco_name_dic[name])
1/43:
for name in coco_name:
    coco_syn[name] = get_syn(coco_name_dic[name][0])
1/44: coco_syn
1/45: coco_name_dic
1/46: wn.synsets("wood_floor")
1/47: wn.synsets("wooden_floor")
1/48: wn.synsets("wood floor")
1/49:
with open("coco_name_syn.json", "w") as f:
    json.dump(coco_syn, f)
1/50: wn.synsets("boat")
1/51: wn.synsets("ball")
1/52: wn.synsets("ball")[0].lemma()
1/53: wn.synsets("ball")[0].lemmas()
1/54: wn.synsets("door")[0].lemmas()
1/55: wn.synsets("door")
1/56: pip install git+https://github.com/openai/CLIP.git
1/57: import clip
1/58: device = "cuda" if torch.cuda.is_available() else "cpu"
1/59: import torch
1/60: device = "cuda" if torch.cuda.is_available() else "cpu"
1/61: %load mask2former/data/datasets/register_coco_panopic_caption.py
1/62: !ls
1/63: %load code/ky_open_voca/mask2former/data/datasets/register_coco_panopic_caption.py
1/64:
# %load code/ky_open_voca/mask2former/data/datasets/register_coco_panopic_caption.py
import copy
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

from detectron2.data.datasets.builtin_meta import _get_builtin_metadata, COCO_CATEGORIES

# from .coco import load_coco_json, load_sem_seg

def load_coco_panoptic_caption_json(json_file, caption_file, image_dir, gt_dir, semseg_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    from pycocotools.coco import COCO
    caption_json = COCO(caption_file)

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = int(ann["image_id"])
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir, ann["file_name"])
        sem_label_file = os.path.join(semseg_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        caption = caption_json.imgToAnns[image_id]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
                "sem_seg_file_name": sem_label_file,
                "caption": caption,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    return ret


def register_coco_panoptic(
    name, metadata, image_root, panoptic_root, panoptic_json, 
    caption_json, instances_json=None, sem_seg_root=None,
):
    """
    Register a "standard" version of COCO panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".

    Args:
        name (str): the name that identifies a dataset,
            e.g. "coco_2017_train_panoptic"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images in COCO format
        panoptic_json (str): path to the json panoptic annotation file in COCO format
        sem_seg_root (none): not used, to be consistent with
            `register_coco_panoptic_separated`.
        instances_json (str): path to the json instance annotation file
    """
    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_coco_panoptic_caption_json(panoptic_json, caption_json, image_root, panoptic_root, sem_seg_root, metadata),
    )
    MetadataCatalog.get(panoptic_name).set(
        sem_seg_root=sem_seg_root,
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )

_PREDEFINED_SPLITS_COCO_PANOPTIC_CAPTION = {
    "coco_2017_train_panoptic_caption": (
        "coco/train2017",
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        "coco/annotations/captions_train2017.json",
        "coco/annotations/instances_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_semseg_train2017",
    ),
    "coco_2017_val_panoptic_caption": (
        "coco/val2017",
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/annotations/captions_val2017.json",
        "coco/annotations/instances_val2017.json",
        "coco/panoptic_semseg_val2017",
    ),
    # "coco_2017_val_100_panoptic": (
    #     "coco/panoptic_val2017_100",
    #     "coco/annotations/panoptic_val2017_100.json",
    #     "coco/panoptic_stuff_val2017_100",
    # ),
}


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def register_all_coco(root):
    for (
        prefix,
        (image_root, panoptic_root, panoptic_json, caption_json, instances_json, stuff_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC_CAPTION.items():
        # print(image_root, panoptic_root, panoptic_json, caption_json, instances_json)
        # prefix_instances = prefix[: -len("_panoptic")]
        # instances_meta = MetadataCatalog.get(prefix_instances)
        # image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_coco_panoptic(
            prefix,
            # _get_builtin_metadata("coco_panoptic_standard"),
            get_metadata(),
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, caption_json),
            os.path.join(root, instances_json),
            os.path.join(root, stuff_root),
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco(_root)
1/65: type(DatasetCatalog.get("coco_2017_train_panoptic_caption"))
1/66: !export DETECTRON2_DATASETS="/opt/tiger/debug/code/ky_open_voca/data"
1/67: type(DatasetCatalog.get("coco_2017_train_panoptic_caption"))
1/68: !'export DETECTRON2_DATASETS="/opt/tiger/debug/code/ky_open_voca/data"'
1/69: export DETECTRON2_DATASETS="/opt/tiger/debug/code/ky_open_voca/data"
1/70: !export DETECTRON2_DATASETS="/opt/tiger/debug/code/ky_open_voca/data"
1/71: %load code/ky_open_voca/mask2former/data/datasets/register_coco_panopic_caption.py
1/72:
# %load code/ky_open_voca/mask2former/data/datasets/register_coco_panopic_caption.py
import copy
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

from detectron2.data.datasets.builtin_meta import _get_builtin_metadata, COCO_CATEGORIES

# from .coco import load_coco_json, load_sem_seg

def load_coco_panoptic_caption_json(json_file, caption_file, image_dir, gt_dir, semseg_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    from pycocotools.coco import COCO
    caption_json = COCO(caption_file)

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = int(ann["image_id"])
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir, ann["file_name"])
        sem_label_file = os.path.join(semseg_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        caption = caption_json.imgToAnns[image_id]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
                "sem_seg_file_name": sem_label_file,
                "caption": caption,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    return ret


def register_coco_panoptic(
    name, metadata, image_root, panoptic_root, panoptic_json, 
    caption_json, instances_json=None, sem_seg_root=None,
):
    """
    Register a "standard" version of COCO panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".

    Args:
        name (str): the name that identifies a dataset,
            e.g. "coco_2017_train_panoptic"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images in COCO format
        panoptic_json (str): path to the json panoptic annotation file in COCO format
        sem_seg_root (none): not used, to be consistent with
            `register_coco_panoptic_separated`.
        instances_json (str): path to the json instance annotation file
    """
    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_coco_panoptic_caption_json(panoptic_json, caption_json, image_root, panoptic_root, sem_seg_root, metadata),
    )
    MetadataCatalog.get(panoptic_name).set(
        sem_seg_root=sem_seg_root,
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )

_PREDEFINED_SPLITS_COCO_PANOPTIC_CAPTION = {
    "coco_2017_train_panoptic_caption": (
        "coco/train2017",
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        "coco/annotations/captions_train2017.json",
        "coco/annotations/instances_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_semseg_train2017",
    ),
    "coco_2017_val_panoptic_caption": (
        "coco/val2017",
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/annotations/captions_val2017.json",
        "coco/annotations/instances_val2017.json",
        "coco/panoptic_semseg_val2017",
    ),
    # "coco_2017_val_100_panoptic": (
    #     "coco/panoptic_val2017_100",
    #     "coco/annotations/panoptic_val2017_100.json",
    #     "coco/panoptic_stuff_val2017_100",
    # ),
}


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def register_all_coco(root):
    for (
        prefix,
        (image_root, panoptic_root, panoptic_json, caption_json, instances_json, stuff_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC_CAPTION.items():
        # print(image_root, panoptic_root, panoptic_json, caption_json, instances_json)
        # prefix_instances = prefix[: -len("_panoptic")]
        # instances_meta = MetadataCatalog.get(prefix_instances)
        # image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_coco_panoptic(
            prefix,
            # _get_builtin_metadata("coco_panoptic_standard"),
            get_metadata(),
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, caption_json),
            os.path.join(root, instances_json),
            os.path.join(root, stuff_root),
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco(_root)
1/73: DatasetCatalog.remove('coco_2017_train_panoptic_caption')
1/74:
# %load code/ky_open_voca/mask2former/data/datasets/register_coco_panopic_caption.py
import copy
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

from detectron2.data.datasets.builtin_meta import _get_builtin_metadata, COCO_CATEGORIES

# from .coco import load_coco_json, load_sem_seg

def load_coco_panoptic_caption_json(json_file, caption_file, image_dir, gt_dir, semseg_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    from pycocotools.coco import COCO
    caption_json = COCO(caption_file)

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = int(ann["image_id"])
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir, ann["file_name"])
        sem_label_file = os.path.join(semseg_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        caption = caption_json.imgToAnns[image_id]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
                "sem_seg_file_name": sem_label_file,
                "caption": caption,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    return ret


def register_coco_panoptic(
    name, metadata, image_root, panoptic_root, panoptic_json, 
    caption_json, instances_json=None, sem_seg_root=None,
):
    """
    Register a "standard" version of COCO panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".

    Args:
        name (str): the name that identifies a dataset,
            e.g. "coco_2017_train_panoptic"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images in COCO format
        panoptic_json (str): path to the json panoptic annotation file in COCO format
        sem_seg_root (none): not used, to be consistent with
            `register_coco_panoptic_separated`.
        instances_json (str): path to the json instance annotation file
    """
    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_coco_panoptic_caption_json(panoptic_json, caption_json, image_root, panoptic_root, sem_seg_root, metadata),
    )
    MetadataCatalog.get(panoptic_name).set(
        sem_seg_root=sem_seg_root,
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )

_PREDEFINED_SPLITS_COCO_PANOPTIC_CAPTION = {
    "coco_2017_train_panoptic_caption": (
        "coco/train2017",
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        "coco/annotations/captions_train2017.json",
        "coco/annotations/instances_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_semseg_train2017",
    ),
    "coco_2017_val_panoptic_caption": (
        "coco/val2017",
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/annotations/captions_val2017.json",
        "coco/annotations/instances_val2017.json",
        "coco/panoptic_semseg_val2017",
    ),
    # "coco_2017_val_100_panoptic": (
    #     "coco/panoptic_val2017_100",
    #     "coco/annotations/panoptic_val2017_100.json",
    #     "coco/panoptic_stuff_val2017_100",
    # ),
}


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def register_all_coco(root):
    for (
        prefix,
        (image_root, panoptic_root, panoptic_json, caption_json, instances_json, stuff_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC_CAPTION.items():
        # print(image_root, panoptic_root, panoptic_json, caption_json, instances_json)
        # prefix_instances = prefix[: -len("_panoptic")]
        # instances_meta = MetadataCatalog.get(prefix_instances)
        # image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_coco_panoptic(
            prefix,
            # _get_builtin_metadata("coco_panoptic_standard"),
            get_metadata(),
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, caption_json),
            os.path.join(root, instances_json),
            os.path.join(root, stuff_root),
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco(_root)
1/75: type(DatasetCatalog.get('coco_2017_train_panoptic_caption'))
1/76: !echo DETECTRON2_DATASETS
1/77: DatasetCatalog.remove('coco_2017_train_panoptic_caption')
1/78:
# %load code/ky_open_voca/mask2former/data/datasets/register_coco_panopic_caption.py
import copy
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

from detectron2.data.datasets.builtin_meta import _get_builtin_metadata, COCO_CATEGORIES

# from .coco import load_coco_json, load_sem_seg

def load_coco_panoptic_caption_json(json_file, caption_file, image_dir, gt_dir, semseg_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    from pycocotools.coco import COCO
    caption_json = COCO(caption_file)

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = int(ann["image_id"])
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir, ann["file_name"])
        sem_label_file = os.path.join(semseg_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        caption = caption_json.imgToAnns[image_id]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
                "sem_seg_file_name": sem_label_file,
                "caption": caption,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    return ret


def register_coco_panoptic(
    name, metadata, image_root, panoptic_root, panoptic_json, 
    caption_json, instances_json=None, sem_seg_root=None,
):
    """
    Register a "standard" version of COCO panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".

    Args:
        name (str): the name that identifies a dataset,
            e.g. "coco_2017_train_panoptic"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images in COCO format
        panoptic_json (str): path to the json panoptic annotation file in COCO format
        sem_seg_root (none): not used, to be consistent with
            `register_coco_panoptic_separated`.
        instances_json (str): path to the json instance annotation file
    """
    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_coco_panoptic_caption_json(panoptic_json, caption_json, image_root, panoptic_root, sem_seg_root, metadata),
    )
    MetadataCatalog.get(panoptic_name).set(
        sem_seg_root=sem_seg_root,
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )

_PREDEFINED_SPLITS_COCO_PANOPTIC_CAPTION = {
    "coco_2017_train_panoptic_caption": (
        "coco/train2017",
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        "coco/annotations/captions_train2017.json",
        "coco/annotations/instances_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_semseg_train2017",
    ),
    "coco_2017_val_panoptic_caption": (
        "coco/val2017",
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/annotations/captions_val2017.json",
        "coco/annotations/instances_val2017.json",
        "coco/panoptic_semseg_val2017",
    ),
    # "coco_2017_val_100_panoptic": (
    #     "coco/panoptic_val2017_100",
    #     "coco/annotations/panoptic_val2017_100.json",
    #     "coco/panoptic_stuff_val2017_100",
    # ),
}


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def register_all_coco(root):
    for (
        prefix,
        (image_root, panoptic_root, panoptic_json, caption_json, instances_json, stuff_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC_CAPTION.items():
        # print(image_root, panoptic_root, panoptic_json, caption_json, instances_json)
        # prefix_instances = prefix[: -len("_panoptic")]
        # instances_meta = MetadataCatalog.get(prefix_instances)
        # image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_coco_panoptic(
            prefix,
            # _get_builtin_metadata("coco_panoptic_standard"),
            get_metadata(),
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, caption_json),
            os.path.join(root, instances_json),
            os.path.join(root, stuff_root),
        )

_root = "/opt/tiger/debug/code/ky_open_voca/data"
register_all_coco(_root)
1/79: DatasetCatalog.remove("coco_2017_train_panoptic_caption")
1/80: MetadataCatalog.remove("coco_2017_train_panoptic_caption")
1/81:
# %load code/ky_open_voca/mask2former/data/datasets/register_coco_panopic_caption.py
import copy
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

from detectron2.data.datasets.builtin_meta import _get_builtin_metadata, COCO_CATEGORIES

# from .coco import load_coco_json, load_sem_seg

def load_coco_panoptic_caption_json(json_file, caption_file, image_dir, gt_dir, semseg_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    from pycocotools.coco import COCO
    caption_json = COCO(caption_file)

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = int(ann["image_id"])
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir, ann["file_name"])
        sem_label_file = os.path.join(semseg_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        caption = caption_json.imgToAnns[image_id]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
                "sem_seg_file_name": sem_label_file,
                "caption": caption,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    return ret


def register_coco_panoptic(
    name, metadata, image_root, panoptic_root, panoptic_json, 
    caption_json, instances_json=None, sem_seg_root=None,
):
    """
    Register a "standard" version of COCO panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".

    Args:
        name (str): the name that identifies a dataset,
            e.g. "coco_2017_train_panoptic"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images in COCO format
        panoptic_json (str): path to the json panoptic annotation file in COCO format
        sem_seg_root (none): not used, to be consistent with
            `register_coco_panoptic_separated`.
        instances_json (str): path to the json instance annotation file
    """
    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_coco_panoptic_caption_json(panoptic_json, caption_json, image_root, panoptic_root, sem_seg_root, metadata),
    )
    MetadataCatalog.get(panoptic_name).set(
        sem_seg_root=sem_seg_root,
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )

_PREDEFINED_SPLITS_COCO_PANOPTIC_CAPTION = {
    "coco_2017_train_panoptic_caption": (
        "coco/train2017",
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        "coco/annotations/captions_train2017.json",
        "coco/annotations/instances_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_semseg_train2017",
    ),
    "coco_2017_val_panoptic_caption": (
        "coco/val2017",
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/annotations/captions_val2017.json",
        "coco/annotations/instances_val2017.json",
        "coco/panoptic_semseg_val2017",
    ),
    # "coco_2017_val_100_panoptic": (
    #     "coco/panoptic_val2017_100",
    #     "coco/annotations/panoptic_val2017_100.json",
    #     "coco/panoptic_stuff_val2017_100",
    # ),
}


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def register_all_coco(root):
    for (
        prefix,
        (image_root, panoptic_root, panoptic_json, caption_json, instances_json, stuff_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC_CAPTION.items():
        # print(image_root, panoptic_root, panoptic_json, caption_json, instances_json)
        # prefix_instances = prefix[: -len("_panoptic")]
        # instances_meta = MetadataCatalog.get(prefix_instances)
        # image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_coco_panoptic(
            prefix,
            # _get_builtin_metadata("coco_panoptic_standard"),
            get_metadata(),
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, caption_json),
            os.path.join(root, instances_json),
            os.path.join(root, stuff_root),
        )

_root = "/opt/tiger/debug/code/ky_open_voca/data"
register_all_coco(_root)
1/82: type(DatasetCatalog.get("coco_2017_train_panoptic_caption"))
1/83: coco_train = DatasetCatalog.get("coco_2017_train_panoptic_caption")
1/84: coco_train[0]['segments_info']
1/85: model, preprocess = clip.load("ViT-B/32", device=device)
1/86: coco_train[0]
1/87: len(coco_train)
1/88: coco_train[0]['image_id']
1/89: meta = get_metadata()
1/90: meta["thing_dataset_id_to_contiguous_id"]
1/91: id2name = {}
1/92:
for cat in COCO_CATEGORIES:
    id2name[meta["stuff_dataset_id_to_contiguous_id"][cat['id']]] = cat["name"]
1/93: id2name
1/94:
with open("coco_name_syn.json") as f:
    coco_cls_syn = json.load(f)
1/95: from PIL import Image
1/96: img = Image.open("/opt/tiger/debug/code/ky_open_voca/data/coco/train2017/000000000009.jpg")
1/97: img.shape
1/98: model.input_resolution
1/99: [0, 14, 434, 374]
1/100: im.crop([0, ])
1/101: [387, 74, 83, 70]
1/102: img.crop([387, 74, 387+83, 74+70]).save("tmp/test.jpg")
1/103: img.save("tmp/ori.jpg")
1/104: coco_train[2]
1/105: coco_train[3]
1/106: coco_train[4]
1/107: img = Image.open("/opt/tiger/debug/code/ky_open_voca/data/coco/train2017/000000000036.jpg")
1/108: [168, 163, 310, 465]
1/109: img.crop([168, 163, 168+310, 163+465]).save("tmp/person.jpg")
1/110: img.crop([163, 168, 163+465, 168+310]).save("tmp/person_1.jpg")
1/111:
def clip_sim(img, cls_list):
    image = preprocess(img).unsqueeze(0).to(device)
    prompt_list = ["a photo of " + c for c in cls_list]
    text = clip.tokenize(prompt_list).to(device)
1/112:
def clip_sim(img, cls_list):
    image = preprocess(img).unsqueeze(0).to(device)
    prompt_list = ["a photo of " + c for c in cls_list]
    text = clip.tokenize(prompt_list).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        logits_per_image, logits_per_text = model(image, text)
    return logits_per_image
1/113: coco_cls_syn[0]
1/114: coco_cls_syn[id2name[0]]
1/115: clip_sim(img.crop([168, 163, 168+310, 163+465]), coco_cls_syn[id2name[0]])
1/116:
def clip_sim(img, cls_list):
    image = preprocess(img).unsqueeze(0).to(device)
    prompt_list = ["a photo of " + c for c in cls_list]
    text = clip.tokenize(prompt_list).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    #return logits_per_image
    return probs
1/117: clip_sim(img.crop([168, 163, 168+310, 163+465]), coco_cls_syn[id2name[0]])
1/118: coco_anno_json = []
1/119:
for c_anno in coco_train[:5]:
    ori_image = Image.open(c_anno['file_name'])
    ret = {'file_name': c_anno['file_name'], 'image_id': c_anno['image_id'], 'segments_info': []}
    for s_anno in c_anno['segments_info']:
        s_ret = {'id': s_anno['id']}
        s_x, s_y, s_w, s_h = s_anno['bbox']
        syn_cls = coco_cls_syn[id2name[s_anno['category_id']]]
        prob = clip_sim(ori_image[s_x, s_y, s_x + s_w, s_y + s_h], syn_cls)
        s_ret['syn_category'] = {cls_ : p for cls_, p in zip(syn_cls, prob)}
        ret['segments_info'].append(s_ret)
    coco_anno_json.append(ret)
with open("coco_anno_syn.json") as f:
    json.dump(coco_anno_json, f)
1/120:
for c_anno in coco_train[:5]:
    ori_image = Image.open(c_anno['file_name'])
    ret = {'file_name': c_anno['file_name'], 'image_id': c_anno['image_id'], 'segments_info': []}
    for s_anno in c_anno['segments_info']:
        s_ret = {'id': s_anno['id']}
        s_x, s_y, s_w, s_h = s_anno['bbox']
        syn_cls = coco_cls_syn[id2name[s_anno['category_id']]]
        prob = clip_sim(ori_image.crop([s_x, s_y, s_x + s_w, s_y + s_h]), syn_cls)
        s_ret['syn_category'] = {cls_ : p for cls_, p in zip(syn_cls, prob)}
        ret['segments_info'].append(s_ret)
    coco_anno_json.append(ret)
with open("coco_anno_syn.json") as f:
    json.dump(coco_anno_json, f)
1/121:
for c_anno in coco_train[:5]:
    ori_image = Image.open(c_anno['file_name'])
    ret = {'file_name': c_anno['file_name'], 'image_id': c_anno['image_id'], 'segments_info': []}
    for s_anno in c_anno['segments_info']:
        s_ret = {'id': s_anno['id'], 'ori_category_id' : s_anno['category_id']}
        s_x, s_y, s_w, s_h = s_anno['bbox']
        syn_cls = coco_cls_syn[id2name[s_anno['category_id']]]
        prob = clip_sim(ori_image.crop([s_x, s_y, s_x + s_w, s_y + s_h]), syn_cls)
        s_ret['syn_category'] = {cls_ : p for cls_, p in zip(syn_cls, prob)}
        ret['segments_info'].append(s_ret)
    coco_anno_json.append(ret)
with open("coco_anno_syn.json", "w") as f:
    json.dump(coco_anno_json, f)
1/122:
for c_anno in coco_train[:5]:
    ori_image = Image.open(c_anno['file_name'])
    ret = {'file_name': c_anno['file_name'], 'image_id': c_anno['image_id'], 'segments_info': []}
    for s_anno in c_anno['segments_info']:
        s_ret = {'id': s_anno['id'], 'ori_category_id' : s_anno['category_id']}
        s_x, s_y, s_w, s_h = s_anno['bbox']
        syn_cls = coco_cls_syn[id2name[s_anno['category_id']]]
        prob = clip_sim(ori_image.crop([s_x, s_y, s_x + s_w, s_y + s_h]), syn_cls)
        s_ret['syn_category'] = {cls_ : float(p) for cls_, p in zip(syn_cls, prob)}
        ret['segments_info'].append(s_ret)
    coco_anno_json.append(ret)
with open("coco_anno_syn.json", "w") as f:
    json.dump(coco_anno_json, f)
1/123: clip_sim(img.crop([168, 163, 168+310, 163+465]), coco_cls_syn[id2name[0]])
1/124:
for c_anno in coco_train[:5]:
    ori_image = Image.open(c_anno['file_name'])
    ret = {'file_name': c_anno['file_name'], 'image_id': c_anno['image_id'], 'segments_info': []}
    for s_anno in c_anno['segments_info']:
        s_ret = {'id': s_anno['id'], 'ori_category_id' : s_anno['category_id']}
        s_x, s_y, s_w, s_h = s_anno['bbox']
        syn_cls = coco_cls_syn[id2name[s_anno['category_id']]]
        prob = clip_sim(ori_image.crop([s_x, s_y, s_x + s_w, s_y + s_h]), syn_cls)
        s_ret['syn_category'] = {cls_ : float(p) for cls_, p in zip(syn_cls, prob.flatten())}
        ret['segments_info'].append(s_ret)
    coco_anno_json.append(ret)
with open("coco_anno_syn.json", "w") as f:
    json.dump(coco_anno_json, f)
1/125: coco_anno_json
1/126:
coco_anno_json = []
for c_anno in coco_train[:5]:
    ori_image = Image.open(c_anno['file_name'])
    ret = {'file_name': c_anno['file_name'], 'image_id': c_anno['image_id'], 'segments_info': []}
    for s_anno in c_anno['segments_info']:
        s_ret = {'id': s_anno['id'], 'ori_category_id' : s_anno['category_id']}
        s_x, s_y, s_w, s_h = s_anno['bbox']
        syn_cls = coco_cls_syn[id2name[s_anno['category_id']]]
        prob = clip_sim(ori_image.crop([s_x, s_y, s_x + s_w, s_y + s_h]), syn_cls)
        s_ret['syn_category'] = {cls_ : float(p) for cls_, p in zip(syn_cls, prob.flatten())}
        ret['segments_info'].append(s_ret)
    coco_anno_json.append(ret)
with open("coco_anno_syn.json", "w") as f:
    json.dump(coco_anno_json, f)
1/127:
coco_anno_json = []
coco_anno_len = len(coco_train)
for i, c_anno in enumerate(coco_train):
    ori_image = Image.open(c_anno['file_name'])
    ret = {'file_name': c_anno['file_name'], 'image_id': c_anno['image_id'], 'segments_info': []}
    for s_anno in c_anno['segments_info']:
        s_ret = {'id': s_anno['id'], 'ori_category_id' : s_anno['category_id']}
        s_x, s_y, s_w, s_h = s_anno['bbox']
        syn_cls = coco_cls_syn[id2name[s_anno['category_id']]]
        prob = clip_sim(ori_image.crop([s_x, s_y, s_x + s_w, s_y + s_h]), syn_cls)
        s_ret['syn_category'] = {cls_ : float(p) for cls_, p in zip(syn_cls, prob.flatten())}
        ret['segments_info'].append(s_ret)
    coco_anno_json.append(ret)
    print("[{}]/[{}]".format(i, coco_anno_len), end='\r')
with open("coco_anno_syn.json", "w") as f:
    json.dump(coco_anno_json, f)
1/128:
a = {'bulwark': 0.38818359375,
     'wall': 0.591796875,
     'rampart': 0.0199432373046875}
1/129:
for k, v in a.item():
    print(k, v)
1/130:
for k, v in a.items():
    print(k, v)
1/131: if None
1/132:
if None:
    print("true")
1/133: del model
1/134: def ori_image
1/135: torch.ones()
1/136: torch.ones(1)
1/137: history
1/138: device = "cuda:3"
1/139: model, preprocess = clip.load("ViT-B/32", device=device)
1/140: del mocel
1/141: del model
1/142: device = "cpu"
1/143: model, preprocess = clip.load("ViT-B/32", device=device)
1/144:
def clip_sim(img, cls_list):
    image = preprocess(Image.open(img)).unsqueeze(0).to(device)
    #prompt_list = ["a photo of " + c for c in cls_list]
    text = clip.tokenize(cls_list).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    #return logits_per_image
    return probs
1/145: clip_sim('code/ky_open_voca/coco_instance/person_01.png', ['person', 'child', 'boy', 'girl'])
1/146: clip_sim('code/ky_open_voca/coco_instance/person_01.png', ['a person', 'a child', 'a boy', 'a girl'])
1/147: clip_sim('code/ky_open_voca/coco_instance/person_01.png', ['a person', 'a child', 'a boy', 'a girl', 'a dog'])
1/148: clip_sim('code/ky_open_voca/coco_instance/person_01.png', ['a person', 'a man', 'a woman', 'a child', 'a boy', 'a girl'])
1/149: clip_sim('code/ky_open_voca/coco_instance/person_02.png', ['a person', 'a man', 'a woman', 'a child', 'a boy', 'a girl'])
1/150: clip_sim('code/ky_open_voca/coco_instance/person_02.png', ['a photo of ' + a for a  in ['person', 'man', 'woman', 'child', 'boy', 'girl']])
1/151: clip_sim('code/ky_open_voca/coco_instance/person_01.png', ['a photo of ' + a for a  in ['person', 'man', 'woman', 'child', 'boy', 'girl']])
1/152: coco_syn
1/153:
with open("coco_syn.json", "w") as f:
    json.dump(coco_syn, f)
1/154: clip_sim('code/ky_open_voca/coco_instance/person_01.png', ['a photo of ' + a for a  in ['person', 'man', 'woman', 'child', 'boy', 'girl']])
1/155: clip_sim('code/ky_open_voca/coco_instance/person_02.png', ['a photo of ' + a for a  in ['person', 'man', 'woman', 'child', 'boy', 'girl']])
1/156: clip_sim('code/ky_open_voca/coco_instance/person_02.png', ['a person', 'a man', 'a woman', 'a child', 'a boy', 'a girl'])
1/157: clip_sim('code/ky_open_voca/coco_instance/person_01.png', ['a person', 'a man', 'a woman', 'a child', 'a boy', 'a girl'])
1/158:
n_list = ['a person', 'a man', 'a woman', 'a child', 'a boy', 'a girl']
for name, prob in zip(n_list, clip_sim('code/ky_open_voca/coco_instance/person_01.png', n_list)[0]):
    print("{}:{.4f}".format(name, prob))
1/159:
n_list = ['a person', 'a man', 'a woman', 'a child', 'a boy', 'a girl']
for name, prob in zip(n_list, clip_sim('code/ky_open_voca/coco_instance/person_01.png', n_list)[0]):
    print("{}:{:.4f}".format(name, prob))
1/160:
n_list = ['a person', 'a man', 'a woman', 'a child', 'a boy', 'a girl']
for name, prob in zip(n_list, clip_sim('code/ky_open_voca/coco_instance/person_01.png', n_list)[0]):
    print("{:10s}:{:.4f}".format(name, prob))
1/161:
n_list = ['a person', 'a man', 'a woman', 'a child', 'a boy', 'a girl']
for name, prob in zip(n_list, clip_sim('code/ky_open_voca/coco_instance/person_02.png', n_list)[0]):
    print("{:10s}:{:.4f}".format(name, prob))
1/162:
n_list = ['a photo of ' + a for a  in ['person', 'man', 'woman', 'child', 'boy', 'girl']]
for name, prob in zip(n_list, clip_sim('code/ky_open_voca/coco_instance/person_02.png', n_list)[0]):
    print("{:10s}:{:.4f}".format(name, prob))
1/163:
n_list = ['a photo of ' + a for a  in ['person', 'man', 'woman', 'child', 'boy', 'girl']]
for name, prob in zip(n_list, clip_sim('code/ky_open_voca/coco_instance/person_02.png', n_list)[0]):
    print("{:20s}:{:.4f}".format(name, prob))
1/164:
n_list = ['a photo of ' + a for a  in ['person', 'man', 'woman', 'child', 'boy', 'girl']]
for name, prob in zip(n_list, clip_sim('code/ky_open_voca/coco_instance/person_01.png', n_list)[0]):
    print("{:20s}:{:.4f}".format(name, prob))
1/165:
n_list = ['a person', 'a man', 'a woman', 'a child', 'a boy', 'a girl']
for name, prob in zip(n_list, clip_sim('code/ky_open_voca/coco_instance/person_01.png', n_list)[0]):
    print("{:10s}:{:.4f}".format(name, prob))
1/166:
n_list = ['a photo of ' + a for a  in ['person', 'man', 'woman', 'child', 'boy', 'girl']]
for name, prob in zip(n_list, clip_sim('code/ky_open_voca/coco_instance/person_02.png', n_list)[0]):
    print("{:20s}:{:.4f}".format(name, prob))
1/167:
n_list = ['a photo of ' + a for a  in ['person', 'man', 'woman', 'child', 'boy', 'girl']]
for name, prob in zip(n_list, clip_sim('code/ky_open_voca/coco_instance/person_02.png', n_list)[0]):
    print("{:20s}:{:.4f}".format(name, prob))
1/168:
n_list = ['a person', 'a man', 'a woman', 'a child', 'a boy', 'a girl']
for name, prob in zip(n_list, clip_sim('code/ky_open_voca/coco_instance/person_02.png', n_list)[0]):
    print("{:10s}:{:.4f}".format(name, prob))
1/169: coco_name
 2/1: import torch
 2/2: a = torch.rand((2, 3, 5))
 2/3: a
 2/4: cls_ind = [[0, 1], [2], [3, 4]]
 2/5: a[:, :, cls_ind]
 2/6: a[[[0, 1], [0, 1], [0, 1]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]], cls_ind]
 2/7: a.requires_grad = True
 2/8: a
 2/9: a.index_select(2, cls_ind)
2/10: a.index_select(2, cls_ind[0])
2/11: a.index_select(2, [0, 1])
2/12: a.index_select(2, torch.LongTensor([0, 1]))
2/13: a[:, :, [0, 1]]
2/14: a.index_select(2, torch.LongTensor([0, 1], [2, 3]))
2/15: a.index_select(2, torch.LongTensor([[0, 1], [2, 3]]))
2/16: a.index_select([2, 2], [torch.LongTensor([0, 1]), torch.LongTensor([2, 3, 4])])
2/17: a.index_select(2, torch.LongTensor([0, 1])).max(dim=2)
2/18: a.index_select(2, torch.LongTensor([0, 1]))
2/19: a.index_select(2, torch.LongTensor([0, 1])).mean(dim=2)
2/20: a.index_select(2, torch.LongTensor([0, 1])).mean(dim=2, keepdim=True)
2/21: a.index_select(2, torch.LongTensor([0, 1])).max(dim=2, keepdim=True)
2/22: a.index_select(2, torch.LongTensor([0, 1])).max(dim=2, keepdim=True)
2/23: a.index_select(2, torch.LongTensor([0, 1])).max(dim=2, keepdim=True)[0]
2/24: torch.cat([a.index_select(2, torch.LongTensor(ind)).max(dim=2, keepdim=True)[0] for ind in cls_ind], dim=2)
2/25: a
2/26: cls_ind
   1: history
   2: %history -g -f ipython_commanc.py
