import copy
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

from detectron2.data.datasets.builtin_meta import _get_builtin_metadata, COCO_CATEGORIES

# from .coco import load_coco_json, load_sem_seg

def load_coco_panoptic_caption_json(json_file, caption_file, image_dir, gt_dir, semseg_dir, panoptic_name, meta):
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
                "dataset": panoptic_name,
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
        lambda: load_coco_panoptic_caption_json(panoptic_json, caption_json, image_root, panoptic_root, sem_seg_root, panoptic_name, metadata),
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
    "coco_2017_train_panoptic_caption_base_ignore": (
        "coco/train2017",
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017_base_ignore.json",
        "coco/annotations/captions_train2017.json",
        "coco/annotations/instances_train2017.json",
        "coco/panoptic_semseg_train2017",
        "base"
    ),
    "coco_2017_train_panoptic_caption_base_delete": (
        "coco/train2017",
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017_base_delete.json",
        "coco/annotations/captions_train2017.json",
        "coco/annotations/instances_train2017.json",
        "coco/panoptic_semseg_train2017",
        "base"
    ),
    "coco_2017_val_panoptic_caption_base": (
        "coco/val2017",
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017_base.json",
        "coco/annotations/captions_val2017.json",
        "coco/annotations/instances_val2017.json",
        "coco/panoptic_semseg_val2017_base",
        "base"
    ),
    "coco_2017_val_panoptic_caption_novel": (
        "coco/val2017",
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017_novel.json",
        "coco/annotations/captions_val2017.json",
        "coco/annotations/instances_val2017.json",
        "coco/panoptic_semseg_val2017_novel",
        "novel"
    ),
}


def get_metadata(split='novel'):
    novel_clsID = [2, 33, 21, 25, 95, 76, 53, 57, 70, 145, 87, 46, 13, 100, 34, 41, 175, 148]
    coco_novel = [k for k in COCO_CATEGORIES if k["id"] in novel_clsID]
    coco_base = [k for k in COCO_CATEGORIES if k["id"] not in novel_clsID]
    if split == 'novel':
        COCO_SPLIT_CATEGORIES = coco_novel
        COCO_REMAIN_CATEGORIES = coco_base
    elif split == 'base':
        COCO_SPLIT_CATEGORIES = coco_base
        COCO_REMAIN_CATEGORIES = coco_novel
    else:
        COCO_SPLIT_CATEGORIES = COCO_CATEGORIES
        COCO_REMAIN_CATEGORIES = None

    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in COCO_SPLIT_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_SPLIT_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in COCO_SPLIT_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_SPLIT_CATEGORIES]

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

    dataset_contiguous_id2name = {}

    for i, cat in enumerate(COCO_SPLIT_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i
        dataset_contiguous_id2name[i] = cat["name"]

    if COCO_REMAIN_CATEGORIES is not None:
        another_split_classname = []
        for i, cat in enumerate(COCO_REMAIN_CATEGORIES):
            another_split_classname.append(cat["name"])
        meta["another_split_classname"] = another_split_classname

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
    meta["dataset_contiguous_id2name"] = dataset_contiguous_id2name

    return meta


def register_all_coco(root):
    for (
        prefix,
        (image_root, panoptic_root, panoptic_json, caption_json, instances_json, stuff_root, split),
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
            get_metadata(split),
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, caption_json),
            os.path.join(root, instances_json),
            os.path.join(root, stuff_root),
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco(_root)