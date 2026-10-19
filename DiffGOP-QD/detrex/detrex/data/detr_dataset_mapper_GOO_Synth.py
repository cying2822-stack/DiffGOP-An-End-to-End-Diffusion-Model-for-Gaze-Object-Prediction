# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
# ------------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# All Rights Reserved
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/facebookresearch/detr/blob/main/d2/detr/dataset_mapper.py
# ------------------------------------------------------------------------------------------------

import copy
import json
import logging
import numpy as np
import torch

from PIL import Image
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode

__all__ = ["DetrDatasetMapper"]


class DetrDatasetMapper:
    """
    目标：
    1. 保留 DetrDatasetMapper 的类结构与 train / infer 主分支
    2. 只整理重复逻辑，不改变原有行为
    3. gaze 相关处理统一收敛到私有函数中
    """

    def __init__(
        self,
        augmentation,
        augmentation_with_crop,
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ):
        self.mask_on = mask_on
        self.augmentation = augmentation
        self.augmentation_with_crop = augmentation_with_crop
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(
                str(self.augmentation), str(self.augmentation_with_crop)
            )
        )

        self.img_format = img_format
        self.is_train = is_train

        # 只做去重，不改原行为：train / val 仍分别读各自 ann 文件
        self.ann_file = (
            "/disk6/cy/DiffusionDet-main/datasets/coco/annotations/instances_train2017.json"
            if is_train
            else "/disk6/cy/DiffusionDet-main/datasets/coco/annotations/instances_val2017.json"
        )
        with open(self.ann_file, "r") as f:
            self.json_data = json.load(f)
            self.gaze_related_ann = self.json_data["annotations_gaze"]
            self.gaze_dict = {item["image_id"]: item for item in self.gaze_related_ann}

    def _get_matching_gaze_item(self, dataset_dict):
        image_id = dataset_dict["image_id"]
        matching_item = self.gaze_dict.get(image_id)
        if matching_item is None:
            raise ValueError(f"No matching gaze annotation found for image_id: {image_id}")
        return matching_item

    def _get_eye_and_gaze(self, matching_item, width, height):
        head_point = matching_item["head_point"]
        hx, hy = head_point
        eye = [float(hx) / width, float(hy) / height]

        gaze_point = matching_item["gaze_point"]
        gaze_cx, gaze_cy = gaze_point
        gaze = [float(gaze_cx) / width, float(gaze_cy) / height]
        return eye, gaze

    def _prepare_segmentation_annotations(self, segmentation_annotations, width, height):
        """
        保持你原先的行为：
        1. xywh -> xyxy
        2. 缩放到 224
        3. clip 到 [0, 224]
        4. 过滤掉宽高 <= 1 的框
        """
        segmentation_annotations_list = []
        for item in segmentation_annotations:
            item = copy.deepcopy(item)
            item["bbox"] = np.array([item["bbox"]])
            item["bbox"][:, 2] = item["bbox"][:, 0] + item["bbox"][:, 2]
            item["bbox"][:, 3] = item["bbox"][:, 1] + item["bbox"][:, 3]
            item["bbox_mode"] = BoxMode.XYXY_ABS
            segmentation_annotations_list.append(item)
        segmentation_annotations = segmentation_annotations_list

        segmentation_annotations_list = []
        for item in segmentation_annotations:
            box = item["bbox"]
            box[:, [0, 2]] = box[:, [0, 2]] * 224 / width
            box[:, [1, 3]] = box[:, [1, 3]] * 224 / height
            item["bbox"] = box.astype(np.float64)
            segmentation_annotations_list.append(item)
        segmentation_annotations = segmentation_annotations_list

        new_segmentation_annotations = []
        for item in segmentation_annotations:
            box = item["bbox"]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > 224] = 224
            box[:, 3][box[:, 3] > 224] = 224
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            box = np.array(box, dtype=np.float64)

            if box.shape[0] != 0:
                box = box[0]
                item["bbox"] = box
                new_segmentation_annotations.append(item)

        return new_segmentation_annotations

    def _compute_head_box(self, eye_x, eye_y, width, height):
        # 与你原逻辑保持一致
        k = 0.1
        x_min = (eye_x - 0.15) * width
        y_min = (eye_y - 0.15) * height
        x_max = (eye_x + 0.15) * width
        y_max = (eye_y + 0.15) * height

        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if x_max < 0:
            x_max = 0
        if y_max < 0:
            y_max = 0

        x_min -= k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += k * abs(x_max - x_min)
        y_max += k * abs(y_max - y_min)

        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if x_max > width:
            x_max = width
        if y_max > height:
            y_max = height

        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])
        return x_min, y_min, x_max, y_max

    def _build_gaze_fields(self, dataset_dict):
        """
        只把重复代码抽出来，不改字段、不改尺寸、不改行为
        """
        width, height = dataset_dict["width"], dataset_dict["height"]
        dataset_dict["gaze_related_ann"] = self.gaze_related_ann

        matching_item = self._get_matching_gaze_item(dataset_dict)
        eye, gaze = self._get_eye_and_gaze(matching_item, width, height)

        gaze_x, gaze_y = gaze
        eye_x, eye_y = eye

        if "annotations" in dataset_dict:
            dataset_dict["annotations"] = self._prepare_segmentation_annotations(
                dataset_dict["annotations"], width, height
            )

        img = Image.open(dataset_dict["file_name"])
        if img.mode != "RGB":
            img = img.convert("RGB")

        x_min, y_min, x_max, y_max = self._compute_head_box(eye_x, eye_y, width, height)

        dataset_dict["gaze"] = [gaze_x, gaze_y]
        dataset_dict["eye"] = [eye_x, eye_y]
        dataset_dict["gaze_cx"] = gaze_x
        dataset_dict["gaze_cy"] = gaze_y
        dataset_dict["hx"] = eye_x
        dataset_dict["hy"] = eye_y

        head_channel = utils.get_head_box_channel(
            x_min, y_min, x_max, y_max, width, height, resolution=224, coordconv=False
        ).unsqueeze(0)
        dataset_dict["head_channel"] = head_channel

        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        face = face.resize((224, 224), Image.BICUBIC)
        face = np.array(face)
        face = torch.as_tensor(np.ascontiguousarray(face.transpose(2, 0, 1)))
        dataset_dict["face"] = face

        img = img.resize((224, 224), Image.BICUBIC)
        img = np.array(img)
        img = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
        dataset_dict["image"] = img

        gaze_heatmap = torch.zeros(64, 64)
        gaze_heatmap = utils.draw_labelmap(
            gaze_heatmap, [gaze_x * 64, gaze_y * 64], 3, type="Gaussian"
        )
        dataset_dict["gaze_heatmap"] = gaze_heatmap

        return dataset_dict

    def _build_train_instances(self, dataset_dict, image_shape):
        # 完全保留 Detr 原来的 instances 处理习惯
        if "annotations" not in dataset_dict:
            return dataset_dict

        annos = dataset_dict["annotations"]

        for anno in dataset_dict["annotations"]:
            if not self.mask_on:
                anno.pop("segmentation", None)
            anno.pop("keypoints", None)

        instances = utils.annotations_to_instances(annos, image_shape)
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)

        # 保留原 Detr 的读取与增强结构，但后续 image 字段仍按你原 gaze 逻辑覆盖成 224
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # if self.augmentation_with_crop is None:
        #     image, transforms = T.apply_transform_gens(self.augmentation, image)
        # else:
        #     if np.random.rand() > 0.5:
        #         image, transforms = T.apply_transform_gens(self.augmentation, image)
        #     else:
        #         image, transforms = T.apply_transform_gens(self.augmentation_with_crop, image)

        image_shape = (224, 224)

        # 这里只统一 gaze 部分，保持你之前那套 224 行为
        dataset_dict = self._build_gaze_fields(dataset_dict)

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        dataset_dict = self._build_train_instances(dataset_dict, image_shape)
        return dataset_dict
