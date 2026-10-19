# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# All Rights Reserved
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/facebookresearch/detr/blob/main/d2/detr/dataset_mapper.py
# ------------------------------------------------------------------------------------------------

import copy
import logging
import numpy as np
import torch
import json
import os

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from PIL import Image, ImageDraw, ImageFont
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)
import torchvision.transforms.functional as TF

__all__ = ["DetrDatasetMapper"]


gaze_json_path = "/disk6/cy/DiffusionDet-main/gaze.json" # 替换为你的gaze.json文件路径
with open(gaze_json_path, 'r', encoding='utf-8') as file:
    gaze_data = json.load(file)
# 将 gaze_data 转换为字典，方便根据 id 查找
gaze_dict = {item['id']: item['gaze_related_ann'] for item in gaze_data}


gaze_json_path_val = "/disk6/cy/DiffusionDet-main/gaze_reference.json"  # 替换为你的gaze.json文件路径
with open(gaze_json_path_val, 'r', encoding='utf-8') as file:
    gaze_data_val = json.load(file)
# 将 gaze_data 转换为字典，方便根据 id 查找
gaze_dict_val = {item['id']: item['gaze_related_ann'] for item in gaze_data_val}
a =0
class DetrDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into the format used by DETR.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors

    Args:
        augmentation (list[detectron.data.Transforms]): The geometric transforms for
            the input raw image and annotations.
        augmentation_with_crop (list[detectron.data.Transforms]): The geometric transforms with crop.
        is_train (bool): Whether to load train set or val set. Default: True.
        mask_on (bool): Whether to return the mask annotations. Default: False.
        img_format (str): The format of the input raw images. Default: RGB.

    Because detectron2 did not implement `RandomSelect` augmentation. So we provide both `augmentation` and
    `augmentation_with_crop` here and randomly apply one of them to the input raw images.
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

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)

        image_id = dataset_dict.get('image_id')
        if self.is_train:
            if image_id in gaze_dict:
                # 将 gaze_related_ann 键及其对应的值导入到 dataset_dict 中
                dataset_dict['gaze_related_ann'] = copy.deepcopy(gaze_dict[image_id])
        else:
            if image_id in gaze_dict_val:
                # 将 gaze_related_ann 键及其对应的值导入到 dataset_dict 中
                dataset_dict['gaze_related_ann'] = copy.deepcopy(gaze_dict_val[image_id])

        width, height = dataset_dict['width'], dataset_dict['height']
        gaze_related_ann = dataset_dict['gaze_related_ann']
        gaze_related_ann['gaze_cx'], gaze_related_ann['gaze_cy'] = gaze_related_ann['gaze_cx'] / 640 * 1920, \
                                                                   gaze_related_ann['gaze_cy'] / 480 * 1080
        gaze_related_ann['hx'], gaze_related_ann['hy'] = gaze_related_ann['hx'] / 640 * 1920, gaze_related_ann[
            'hy'] / 480 * 1080

        ##########################################################
        segmentation_annotations_list = []
        segmentation_annotations = dataset_dict['annotations']
        for index, item in enumerate(segmentation_annotations):
            item['bbox'] = np.array([item['bbox']])
            item['bbox'][:, 2] = item['bbox'][:, 0] + item['bbox'][:, 2]
            item['bbox'][:, 3] = item['bbox'][:, 1] + item['bbox'][:, 3]
            item['bbox_mode'] = BoxMode.XYXY_ABS
            segmentation_annotations_list.append(item)
        segmentation_annotations = segmentation_annotations_list
        ##########################################################

        gaze_box = gaze_related_ann['gaze_box']
        gaze_box = np.array(gaze_box, dtype=np.int32).reshape(1, 5)

        eye = [float(gaze_related_ann['hx']) / 1920, float(gaze_related_ann['hy']) / 1080]
        gaze = [float(gaze_related_ann['gaze_cx']) / 1920, float(gaze_related_ann['gaze_cy']) / 1080]
        output_folder = 'output_images'
        # for img in dataset_dict['file_name']:
        # count = 1
        # ############################################
        # # 2024.11.19
        # img = Image.open(dataset_dict['file_name'])
        # img = img.convert('RGB')
        # #############################################
        img = Image.open(dataset_dict['file_name'])

        gaze_x, gaze_y = gaze
        eye_x, eye_y = eye

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

        if self.is_train:
            ##############################################################################
            # 随机裁剪：在50%的情况下进行随机裁剪操作。
            #
            # 裁剪范围的确定：
            #
            # crop_x_min, crop_y_min, crop_x_max, crop_y_max 是计算裁剪区域的边界，确保裁剪的范围不会排除关键元素（如注视点 gaze 和脸部边界框 x_min, y_min, x_max, y_max）。
            # 使用 np.min 和 np.max 计算出裁剪区域的最小和最大边界，这样即使裁剪，目标对象（如注视点或脸）依然保留在裁剪后的图像中。
            # 随机选择裁剪起点：
            #
            # 从左上角开始，crop_x_min 和 crop_y_min 是根据裁剪范围计算出来的。
            # 然后使用 np.random.uniform() 随机选择在这个范围内的一个点作为裁剪的起始位置。
            # 裁剪的宽度和高度：
            #
            # crop_width_min 和 crop_height_min 是裁剪范围的最小值，即从选定的左上角到裁剪区域的右下角的距离。
            # crop_width_max 和 crop_height_max 则是整个图像的剩余宽度和高度，从裁剪起始点开始。
            # 然后，使用 np.random.uniform() 随机选择裁剪的实际宽度和高度，使裁剪范围在最小和最大值之间变化。
            # 执行裁剪：
            #
            # 使用 TF.crop() 对图像执行裁剪操作，crop_x_min 和 crop_y_min 是裁剪的左上角，crop_width 和 crop_height 是裁剪的尺寸。
            # 裁剪后的图像保存为 '5.jpg'。
            #########################################################
            # Jitter (expansion-only) bounding box size
            if np.random.random_sample() <= 0.5:
                k = np.random.random_sample() * 0.2
                x_min -= k * abs(x_max - x_min)
                y_min -= k * abs(y_max - y_min)
                x_max += k * abs(x_max - x_min)
                y_max += k * abs(y_max - y_min)
                Jitter = True
            # Random Crop
            if np.random.random_sample() <= 0.5:
                Crop = True
                # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
                crop_x_min = np.min([gaze_x * width, x_min, x_max])
                crop_y_min = np.min([gaze_y * height, y_min, y_max])
                crop_x_max = np.max([gaze_x * width, x_min, x_max])
                crop_y_max = np.max([gaze_y * height, y_min, y_max])

                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)

                # Crop it
                img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)

                # Record the crop's (x, y) offset
                offset_x, offset_y = crop_x_min, crop_y_min

                # convert coordinates into the cropped frame
                x_min, y_min, x_max, y_max = x_min - offset_x, y_min - offset_y, x_max - offset_x, y_max - offset_y
                # if gaze_inside:
                gaze_x, gaze_y = (gaze_x * width - offset_x) / float(crop_width), \
                                 (gaze_y * height - offset_y) / float(crop_height)
                eye_x, eye_y = (eye_x * width - offset_x) / float(crop_width), \
                               (eye_y * height - offset_y) / float(crop_height)
                # 分割对象边界框转换
                segmentation_annotations_list = []
                for index, item in enumerate(segmentation_annotations):
                    item['bbox'][:, [0, 2]] = item['bbox'][:, [0, 2]] - crop_x_min
                    item['bbox'][:, [1, 3]] = item['bbox'][:, [1, 3]] - crop_y_min
                    segmentation_annotations_list.append(item)
                segmentation_annotations = segmentation_annotations_list

                width, height = crop_width, crop_height

                # head_box[:, [0, 2]] = head_box[:, [0, 2]] - crop_x_min
                # head_box[:, [1, 3]] = head_box[:, [1, 3]] - crop_y_min

                # operate gt_box
                gaze_box[:, [0, 2]] = gaze_box[:, [0, 2]] - crop_x_min
                gaze_box[:, [1, 3]] = gaze_box[:, [1, 3]] - crop_y_min
            # Random flip
            if np.random.random_sample() <= 0.5:
                flip = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                x_max_2 = width - x_min
                x_min_2 = width - x_max
                x_max = x_max_2
                x_min = x_min_2
                gaze_x = 1 - gaze_x
                eye_x = 1 - eye_x
                segmentation_annotations_list = []
                for index, item in enumerate(segmentation_annotations):
                    item['bbox'][:, [0, 2]] = width - item['bbox'][:, [2, 0]]
                    segmentation_annotations_list.append(item)
                segmentation_annotations = segmentation_annotations_list

                # head_box[:, [0, 2]] = width - head_box[:, [2, 0]]
                gaze_box[:, [0, 2]] = width - gaze_box[:, [2, 0]]

            # Random color change
            if np.random.random_sample() <= 0.5:
                img = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

            # Random color change
            if np.random.random_sample() <= 0.5:
                img = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

        gaze_p = np.array([gaze_x, gaze_y])
        eye_p = np.array([eye_x, eye_y])

        dataset_dict['gaze'] = [gaze_x, gaze_y]
        dataset_dict['eye'] = [eye_x, eye_y]
        dataset_dict['gaze_related_ann']['gaze_cx'] = gaze_x
        dataset_dict['gaze_related_ann']['gaze_cy'] = gaze_y
        dataset_dict['gaze_related_ann']['hx'] = eye_x
        dataset_dict['gaze_related_ann']['hy'] = eye_y

        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if x_max > width:
            x_max = width
        if y_max > height:
            y_max = height

        head_channel = utils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                                  resolution=224, coordconv=False).unsqueeze(0)
        dataset_dict['head_channel'] = head_channel
        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        face = face.resize((224, 224), Image.BICUBIC)
        face = np.array(face)
        face = torch.as_tensor(np.ascontiguousarray(face.transpose(2, 0, 1)))
        dataset_dict['face'] = face
        # face = self.transform(face)

        img = img.resize(([224, 224]), Image.BICUBIC)  # 283行image
        dataset_dict['ori_img'] = img  # 322行image
        #
        im = img  # 322行image
        dataset_dict['keshihua'] = im
        img = np.array(img)  ##322行image
        ims = img  # 326行image
        img = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))  # 326行image
        dataset_dict['image'] = img  # 328行image

        # operate_gt_box
        gaze_box[:, [0, 2]] = gaze_box[:, [0, 2]] * 224 / width
        gaze_box[:, [1, 3]] = gaze_box[:, [1, 3]] * 224 / height

        gaze_box[:, 0:2][gaze_box[:, 0:2] < 0] = 0
        gaze_box[:, 2:4][gaze_box[:, 2:4] > 224] = 224

        dataset_dict['gaze_box'] = gaze_box
        normal_box = normalize_gaze_box_xyxy(gaze_box, 224, 224)
        dataset_dict['normal_box'] = normal_box


        segmentation_annotations_list = []
        for index, item in enumerate(segmentation_annotations):
            box = item['bbox']
            box[:, [0, 2]] = box[:, [0, 2]] * 224 / width
            box[:, [1, 3]] = box[:, [1, 3]] * 224 / height
            item['bbox'] = box.astype(np.float64)

            segmentation_annotations_list.append(item)
        segmentation_annotations = segmentation_annotations_list

        new_segmentation_annotations = []
        for index, item in enumerate(segmentation_annotations):
            box = item['bbox']
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > 224] = 224
            box[:, 3][box[:, 3] > 224] = 224
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            box = np.array(box, dtype=np.float64)

            if box.shape[0] != 0:
                box = box[0]
                item['bbox'] = box
                new_segmentation_annotations.append(item)
        dataset_dict['annotations'] = new_segmentation_annotations
        gaze_heatmap = torch.zeros(64, 64)  # set the size of the output
        gaze_heatmap = utils.draw_labelmap(gaze_heatmap, [gaze_x * 64, gaze_y * 64],
                                           3,
                                           type='Gaussian')
        dataset_dict['gaze_heatmap'] = gaze_heatmap
        image_shape = (224, 224)
        CLASSES = [
            "Kellogg Corn Flakes", "Honey Stars", "Koko Krunch", "Fita", "Mamon", "Buttercake",
            "Wafer", "Choco Crunchies", "iHop", "Butterlicious", "Pik-nik", "Pringles",
            "Super Fruits", "Locally Mango", "Camomille", "Oishi Prawn", "Kimchi Noodles",
            "Grahams", "Hansel", "Danisa", "Chocolate Chips", "Fresh Milk", "Bear brand", "Chuckie"
        ]

        def draw_label_box(draw, box_xyxy, label, outline='red', width=3, bg='black', fg='white'):
            """在框的左上角加一个带底色的标签"""
            draw.rectangle(box_xyxy, outline=outline, width=width)
            try:
                font = ImageFont.truetype("arial.ttf", 16)  # 如需中文请换成本机可用中文字体
            except:
                font = ImageFont.load_default()
            x1, y1, x2, y2 = box_xyxy
            bbox = draw.textbbox((0, 0), label, font=font)  # 返回 (x0, y0, x1, y1)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pad = 2
            bg_box = [x1, max(0, y1 - th - 2 * pad), x1 + tw + 2 * pad, max(0, y1 - th - 2 * pad) + th + 2 * pad]
            draw.rectangle(bg_box, fill=bg)
            draw.text((bg_box[0] + pad, bg_box[1] + pad), label, fill=fg, font=font)

        # utils.check_image_size(dataset_dict, image)
        #
        # if self.augmentation_with_crop is None:
        #     image, transforms = T.apply_transform_gens(self.augmentation, image)
        # else:
        #     if np.random.rand() > 0.5:
        #         image, transforms = T.apply_transform_gens(self.augmentation, image)
        #     else:
        #         image, transforms = T.apply_transform_gens(self.augmentation_with_crop, image)
        #
        # image_shape = image.shape[:2]  # h, w
        #
        # # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # # Therefore it's important to use torch.Tensor.
        # dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            annos = new_segmentation_annotations
            #
            # # visualization check
            # output_folder = "/disk6/cy/DiffuDETR-main/output_images/"
            # file_name = dataset_dict['file_name']
            # file_name = os.path.basename(file_name)  # 获取文件名而非完整路径
            # file_path = os.path.join(output_folder, file_name)  # 保存路径
            # draw = ImageDraw.Draw(im)
            # # for item in annos:
            # #     bbox = item['bbox'].astype(int)
            # #     bbox = bbox.tolist()
            # #     draw.rectangle(bbox, fill=None, outline='green', width=3)
            # gb = np.asarray(gaze_box).reshape(-1, 5)[0]  # 保稳妥
            # x1, y1, x2, y2 = gb[:4].tolist()
            # cls_id = int(gb[4])
            # # 防越界与合法性
            # x1, y1, x2, y2 = map(int, gb[:4])
            # cls_id = max(0, min(cls_id, len(CLASSES) - 1))
            # label = CLASSES[cls_id]
            # draw_label_box(draw, [x1, y1, x2, y2], label, outline='red', width=3)
            # # draw.rectangle(gaze_box[0][:4].tolist(), fill=None, outline='red', width=3)
            # # draw.rectangle(head_box_large[0].tolist(), fill=None, outline='red', width=3)
            # # draw.rectangle([eye_x * 224 - 40, eye_y * 224 - 40, eye_x * 224 + 40, eye_y * 224 + 40],fill=None, outline='red', width=3)
            # draw.ellipse([gaze_x*224 - 2, gaze_y*224 - 2, gaze_x*224 + 2, gaze_y*224 + 2],fill='red', outline=None)
            # draw.ellipse([eye_x*224- 2, eye_y*224 - 2, eye_x*224 + 2, eye_y*224 + 2], fill='red',outline=None)
            # draw.line([eye_x*224, eye_y*224, gaze_x*224, gaze_y*224],fill='yellow',width=2)
            # im.save(file_path)
            # x=0

            # draw = ImageDraw.Draw(im)
            # for item in annos:
            #     # bbox = item['bbox'].astype(int)    bbox是一个list，没有astype(int)方法，把他转化成numpy数组
            #     bbox = np.array(item['bbox']).astype(int)
            #     # x0, y0, w, h = bbox
            #     # x1 = x0 + w
            #     # y1 = y0 + h
            #     # # 转换为列表格式以用于 PIL 绘制
            #     # bbox = [x0, y0, x1, y1]
            #     # 绘制目标检测的边界框，使用绿色，线条宽度为3
            #     coordinates = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
            #     draw.rectangle(coordinates, fill=None, outline='blue', width=3)
            #
            #     bbox = np.array(bbox)
            #     bbox = bbox.tolist()
            #     draw.rectangle(bbox, fill=None, outline='green', width=3)
            # draw.rectangle(gaze_box[0][:4].tolist(), fill=None, outline='red', width=3)
            # # draw.rectangle(head_box_large[0].tolist(), fill=None, outline='red', width=3)
            # # draw.rectangle([eye_x * 224 - 40, eye_y * 224 - 40, eye_x * 224 + 40, eye_y * 224 + 40],fill=None, outline='red', width=3)
            # # 绘制眼睛位置
            # draw.ellipse([gaze_x * 224 - 2, gaze_y * 224 - 2, gaze_x * 224 + 2, gaze_y * 224 + 2], fill='red',
            #              outline=None)
            # # 绘制注视点
            # draw.ellipse([eye_x * 224 - 2, eye_y * 224 - 2, eye_x * 224 + 2, eye_y * 224 + 2], fill='red', outline=None)
            # draw.line([eye_x * 224, eye_y * 224, gaze_x * 224, gaze_y * 224], fill='yellow', width=2)
            # # im_with_hm = utils.generate_attention_map(im, gaze_cone)
            # # im_with_hm.show()
            # im.show()
            # im.save('1.jpg')
            # x = 0

            # USER: Implement additional transformations if you have other types of data
            # annos = [
            #     utils.transform_instance_annotations(obj, transforms, image_shape)
            #     for obj in dataset_dict.pop("annotations")
            #     if obj.get("iscrowd", 0) == 0
            # ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict

def normalize_gaze_box_xyxy(gaze_bboxes, W, H):
    # gaze_bboxes: (1,5) -> [x1, y1, x2, y2, cls]
    # gaze_box = gaze_bboxes[0, :4].float().clone()
    gaze_box = torch.as_tensor(gaze_bboxes[0, :4], dtype=torch.float32)
    gaze_box[[0, 2]] /= W
    gaze_box[[1, 3]] /= H
    return gaze_box
