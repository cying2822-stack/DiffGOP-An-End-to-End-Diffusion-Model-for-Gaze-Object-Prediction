# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
import json

import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
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
import time

# ===================================================================
from diffusiondet.common import pair, expand_bbox, generate_gaze_heatmap, generate_mask, get_img_size, square_bbox
from diffusiondet.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomCropSafeGaze,
    RandomHeadBboxJitter,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
__all__ = ["DiffusionDetDatasetMapper"]

def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    # ResizeShortestEdge
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))

    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


class DiffusionDetDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DiffusionDet.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

        self.jitter_bbox = RandomHeadBboxJitter(p=1.0, tr= (-0.1, 0.1) )
        self.ann_file = " "  # Path to the GOO-Synth training annotation file

        with open (self.ann_file, 'r') as f:    
            self.json_data = json.load(f)
            self.gaze_dict = {
                item['image_id']: item
                for item in self.json_data['annotations_gaze'] }
    def __call__(self, dataset_dict):

        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        width, height = dataset_dict['width'], dataset_dict['height']

        gaze_related_ann = self.json_data['annotations_gaze']
        dataset_dict['gaze_related_ann'] = gaze_related_ann

        # gaze
        segmentation_annotations_list = []
        segmentation_annotations = dataset_dict['annotations']
        for index, item in enumerate(segmentation_annotations):
            item['bbox'] = np.array([item['bbox']])
            item['bbox'][:, 2] = item['bbox'][:, 0] + item['bbox'][:, 2]
            item['bbox'][:, 3] = item['bbox'][:, 1] + item['bbox'][:, 3]
            item['bbox_mode'] = BoxMode.XYXY_ABS
            segmentation_annotations_list.append(item)
        segmentation_annotations = segmentation_annotations_list


        image_id = dataset_dict['image_id']
        matching_item = self.gaze_dict.get(image_id)

        if matching_item is None:
            raise ValueError(f"No matching gaze annotation found for image_id: {image_id}")

        gaze_box = matching_item['gaze_bbox']
        x, y, w, h = gaze_box
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        gaze_box = x1, y1,x2,y2
        gaze_box = np.array(gaze_box, dtype=np.int32).reshape(1, 4)

        head_point = matching_item['head_point']
        hx, hy = head_point
        eye = [float(hx) / 640, float(hy) / 480]
        gaze_point = matching_item['gaze_point']
        gaze_cx, gaze_cy = gaze_point
        gaze = [float(gaze_cx) / 640, float(gaze_cy) / 480]

        img = Image.open(dataset_dict['file_name'])

        if img.mode == "RGBA":
            img = img.convert("RGB")

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
        dataset_dict['gaze_cx'] = gaze_x
        dataset_dict['gaze_cy'] = gaze_y
        dataset_dict['hx'] = eye_x
        dataset_dict['hy'] = eye_y

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
        img = img.resize(([224, 224]), Image.BICUBIC)    #283��image
        dataset_dict['ori_img'] = img   #322��image
        #
        im = img   #322��image
        dataset_dict['keshihua'] = im
        img = np.array(img)   ##322��image
        ims = img     #326��image
        img = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))   #326��image
        dataset_dict['image'] = img   #328��image




        # operate_gt_box
        gaze_box[:, [0, 2]] = gaze_box[:, [0, 2]] * 224 / width
        gaze_box[:, [1, 3]] = gaze_box[:, [1, 3]] * 224 / height

        gaze_box[:, 0:2][gaze_box[:, 0:2] < 0] = 0
        gaze_box[:, 2:4][gaze_box[:, 2:4] > 224] = 224
        # head_box[:, [0, 2]] = head_box[:, [0, 2]] * 224 / width
        # head_box[:, [1, 3]] = head_box[:, [1, 3]] * 224 / height
        #
        # head_box[:, 0:2][head_box[:, 0:2] < 0] = 0
        # head_box[:, 2:4][head_box[:, 2:4] > 224] = 224


        dataset_dict['gaze_box'] = gaze_box
        # dataset_dict['head_box'] = head_box


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

# =============================================================================================
#         head_box_1 = head_box.reshape(-1)
#         head_box_gaze = torch.from_numpy(head_box_1.astype(np.float32)).unsqueeze(0)
#         head_box_gaze = expand_bbox(head_box_gaze, width, height, k=0.1)  # annotated boxes are a bit tight
#         target_head_center = torch.stack([head_box_gaze[0, [0, 2]].mean(), head_box_gaze[0, [1, 3]].mean()])
#         target_head_center /= torch.tensor([width, height], dtype=torch.float)
#         head_box_gaze = head_box_gaze.to(torch.float)
#         head_box_gaze = square_bbox(head_box_gaze, width, height)
#         head_box_gaze /= torch.tensor([width, height, width, height], dtype=torch.float)
#         head_box_gaze = torch.clamp(head_box_gaze, min=0.0, max=1.0)
#         dataset_dict['head_box_gaze'] = head_box_gaze
# -------------------------------------------------------------------------------


        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                # Let's always keep mask
                anno.pop("keypoints", None)
            annos = new_segmentation_annotations

            # # visualization check
            # output_folder = "/data1/cy/DiffusionDet-main/output_images/"
            # file_name = dataset_dict['file_name']
            # file_name = os.path.basename(file_name) 
            # file_path = os.path.join(output_folder, file_name)  
            # draw = ImageDraw.Draw(im)
            # for item in annos:
            #     bbox = item['bbox'].astype(int)
            #     bbox = bbox.tolist()
            #     draw.rectangle(bbox, fill=None, outline='green', width=3)
            # #draw.rectangle(gaze_box[0][:4].tolist(), fill=None, outline='red', width=3)
            # # draw.rectangle(head_box_large[0].tolist(), fill=None, outline='red', width=3)
            # # draw.rectangle([eye_x * 224 - 40, eye_y * 224 - 40, eye_x * 224 + 40, eye_y * 224 + 40],fill=None, outline='red', width=3)
            # draw.ellipse([gaze_x*224 - 2, gaze_y*224 - 2, gaze_x*224 + 2, gaze_y*224 + 2],fill='red', outline=None)
            # draw.ellipse([eye_x*224- 2, eye_y*224 - 2, eye_x*224 + 2, eye_y*224 + 2], fill='red',outline=None)
            # draw.line([eye_x*224, eye_y*224, gaze_x*224, gaze_y*224],fill='yellow',width=2)
            # # im_with_hm = utils.generate_attention_map(im, gaze_cone)
            # # im_with_hm.show()
            # # file_path = os.path.join(output_folder, f"{file_name}")
            # im.save(file_path)
            # x=0

# ############################################################################
#             draw = ImageDraw.Draw(im)
#
#             for item in annos:
            
#                 bbox = np.array(item['bbox']).astype(int)
#                 coordinates = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
#                 draw.rectangle(coordinates, fill=None, outline='blue', width=3)
#
#                 bbox = np.array(bbox)
#                 bbox = bbox.tolist()
#                 draw.rectangle(bbox, fill=None, outline='green', width=3)
#             # draw.rectangle(head_box[0].tolist(), fill=None, outline='red', width=3)
#             draw.rectangle(gaze_box[0].tolist(), fill=None, outline='red', width=3)
#             
#             draw.ellipse([gaze_x * 224 - 2, gaze_y * 224 - 2, gaze_x * 224 + 2, gaze_y * 224 + 2], fill='red',
#                          outline=None)
#            
#             draw.ellipse([eye_x * 224 - 2, eye_y * 224 - 2, eye_x * 224 + 2, eye_y * 224 + 2], fill='red', outline=None)
#             draw.line([eye_x * 224, eye_y * 224, gaze_x * 224, gaze_y * 224], fill='yellow', width=2)
#             # im_with_hm = utils.generate_attention_map(im, gaze_cone)
#             # im_with_hm.show()
#             if im.mode == 'RGBA':
#                 im = im.convert('RGB')
#             im.show()
#             im.save('1.jpg')
#             x = 0
            ######################################################

            instances = utils.annotations_to_instances_goo(annos, (224, 224))
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict