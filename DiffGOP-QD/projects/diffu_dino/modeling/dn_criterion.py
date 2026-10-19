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

import torch

from detrex.utils import get_world_size, is_dist_avail_and_initialized

from .two_stage_criterion import TwoStageCriterion
from torch import nn

class DINOCriterion(TwoStageCriterion):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def forward(self, outputs, targets, gt_gaze_heatmap, gaze_outputs, normal_box, dn_metas=None, indices=None, t_weight=None, **kwargs):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = super(DINOCriterion, self).forward(outputs, targets, indices, t_weight, **kwargs)
        # import pdb;pdb.set_trace()
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        gaze_heatmap_pred = gaze_outputs.squeeze(1)

        # GOO Loss
        # l2 loss computed only for inside case
        gaze_heatmaps = torch.stack(gt_gaze_heatmap, dim=0)
        mse_loss = nn.MSELoss(reduce=False)
        l2_loss = mse_loss(gaze_heatmap_pred, gaze_heatmaps) * 1000
        l2_loss = torch.mean(l2_loss, dim=1)
        l2_loss = torch.mean(l2_loss, dim=1)
        gaze_loss = torch.mean(l2_loss)
        losses.update({'loss_gaze': gaze_loss})

        # box_energy_loss = compute_energy_loss(normal_box, gaze_heatmap_pred)
        # losses.update({'loss_energy': box_energy_loss})

        aux_num = 0
        if "aux_outputs" in outputs:
            aux_num = len(outputs["aux_outputs"])
        dn_losses = self.compute_dn_loss(dn_metas, targets, aux_num, num_boxes)
        losses.update(dn_losses)
        self

        return losses

    def compute_dn_loss(self, dn_metas, targets, aux_num, num_boxes):
        """
        compute dn loss in criterion
        Args:
            dn_metas: a dict for dn information
            training: training or inference flag
            aux_num: aux loss number
            focal_alpha:  for focal loss
        """
        losses = {}
        if dn_metas and "output_known_lbs_bboxes" in dn_metas:
            output_known_lbs_bboxes, dn_num, single_padding = (
                dn_metas["output_known_lbs_bboxes"],
                dn_metas["dn_num"],
                dn_metas["single_padding"],
            )
            dn_idx = []
            for i in range(len(targets)):
                if len(targets[i]["labels"]) > 0:
                    t = torch.arange(0, len(targets[i]["labels"])).long().to(targets[i]["labels"].device)
                    t = t.unsqueeze(0).repeat(dn_num, 1)
                    tgt_idx = t.flatten()
                    output_idx = (
                        torch.tensor(range(dn_num)) * single_padding
                    ).long().to(targets[i]["labels"].device).unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().to(targets[i]["labels"].device)

                dn_idx.append((output_idx, tgt_idx))
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if "labels" in loss:
                    kwargs = {"log": False}
                l_dict.update(
                    self.get_loss(
                        loss, output_known_lbs_bboxes, targets, dn_idx, num_boxes * dn_num,  None, **kwargs
                    )
                )

            l_dict = {k + "_dn": v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            losses["loss_bbox_dn"] = torch.as_tensor(0.0).to(targets[0]["labels"].device)
            losses["loss_giou_dn"] = torch.as_tensor(0.0).to(targets[0]["labels"].device)
            losses["loss_class_dn"] = torch.as_tensor(0.0).to(targets[0]["labels"].device)

        for i in range(aux_num):
            # dn aux loss
            l_dict = {}
            if dn_metas and "output_known_lbs_bboxes" in dn_metas:
                output_known_lbs_bboxes_aux = output_known_lbs_bboxes["aux_outputs"][i]
                for loss in self.losses:
                    kwargs = {}
                    if "labels" in loss:
                        kwargs = {"log": False}
                    l_dict.update(
                        self.get_loss(
                            loss,
                            output_known_lbs_bboxes_aux,
                            targets,
                            dn_idx,
                            num_boxes * dn_num,
                            None,
                            **kwargs,
                        )
                    )
                l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
            else:
                l_dict["loss_bbox_dn"] = torch.as_tensor(0.0).to(targets[0]["labels"].device)
                l_dict["loss_giou_dn"] = torch.as_tensor(0.0).to(targets[0]["labels"].device)
                l_dict["loss_class_dn"] = torch.as_tensor(0.0).to(targets[0]["labels"].device)
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
            losses.update(l_dict)
        return losses


import math
import torch

# def compute_energy_loss(box, heatmap, heatmap_size=64):
#     """
#     box: Tensor, shape [B, 4], format = [x1, y1, x2, y2], normalized to [0, 1]
#     heatmap: Tensor, shape [B, H, W] or [B, 1, H, W]
#     """
#     if heatmap.dim() == 4:
#         heatmap = heatmap.squeeze(1)
#
#     batch_size = heatmap.size(0)
#     # power = 0.0
#     power = torch.tensor(0.0, device=heatmap.device)
#     for i in range(batch_size):
#         cur_box = box[i]
#         cur_heatmap = heatmap[i]
#
#         x1, y1, x2, y2 = cur_box[0], cur_box[1], cur_box[2], cur_box[3]
#
#         # 映射到 heatmap 坐标
#         xmin = math.floor(x1.item() * heatmap_size)
#         ymin = math.floor(y1.item() * heatmap_size)
#         xmax = math.ceil(x2.item() * heatmap_size)
#         ymax = math.ceil(y2.item() * heatmap_size)
#
#         # 裁剪到合法范围
#         xmin = max(0, min(xmin, heatmap_size - 1))
#         ymin = max(0, min(ymin, heatmap_size - 1))
#         xmax = max(0, min(xmax, heatmap_size - 1))
#         ymax = max(0, min(ymax, heatmap_size - 1))
#
#         # 防止非法框
#         if xmax < xmin or ymax < ymin:
#             continue
#
#         box_w = xmax - xmin + 1
#         box_h = ymax - ymin + 1
#
#         region_power = torch.sum(cur_heatmap[ymin:ymax + 1, xmin:xmax + 1])
#         power = power + region_power / (box_w * box_h)
#
#     if power < 0:
#         # energy_rate = 0.99
#         energy_rate = torch.tensor(0.99, device=heatmap.device)
#     else:
#         energy_rate = 1 - power / batch_size
#
#     return energy_rate
import math
import torch

def compute_energy_loss(box, heatmap, heatmap_size=64):
    """
    box: Tensor [B,4], format [x1,y1,x2,y2], normalized to [0,1]
    heatmap: Tensor [B,H,W] or [B,1,H,W]
    """
    if heatmap.dim() == 4:
        heatmap = heatmap.squeeze(1)

    # 先把热图压到非负范围
    heatmap = torch.sigmoid(heatmap)

    batch_size = heatmap.size(0)
    loss = torch.tensor(0.0, device=heatmap.device)

    for i in range(batch_size):
        cur_box = box[i]
        cur_heatmap = heatmap[i]

        x1, y1, x2, y2 = cur_box[0], cur_box[1], cur_box[2], cur_box[3]

        xmin = max(0, min(math.floor(x1.item() * heatmap_size), heatmap_size - 1))
        ymin = max(0, min(math.floor(y1.item() * heatmap_size), heatmap_size - 1))
        xmax = max(0, min(math.ceil(x2.item() * heatmap_size), heatmap_size - 1))
        ymax = max(0, min(math.ceil(y2.item() * heatmap_size), heatmap_size - 1))

        if xmax < xmin or ymax < ymin:
            continue

        inside = torch.sum(cur_heatmap[ymin:ymax + 1, xmin:xmax + 1])
        total = torch.sum(cur_heatmap) + 1e-6

        inside_ratio = inside / total
        loss = loss + (1.0 - inside_ratio)

    return loss / batch_size


