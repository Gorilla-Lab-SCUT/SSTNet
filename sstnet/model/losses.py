# Copyright (c) Gorilla-Lab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

import pointgroup_ops
import gorilla
from gorilla.losses import dice_loss_multi_calsses, iou_guided_loss
from .func_helper import *


@gorilla.LOSSES.register_module()
class SSTLoss(nn.Module):
    def __init__(self,
                 ignore_label: int,
                 fusion_epochs: int=128,
                 score_epochs: int=160,
                 bg_thresh: float=0.25,
                 fg_thresh: float=0.75,
                 semantic_dice: bool=True,
                 loss_weight: List[float]=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]):
        super().__init__()
        self.ignore_label = ignore_label
        self.fusion_epochs = fusion_epochs
        self.score_epochs = score_epochs
        self.bg_thresh = bg_thresh
        self.fg_thresh = fg_thresh
        self.semantic_dice = semantic_dice
        self.loss_weight = loss_weight

        #### criterion
        self.semantic_criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_label)
        self.score_criterion = nn.BCELoss(reduction="none")

    def forward(self, loss_inp, epoch):
        loss_out = {}
        fusion_flag = (epoch > self.fusion_epochs)
        prepare_flag = (epoch > self.score_epochs)

        """semantic loss"""
        semantic_scores, semantic_labels = loss_inp["semantic_scores"]
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda

        semantic_loss = self.semantic_criterion(semantic_scores, semantic_labels)
        if self.semantic_dice:
            filter_ids = (semantic_labels != self.ignore_label)
            semantic_scores = semantic_scores[filter_ids]
            semantic_scores = F.softmax(semantic_scores, dim=-1)
            semantic_labels = semantic_labels[filter_ids]
            one_hot_labels = F.one_hot(semantic_labels, num_classes=20)
            semantic_loss += dice_loss_multi_calsses(semantic_scores, one_hot_labels).mean()
        loss_out["semantic_loss"] = (semantic_loss, semantic_scores.shape[0])

        """offset loss"""
        pt_offsets, coords, instance_info, instance_labels, instance_pointnum = loss_inp["pt_offsets"]
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 9), float32 tensor (meanxyz, minxyz, maxxyz)
        # instance_labels: (N), long
        # instance_pointnum: (total_num_inst), int

        gt_offsets = instance_info[:, 0:3] - coords   # [N, 3]
        pt_diff = pt_offsets - gt_offsets   # [N, 3]
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # [N]
        valid = (instance_labels != self.ignore_label)

        offset_norm_loss = torch.sum(pt_dist * valid) / (valid.sum() + 1e-6)

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)   # [N], float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
        pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)   # [N]
        offset_dir_loss = torch.sum(direction_diff * valid) / (valid.sum() + 1e-6)

        loss_out["offset_norm_loss"] = (offset_norm_loss, valid.sum())
        loss_out["offset_dir_loss"] = (offset_dir_loss, valid.sum())

        empty_flag = loss_inp["empty_flag"]
        """superpoint clustering loss"""
        if fusion_flag:
            fusion_scores, fusion_labels = loss_inp["fusion"]
            # fusion_scores: [num_superpoint - 1], float
            # fusion_labels: [num_superpoint - 1], float
            fusion_loss = F.binary_cross_entropy_with_logits(fusion_scores, fusion_labels)
            fusion_count = fusion_labels.shape[0]
            loss_out["fusion_loss"] = (fusion_loss, fusion_count)

        if "refine" in loss_inp and not empty_flag:
            """refine loss"""
            (refine_scores, refine_labels) = loss_inp["refine"]
            refine_loss = F.binary_cross_entropy_with_logits(refine_scores, refine_labels)
            refine_count = refine_labels.shape[0]
            loss_out["refine_loss"] = (refine_loss, refine_count)
        if prepare_flag and not empty_flag:
            proposals_idx, proposals_offset = loss_inp["proposals"]
            # scores: (num_prop, 1), float32
            # proposals_idx: (sum_points, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (num_prop + 1), int, cpu
            ious = pointgroup_ops.get_iou(proposals_idx[:, 1].int().cuda(),
                                          proposals_offset.int().cuda(),
                                          instance_labels,
                                          instance_pointnum.int()) # [num_prop, num_inst], float
            gt_ious, gt_inst_idxs = ious.max(1)  # [num_prop] float, long
            """score loss"""
            scores = loss_inp["proposal_scores"]
            gt_ious, gt_inst_idxs = ious.max(1)  # [num_prop] float, long
            score_loss = iou_guided_loss(scores.view(-1), gt_ious, self.fg_thresh, self.bg_thresh, use_sigmoid=False)
            score_loss = score_loss.mean()

            loss_out["score_loss"] = (score_loss, gt_ious.shape[0])

        """total loss"""
        # loss = fusion_loss
        loss = self.loss_weight[0] * semantic_loss + self.loss_weight[1] * offset_norm_loss + self.loss_weight[2] * offset_dir_loss

        if fusion_flag:
            loss += fusion_loss

        if prepare_flag and not empty_flag:
            loss += (self.loss_weight[3] * score_loss)

        if "refine" in loss_inp and not empty_flag:
            loss += refine_loss

        return loss, loss_out


