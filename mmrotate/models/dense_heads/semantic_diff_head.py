# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean
from mmcv.cnn import ConvModule
from mmcv.cnn import (ConvModule, caffe2_xavier_init, constant_init, is_norm,
                      normal_init)

from mmdet.models.necks.dilated_encoder import Bottleneck
from ..builder import ROTATED_HEADS, build_loss
from mmdet.core.anchor.point_generator import MlvlPointGenerator
from .rotated_fcos_head import RotatedFCOSHead
from mmrotate.core.bbox.iou_calculators import RBboxOverlaps2D
import os
import math
import csv

INF = 1e8

class CatWiseModule(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 cls_conv_channels,
                 add_identity=True,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(CatWiseModule, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_conv_channels = cls_conv_channels
        self.out_channels = num_classes * cls_conv_channels
        self.add_identity = add_identity
        self.norm_cfg = norm_cfg

        self.query = nn.Linear(in_channels, self.out_channels)
        self.key = nn.Linear(in_channels, self.out_channels)
        self.value = nn.Linear(in_channels, self.out_channels)
        self.cls_conv = ConvModule(
            self.in_channels, self.num_classes * self.cls_conv_channels, 1, norm_cfg=norm_cfg)

    def forward(self, x):
        b = x.size(0)
        inter_feat = self.cls_conv(x)
        weight = F.adaptive_avg_pool2d(x, (1, 1)).squeeze()
        Q = self.query(weight).view(b, self.num_classes, self.cls_conv_channels, 1)
        K = self.key(weight).view(b, self.num_classes, self.cls_conv_channels, 1)
        V = self.value(weight).view(b, self.num_classes, self.cls_conv_channels, 1).transpose(-2, -1)
        scores = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (1 ** 0.5), dim=-1)
        attn = torch.matmul(V, scores).contiguous().view(b, self.out_channels, 1, 1)
        out = inter_feat * attn
        if self.add_identity:
            out = out + inter_feat
        return out

class ReweightModule(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 cls_conv_channels,
                 add_identity=True,
                 r=2,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(ReweightModule, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_conv_channels = cls_conv_channels
        self.out_channels = num_classes * cls_conv_channels
        self.add_identity = add_identity
        self.norm_cfg = norm_cfg
        self.cls_conv = ConvModule(
            self.in_channels, self.out_channels, 1, norm_cfg=norm_cfg)
        self.cat_conv = ConvModule(
            self.in_channels, self.out_channels, 1, norm_cfg=norm_cfg)
        self.fc1 = nn.Linear(self.out_channels, self.out_channels // r)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.out_channels // r, self.out_channels)

    def forward(self, x):
        b = x.size(0)
        inter_feat = self.cls_conv(x)
        global_weight = F.adaptive_avg_pool2d(x, (1, 1)).view(b, -1, 1, 1)
        weight = self.cat_conv(global_weight).view(b, -1)
        weight = self.relu(self.fc1(weight))
        weight = self.fc2(weight)
        weight = torch.sigmoid(weight).view(b, -1, 1, 1)
        out = inter_feat * weight
        if self.add_identity:
            out = out + inter_feat
        return out

@ROTATED_HEADS.register_module()
class SemanticDiffHead(RotatedFCOSHead):
    """
    Semantic-specific Differentiation Head
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 strides=[8, 16, 32, 64],
                 regress_ranges=((-1, 128), (128, 256), (256, 512), (512, INF)),
                 cls_kernel=3,
                 cls_conv_channels=32,
                 dilation=2,
                 ins_norm_on_reg=False,
                 is_act=False,
                 in_conv='reweight',  # ['conv', 'reweight', 'catwise']
                 with_res=True,
                 add_identity=True,
                 with_static=False,
                 norm_on_cls='gn', # 'bn', None
                 encoder_norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),  # dict(type='BN', requires_grad=True)
                 loss_filter_cfg=dict(
                     type='focal',  # ['con', 'margin', 'focal', None]
                     loss_weight=1.0,
                     gamma=2.0,
                     alpha=0.25,
                     tau=0.5,
                     pos_margin=0.2,
                     neg_margin=0.3),
                 score=1.0,
                 cnt_target_type='sqrt',  # ['sqrt', 'qurt']
                 cnt_gamma=3,
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_kernel = cls_kernel
        self.cls_conv_channels = cls_conv_channels
        self.dilation = dilation
        self.ins_norm_on_reg = ins_norm_on_reg
        self.is_act = is_act
        self.norm_on_cls = norm_on_cls
        self.encoder_norm_cfg = encoder_norm_cfg
        self.loss_filter = loss_filter_cfg.pop('type')
        if self.loss_filter == 'focal':
            self.loss_filter_gamma = loss_filter_cfg.pop('gamma')
            self.loss_filter_alpha = loss_filter_cfg.pop('alpha')
            self.loss_filter_pos_margin = loss_filter_cfg.pop('pos_margin')
            self.loss_filter_neg_margin = loss_filter_cfg.pop('neg_margin')
        self.loss_filter_weight = loss_filter_cfg.pop('loss_weight')
        self.tau = loss_filter_cfg.pop('tau')
        self.score = score
        self.with_res = with_res
        self.with_static = with_static
        self.cnt_target_type = cnt_target_type
        self.cnt_gamma = cnt_gamma
        self.iou_calculator = RBboxOverlaps2D()
        self.epoch = 0  # which would be update in SetEpochInfoHook!
        super().__init__(
            num_classes,
            in_channels,
            init_cfg=init_cfg,
            **kwargs)
        if in_conv == 'conv':
            self.in_conv = ConvModule(
                self.in_channels, self.num_classes * self.cls_conv_channels, 1, norm_cfg=self.encoder_norm_cfg)
        elif in_conv == 'catwise':
            self.in_conv = CatWiseModule(num_classes, in_channels, cls_conv_channels, add_identity)
        else:
            self.in_conv = ReweightModule(num_classes, in_channels, cls_conv_channels, add_identity)
        self.regress_ranges = regress_ranges
        self.strides = strides
        self.prior_generator = MlvlPointGenerator(strides)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.conv_angle = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        if self.is_scale_angle:
            self.scale_angle = Scale(1.0)
        self.out_conv = ConvModule(
            self.num_classes * self.cls_conv_channels, self.in_channels, 1, norm_cfg=self.encoder_norm_cfg)
        if self.norm_on_cls:
            self.cls_norms = [nn.BatchNorm2d(self.cls_conv_channels) for i in range(self.num_classes)] if self.norm_on_cls == 'bn' \
                else [nn.GroupNorm(num_groups=32, num_channels=self.cls_conv_channels) for i in range(self.num_classes)]
        self.relu = nn.ReLU(inplace=True)
        self.sem_kernel = nn.Parameter(
            torch.randn(self.num_classes, self.cls_kernel ** 2 * self.cls_conv_channels ** 2))
        self.linear_layer = nn.Linear(
            self.cls_kernel ** 2 * self.cls_conv_channels ** 2, self.in_channels)
        self.class_diff_encoder = Bottleneck(
            self.num_classes * self.cls_conv_channels, self.in_channels,  dilation=self.dilation, norm_cfg=self.encoder_norm_cfg)
        self.fusion_conv = ConvModule(
            self.num_classes * self.cls_conv_channels, self.in_channels, 1, norm_cfg=self.encoder_norm_cfg)
        self.class_fused_encoder = Bottleneck(
            self.in_channels, self.in_channels, dilation=self.dilation, norm_cfg=self.encoder_norm_cfg)

    def init_weights(self):
        if self.norm_on_cls:
            for m in self.cls_norms:
                constant_init(m, 1)
        for m in self.class_diff_encoder.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)
        for m in self.class_fused_encoder.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      **kwargs):
        
        results = self(x)
        flatten_img_feats = []
        for img_id, _ in enumerate(img_metas):
            flatten_img_feat = [x[i][img_id].permute(1, 2, 0).reshape(-1, self.feat_channels)
                                for i in range(len(x))]
            flatten_img_feat = torch.cat(flatten_img_feat, dim=0)
            flatten_img_feats.append(flatten_img_feat)
        if gt_labels is None:
            loss_inputs = results + (gt_bboxes, flatten_img_feats, img_metas)
        else:
            loss_inputs = results + (gt_bboxes, gt_labels, flatten_img_feats, img_metas)
        losses, pos_signs_list = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        if self.loss_filter is not None:
            batch_size = len(img_metas)
            loss_filter = torch.zeros_like(losses['loss_cls_ssd'])
            for bs in range(batch_size):
                pos_inds = torch.where(pos_signs_list[bs] != -1)[0]
                if pos_inds.size(0) > 0:
                    pos_feat = F.normalize(flatten_img_feats[bs][pos_inds], dim=-1)
                    pos_mask = pos_signs_list[bs].new_zeros((self.num_classes, pos_inds.size(0)))
                    col_ind = torch.arange(pos_inds.size(0))
                    row_ind = pos_signs_list[bs][pos_inds]
                    pos_mask[row_ind, col_ind] = 1

                    predict = F.normalize(self.linear_layer(self.sem_kernel), dim=-1)
                    # filter loss
                    cls_filter_loss = self.focal_loss(predict, pos_feat, pos_mask)
                    loss_filter = loss_filter + self.loss_filter_weight * cls_filter_loss
            losses.update(dict(loss_filter=loss_filter))

        class_diff_feats = []
        for feat in x:
            in_feats = self.in_conv(feat)
            class_diff_feat = []
            cur_cls_feats = torch.split(in_feats, self.cls_conv_channels, dim=1)
            for cls in range(self.num_classes):
                cur_cls_kernel = self.sem_kernel[cls].view(self.cls_conv_channels, self.cls_conv_channels, 3, 3)
                cls_conv_feat = F.conv2d(cur_cls_feats[cls], cur_cls_kernel, padding=1)
                if self.norm_on_cls:
                    cls_conv_feat = self.cls_norms[cls].cuda()(cls_conv_feat)
                if self.is_act:
                    cls_act_feat = self.relu(cls_conv_feat)
                    class_diff_feat.append(cls_act_feat)
                else:
                    class_diff_feat.append(cls_conv_feat)
            class_diff_feat = torch.cat(class_diff_feat, dim=1)  # tensor_dim=b,cls*multi,h,w

            class_diff_feat = self.class_diff_encoder(in_feats + class_diff_feat)
            class_diff_feat = self.fusion_conv(class_diff_feat)
            class_diff_feat = self.class_fused_encoder(class_diff_feat + feat) if self.with_res \
                else self.class_fused_encoder(class_diff_feat)

            class_diff_feats.append(class_diff_feat)
        return losses, class_diff_feats

    def focal_loss(self, predict, pos_feat, labels):
        num_pos = pos_feat.size(0)
        sim_logits = torch.einsum('ct, nt->cn', predict, pos_feat)
        labels[labels == 0] = -1
        pos_logits = torch.clamp(sim_logits + self.loss_filter_pos_margin, -1, 1)
        neg_logits = torch.clamp(sim_logits + self.loss_filter_neg_margin, -1, 1)
        sim_logits = torch.where(labels == 1, pos_logits, neg_logits)
        sim_logits = (sim_logits + 1) / 2
        labels = (labels + 1) / 2
        bce_loss = F.binary_cross_entropy_with_logits(sim_logits, labels.float(), reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.loss_filter_alpha * (1 - pt) ** self.loss_filter_gamma * bce_loss
        loss = focal_loss.sum() / num_pos
        return loss

    def margin_loss(self, predict, pos_feat, pos_signs):
        sim_logits = torch.einsum('ct, nt->cn', predict, pos_feat)
        margin_loss = sim_logits.new_zeros((self.num_classes, 1))
        pos_negs = []
        for i, sim_logit in enumerate(sim_logits):
            pos_idx = torch.nonzero(pos_signs[i] == 1).squeeze()
            neg_idx = torch.nonzero(pos_signs[i] == 0).squeeze()
            if torch.numel(pos_idx) > 0:
                pos_min = torch.min(sim_logit[pos_idx])
                if torch.numel(neg_idx) > 0:
                    neg_max = torch.max(sim_logit[neg_idx])
                else:
                    neg_max = sim_logits.new_zeros(1)
            else:
                neg_max = torch.max(sim_logit[neg_idx])
                pos_min = sim_logits.new_zeros(1)
            pos_neg = torch.cat([pos_min.reshape(1, 1), neg_max.reshape(1, 1), ], dim=1)  # [1, 2]
            pos_negs.append(pos_neg)
        pos_negs = torch.cat(pos_negs, dim=0)
        margins = sim_logits.new_ones((self.num_classes, 1)) * self.tau
        pos_neg_margins = torch.cat([pos_negs, margins], dim=1)
        distance = pos_neg_margins[:, 0] - pos_neg_margins[:, 1] - pos_neg_margins[:, 2]
        margin_loss = torch.max(margin_loss, -distance.view(-1, 1))
        loss = margin_loss.sum()
        if self.margin_cls_mean:
            loss = loss / self.num_classes
        return loss

    def forward(self, feats):
        results = multi_apply(self.forward_single, feats, self.scales, self.strides)
        return results

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             angle_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             flatten_feats,
             img_metas,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) \
               == len(angle_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, bbox_targets, angle_targets, pos_signs_list, gt_ids = self.get_targets(
             cls_scores, bbox_preds, angle_preds, img_metas, all_level_points,
            gt_bboxes, gt_labels, flatten_feats)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for angle_pred in angle_preds]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)
        flatten_gt_ids = torch.cat(gt_ids)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # normalize the bbox weights
        if self.ins_norm_on_reg:
            norm_pos_centerness_targets = pos_centerness_targets.view(1, -1)
            pos_gt_ids = flatten_gt_ids[pos_inds]
            for i in range(gt_labels[0].numel()):
                cur_id = pos_gt_ids == i
                if cur_id.sum() == 0:
                    continue
                cur_max_cnt = norm_pos_centerness_targets[0, cur_id].max()
                norm_pos_centerness_targets[0, cur_id] = norm_pos_centerness_targets[0, cur_id] / cur_max_cnt
            # for centerness weighted iou loss
            reg_weight = norm_pos_centerness_targets.view(-1)
            reg_avg_factor = max(
                reduce_mean(norm_pos_centerness_targets.sum().detach()), 1e-6)
        else:
            reg_weight = pos_centerness_targets.view(-1)
            reg_avg_factor = max(
                reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            if self.separate_angle:
                bbox_coder = self.h_bbox_coder
            else:
                bbox_coder = self.bbox_coder
                pos_bbox_preds = torch.cat([pos_bbox_preds, pos_angle_preds],
                                           dim=-1)
                pos_bbox_targets = torch.cat(
                    [pos_bbox_targets, pos_angle_targets], dim=-1)
            pos_decoded_bbox_preds = bbox_coder.decode(pos_points,
                                                       pos_bbox_preds)
            pos_decoded_target_preds = bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=reg_weight,
                avg_factor=reg_avg_factor)
            if self.separate_angle:
                loss_angle = self.loss_angle(
                    pos_angle_preds, pos_angle_targets, avg_factor=num_pos)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            if self.separate_angle:
                loss_angle = pos_angle_preds.sum()

        if self.separate_angle:
            return dict(
                loss_cls_ssd=loss_cls,
                loss_bbox_ssd=loss_bbox,
                loss_angle_ssd=loss_angle,
                loss_centerness_ssd=loss_centerness), pos_signs_list
        else:
            return dict(
                loss_cls_ssd=loss_cls,
                loss_bbox_ssd=loss_bbox,
                loss_centerness_ssd=loss_centerness), pos_signs_list

    def get_targets(self, cls_scores, bbox_preds, angle_preds, img_metas, points,
                    gt_bboxes_list, gt_labels_list, flatten_feats_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]
        cls_scores_list = []
        bbox_preds_list = []
        angle_preds_list = []
        for img_id, _ in enumerate(img_metas):
            img_cls_score = [cls_scores[i][img_id] for i in range(num_levels)]
            cls_scores_list.append(img_cls_score)
            img_bbox_pred = [bbox_preds[i][img_id] for i in range(num_levels)]
            bbox_preds_list.append(img_bbox_pred)
            img_angle_pred = [angle_preds[i][img_id] for i in range(num_levels)]
            angle_preds_list.append(img_angle_pred)

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, angle_targets_list, pos_signs_list, gt_ids_list = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            angle_preds_list,
            img_metas,
            gt_bboxes_list,
            gt_labels_list,
            flatten_feats_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [
            labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0) for bbox_targets in bbox_targets_list]
        angle_targets_list = [
            angle_targets.split(num_points, 0) for angle_targets in angle_targets_list]
        gt_ids_list = [
            gt_ids.split(num_points, 0) for gt_ids in gt_ids_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_angle_targets = []
        concat_lvl_gt_ids = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            angle_targets = torch.cat(
                [angle_targets[i] for angle_targets in angle_targets_list])
            concat_lvl_gt_ids.append(
                torch.cat([gt_ids[i] for gt_ids in gt_ids_list]))
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_angle_targets.append(angle_targets)
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_angle_targets, pos_signs_list, concat_lvl_gt_ids)

    def _get_target_single(self, img_cls_scores, bbox_preds, angle_preds, img_meta, gt_bboxes,
                           gt_labels, flatten_feats, points, regress_ranges, num_points_per_lvl):
        """Compute regression, classification and angle targets for a single
        image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 1))
        # classification scores and points will be used later
        mlvl_cls_scores = [cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid() #____________
                              for cls_score in img_cls_scores]
        flatten_cls_scores = torch.cat(mlvl_cls_scores)
        img_points = points.clone()
        gt_ares = gt_bboxes[:, -2] * gt_bboxes[:, -3] / (img_meta['scale_factor'][0] * img_meta['scale_factor'][1])

        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            stride = offset.new_zeros(offset.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
            inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask,
                                                    inside_gt_bbox_mask)

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        angle_targets = gt_angle[range(num_points), min_area_inds]
        gt_ids = min_area_inds.clone()
        gt_ids[min_area == INF] = num_gts + 1

        # mine positive samples for subsequent kernel embedding
        img_shape = img_meta['img_shape']
        bbox_tars = bbox_targets.clone()
        angle_tars = angle_targets.clone()
        if self.norm_on_bbox:
            norm_stride = [gt_labels.new_ones(num_lvl_points) * stride for (num_lvl_points, stride) in
                           zip(num_points_per_lvl, self.strides)]
            norm_stride = torch.cat(norm_stride).view(-1, 1)
            bbox_tars = bbox_tars / norm_stride
        ltrba_targets = torch.cat([bbox_tars, angle_tars], dim=1)
        # Note decoded_targets is different from gt_bboxes!!!
        decoded_targets = self.bbox_coder.decode(img_points, ltrba_targets, max_shape=img_shape)
        bbox_preds = [bp.permute(1, 2, 0).view(-1, 4) for bp in bbox_preds]
        bbox_preds = torch.cat(bbox_preds, dim=0)
        angle_preds = [ap.permute(1, 2, 0).view(-1, 1) for ap in angle_preds]
        angle_preds = torch.cat(angle_preds, dim=0)
        ltrba_preds = torch.cat([bbox_preds, angle_preds], dim=1)
        decoded_preds = self.bbox_coder.decode(img_points, ltrba_preds, max_shape=img_shape)
        pos_inds = torch.nonzero(labels != self.num_classes, as_tuple=False).view(-1)
        pos_targets = decoded_targets[pos_inds]
        pos_preds = decoded_preds[pos_inds]
        overlaps = self.iou_calculator(pos_targets, pos_preds, is_aligned=True)

        pos_signs = -1 * torch.ones_like(labels)
        pos_min_area_inds = min_area_inds[pos_inds]

        if self.with_static:
            write_areas = [round(area, 0) for area in gt_ares.tolist()]
            write_inds = pos_inds.tolist()
            write_gts = pos_min_area_inds.tolist()
            write_confs = [round(conf, 4) for conf in flatten_cls_scores[pos_inds, gt_labels[pos_min_area_inds]].tolist()]
            write_ious = [round(iou, 4) for iou in overlaps.tolist()]
            write_pth = './sta/ssd_score/' + img_meta['ori_filename'].replace('.png', '.csv')
            if self.epoch > 0:
                write_data = [write_inds, write_gts, write_confs, write_ious]
            else:
                write_data = [write_areas, write_inds, write_gts, write_confs, write_ious]
            self.write_csv(write_pth, write_data)

        for g in range(num_gts):
            cur_gt_signs = pos_min_area_inds == g
            if cur_gt_signs.sum() == 0:
                continue
            cur_gt_pos_inds = pos_inds[cur_gt_signs]
            ious = overlaps[cur_gt_signs]
            confs = flatten_cls_scores[cur_gt_pos_inds, gt_labels[g]]
            scores = ious + confs
            val_signs = cur_gt_pos_inds[scores > self.score]
            pos_signs[val_signs] = gt_labels[g]  # semantic-level labels instead of instance-level labels
        return labels, bbox_targets, angle_targets, pos_signs, gt_ids

    def centerness_target(self, pos_bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        if self.cnt_target_type == 'sqrt':
            return torch.sqrt(centerness_targets)
        elif self.cnt_target_type == 'qurt':
            return torch.sqrt(torch.sqrt(centerness_targets))
        elif self.cnt_target_type == 'sin':
            return torch.pow(torch.sin(math.pi / 2 * centerness_targets), 1 / self.cnt_gamma)
        else:
            raise NotImplementedError

    def write_csv(self, path, data):
        with open(path, 'a+', newline='\n') as f:
            csv_write = csv.writer(f)
            csv_write.writerows(data)

    def forward_test(self, feats, img_metas):
        class_diff_feats = []
        for feat in feats:
            in_feats = self.in_conv(feat)
            class_diff_feat = []
            cur_cls_feats = torch.split(in_feats, self.cls_conv_channels, dim=1)
            for cls in range(self.num_classes):
                cur_cls_kernel = self.sem_kernel[cls].view(self.cls_conv_channels, self.cls_conv_channels, 3, 3)
                cls_conv_feat = F.conv2d(cur_cls_feats[cls], cur_cls_kernel, padding=1)
                if self.norm_on_cls:
                    cls_conv_feat = self.cls_norms[cls].cuda()(cls_conv_feat)
                if self.is_act:
                    cls_act_feat = self.relu(cls_conv_feat)
                    class_diff_feat.append(cls_act_feat)
                else:
                    class_diff_feat.append(cls_conv_feat)
            class_diff_feat = torch.cat(class_diff_feat, dim=1)  # tensor_dim=b,cls*multi,h,w

            class_diff_feat = self.class_diff_encoder(in_feats + class_diff_feat)
            class_diff_feat = self.fusion_conv(class_diff_feat)
            class_diff_feat = self.class_fused_encoder(class_diff_feat + feat) if self.with_res \
                else self.class_fused_encoder(class_diff_feat)

            class_diff_feats.append(class_diff_feat)
        return tuple(class_diff_feats)

