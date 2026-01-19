# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from .base import RotatedBaseDetector
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from mmrotate.core import rbbox2result

@ROTATED_DETECTORS.register_module()
class RotatedFSA(RotatedBaseDetector):
    def __init__(self,
                 backbone,
                 neck,
                 ssd_head,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RotatedFSA, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        if ssd_head is not None:
            self.ssd_head = build_head(ssd_head)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def set_epoch(self, epoch):
        self.bbox_head.epoch = epoch
        if self.ssd_head is not None:
            self.ssd_head.epoch = epoch

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      **kwargs):
        x = self.extract_feat(img)

        losses = dict()
        # Semantic-Specific Differentiation loss
        ssd_losses, class_diff_feats = self.ssd_head.forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            **kwargs)
        losses.update(ssd_losses)
        # Bbox loss
        det_losses = self.bbox_head.forward_train(class_diff_feats, img_metas,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore,
                                                 **kwargs)
        losses.update(det_losses)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        x = self.ssd_head.forward_test(x, img_metas)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list]
        return bbox_results

    # TODO: make it works
    def aug_test(self, imgs, img_metas, rescale=False):
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

