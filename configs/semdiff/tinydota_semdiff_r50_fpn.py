_base_ = [
    '../_base_/datasets/tinydota.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

angle_version = 'le90'
find_unused_parameters=True
# model settings
model = dict(
    type='RotatedFSA',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P6
        num_outs=4,
        relu_before_extra_convs=True),
    ssd_head=dict(
        type='SemanticDiffHead',
        num_classes=8,
        in_channels=256,
        strides=[8, 16, 32, 64],
        regress_ranges=((-1, 128), (128, 256), (256, 512), (512, 3000)),
        stacked_convs=4,
        feat_channels=256,
        cls_kernel=3,
        cls_conv_channels=32,
        dilation=2,
        is_act=False,
        in_conv='conv',  # ['conv', 'reweight', 'catwise']
        with_res=True,
        add_identity=True,
        with_static=False,
        norm_on_cls='bn',
        ins_norm_on_reg=False,
        encoder_norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),  # dict(type='BN', requires_grad=True)
        loss_filter_cfg=dict(
            type='focal',  # ['con', 'margin', 'focal', None]
            loss_weight=0.1,
            gamma=2.0,
            alpha=0.25,
            tau=0.5,
            pos_margin=0.0,
            neg_margin=0.0),
        cnt_target_type='sin',
        cnt_gamma=3,
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.5),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=0.5),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5)),
    bbox_head=dict(
        type='AffinityAugHead',
        num_classes=8,
        in_channels=256,
        strides=[8, 16, 32, 64],
        ins_norm_on_reg=False,
        with_static=False,
        cnt_target_type='sin',
        cnt_gamma=3,
        regress_ranges=((-1, 128), (128, 256), (256, 512), (512, 3000)),
        stacked_convs=4,
        feat_channels=256,
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    # train=dict(filter_dense=300)
)

optimizer = dict(lr=0.0025)    # 0.0025 for batch size = 2
lr_config = dict(warmup_iters=2000)
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=4)
evaluation = dict(interval=12, metric='mAP')
# custom hooks
custom_hooks = [dict(type='SetEpochInfoHook')]
