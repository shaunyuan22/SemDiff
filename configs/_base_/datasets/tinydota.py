# dataset settings
dataset_type = 'TinyDOTADataset'
data_root = 'path-to-tinydota'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        version='le90',
        ann_file=data_root + 'div/train/labelTxt',
        ori_ann_file=data_root + 'raw/train/labelTxt-Tiny',
        img_prefix=data_root + 'div/train/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        version='le90',
        ann_file=data_root + 'div/val-test/labelTxt',
        ori_ann_file=data_root + 'raw/val-test/labelTxt-Tiny',
        img_prefix=data_root + 'div/val-test/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        version='le90',
        ann_file=data_root + 'div/val-test/labelTxt',
        ori_ann_file=data_root + 'raw/val-test/labelTxt-Tiny',
        img_prefix=data_root + 'div/val-test/images/',
        pipeline=test_pipeline))
