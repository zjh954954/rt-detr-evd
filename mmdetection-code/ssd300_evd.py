_base_ = [
    '../_base_/models/ssd300.py', '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

# dataset settings
# 类别数
model = dict(
    bbox_head=dict(
        num_classes=4
    )
)

# 数据集路径和类别信息
data_root = '/root/'
metainfo = {
    'classes': ('Government-cars', 'car', 'truck', 'bus'),
}

input_size = 300
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean={{_base_.model.data_preprocessor.mean}},
        to_rgb={{_base_.model.data_preprocessor.bgr_to_rgb}},
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        ann_file='train/annotations/train.json',
        data_prefix=dict(img='train/images/'),
        metainfo=metainfo
    )
)

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        ann_file='val/annotations/val.json',
        data_prefix=dict(img='val/images/'),
        metainfo=metainfo
    )
)

test_dataloader = val_dataloader

# 修改 bbox_head 部分
bbox_head=dict(
    type='mmdet.SSDHead',
    in_channels=(512, 1024, 512, 256, 256, 256),
    num_classes=4,
    anchor_generator=dict(
        type='mmdet.SSDAnchorGenerator',
        scale_major=False,
        input_size=300,
        strides=[8, 16, 32, 64, 100, 300],
        ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        basesize_ratio_range=(0.15, 0.9)),
    bbox_coder=dict(
        type='mmdet.DeltaXYWHBBoxCoder',
        target_means=[0.0, 0.0, 0.0, 0.0],
        target_stds=[0.1, 0.1, 0.2, 0.2]),
    # 添加必要的损失函数定义
    loss_cls=dict(
        type='mmdet.CrossEntropyLoss',
        use_sigmoid=True,
        loss_weight=1.0),
    loss_bbox=dict(
        type='mmdet.SmoothL1Loss',
        beta=1.0,
        loss_weight=1.0),
    # 确保添加以下参数
    prior_generator=dict(
        type='mmdet.SSDAnchorGenerator',
        scale_major=False,
        input_size=300,
        strides=[8, 16, 32, 64, 100, 300],
        ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        basesize_ratio_range=(0.15, 0.9)),
)
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=2e-3, momentum=0.9, weight_decay=5e-4))

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)

# 评估器
val_evaluator = dict(ann_file=data_root + 'val/annotations/val.json')
test_evaluator = val_evaluator
