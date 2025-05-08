_base_ = 'faster-rcnn_r50_fpn_1x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=4
        )
    )
)

data_root = '/root/'
metainfo = {
    'classes': ('Government-cars', 'car' , 'truck', 'bus' ),
    #'palette': [
    #    (220, 20, 60),
    #]
}
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/annotations/train.json',
        data_prefix=dict(img='train/images/')))

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/annotations/val.json',
        data_prefix=dict(img='val/images/')))

test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/annotations/val.json',
        data_prefix=dict(img='val/images/')))

val_evaluator = dict(ann_file=data_root + 'val/annotations/val.json')
test_evaluator = dict(ann_file=data_root + 'val/annotations/val.json')