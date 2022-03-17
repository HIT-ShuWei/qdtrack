_base_ = './qdtrack-frcnn_r50_fpn_12e_bdd100k.py'

model = dict(
    type='VPMTrack',
    # freeze_detector = True,
    freeze_loc_layer = True,
    roi_head=dict(
        type="SelfSupervisionRoIHead",
        track_roi_extractor=dict(
            roi_layer=dict(type='RoIAlign', output_size=27, sampling_ratio=0),
        ),
        track_head=dict(
            type='SelfSupervisionEmbedHead',
            roi_feat_size=27,
            num_regions = 3,
            loss_loc=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=0),
            loss_loc_ref = dict(type='CrossEntropyLoss', use_mask=True,loss_weight=0),
            loss_track=dict(
                     type='MultiPosCrossEntropyLoss', loss_weight=0.25),
        ),
    ),
    tracker=dict(
        type='SelfSupervisionEmbedTracker',
        match_metric='cosine',
    ),
)

dataset_type = 'BDDSelfSupervisionDatasetV'

data_root = 'data/bdd/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    # dict(type='SeqLoadAnnotations', with_bbox=True, with_ins_id=True, with_loc_map=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_ins_id=True, with_loc_map=False),
    dict(type='SeqResize', img_scale=(1296, 720), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='SeqDefaultFormatBundle'),
    dict(
        type='SeqCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_match_indices'],
        ref_prefix='ref'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1296, 720),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]
data = dict(
    train = [
        dict(
            type=dataset_type,
            # ann_file=data_root + 'labels/box_track_20/box_track_train_cocofmt.json',
            only_holistic = False,
            ann_file=data_root + 'labels/box_track_20/box_track_train_remap_cocofmt_add_atri.json',
            img_prefix=data_root + 'images/track/train/',
            key_img_sampler=dict(interval=1),
            ref_img_sampler=dict(num_ref_imgs=1, scope=3, method='uniform'),
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            only_holistic = False,
            load_as_video = False,
            ann_file=data_root + 'labels/det_20/det_train_cocofmt_add_atri.json',
            img_prefix=data_root + 'images/100k/train/',
            pipeline=train_pipeline
        )
    ],
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'labels/box_track_20/box_track_val_cocofmt.json',
        ann_file=data_root + 'labels/box_track_20/box_track_val_remap_cocofmt.json',
        img_prefix=data_root + 'images/track/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'labels/box_track_20/box_track_val_cocofmt.json',
        ann_file=data_root + 'labels/box_track_20/box_track_val_remap_cocofmt.json',
        img_prefix=data_root + 'images/track/val/',
        pipeline=test_pipeline)
)

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[8, 11])
# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'work_dirs/self_supervision_bdd100k_only_vehicle/latest.pth'
resume_from = None
workflow = [('train', 1)]
evaluation = dict(metric=['bbox', 'track'], interval=2)
find_unused_parameters = True