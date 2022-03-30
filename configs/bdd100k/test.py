model = dict(
    type='VPMTrack',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', loss_weight=1.0, beta=0.1111111111111111)),
    roi_head=dict(
        type='SelfSupervisionRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=3,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        track_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=27, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        track_head=dict(
            type='SelfSupervisionEmbedHeadV2',
            num_convs=4,
            num_fcs=1,
            embed_channels=256,
            norm_cfg=dict(type='GN', num_groups=32),
            loss_track=dict(type='MultiPosCrossEntropyLoss', loss_weight=0.25),
            loss_track_aux=dict(
                type='L2Loss',
                neg_pos_ub=3,
                pos_margin=0,
                neg_margin=0.1,
                hard_mining=True,
                loss_weight=1.0),
            roi_feat_size=27,
            num_regions=3,
            loss_loc=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1),
            loss_loc_ref=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=0.001))),
            
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        embed=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='CombinedSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=3,
                add_gt_as_proposals=True,
                pos_sampler=dict(type='InstanceBalancedPosSampler'),
                neg_sampler=dict(
                    type='IoUBalancedNegSampler',
                    floor_thr=-1,
                    floor_fraction=0,
                    num_bins=3)))),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)),
    tracker=dict(
        type='QuasiDenseEmbedTracker',
        init_score_thr=0.7,
        obj_score_thr=0.3,
        match_score_thr=0.5,
        memo_tracklet_frames=10,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        with_cats=True,
        match_metric='bisoftmax'))
dataset_type = 'BDDSelfSupervisionDatasetV'
data_root = 'data/bdd/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(
        type='SeqLoadAnnotations',
        with_bbox=True,
        with_ins_id=True,
        with_loc_map=False),
    dict(type='SeqResize', img_scale=(1296, 720), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(
        type='SeqNormalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='SeqPad', size_divisor=32),
    dict(type='SeqDefaultFormatBundle'),
    dict(
        type='SeqCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_match_indices'],
        ref_prefix='ref')
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
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=[
        dict(
            type='BDDSelfSupervisionDatasetV',
            only_holistic=True,
            ann_file=
            'data/bdd/labels/box_track_20/box_track_train_remap_cocofmt_add_atri.json',
            img_prefix='data/bdd/images/track/train/',
            key_img_sampler=dict(interval=1),
            ref_img_sampler=dict(num_ref_imgs=1, scope=3, method='uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations',
                    with_bbox=True,
                    with_ins_id=True,
                    with_loc_map=False),
                dict(type='SeqResize', img_scale=(1296, 720), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=32),
                dict(type='SeqDefaultFormatBundle'),
                dict(
                    type='SeqCollect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_match_indices'],
                    ref_prefix='ref')
            ]),
        dict(
            type='BDDSelfSupervisionDatasetV',
            only_holistic=True,
            load_as_video=False,
            ann_file='data/bdd/labels/det_20/det_train_cocofmt_add_atri.json',
            img_prefix='data/bdd/images/100k/train/',
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations',
                    with_bbox=True,
                    with_ins_id=True,
                    with_loc_map=False),
                dict(type='SeqResize', img_scale=(1296, 720), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=32),
                dict(type='SeqDefaultFormatBundle'),
                dict(
                    type='SeqCollect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_match_indices'],
                    ref_prefix='ref')
            ])
    ],
    val=dict(
        type='BDDSelfSupervisionDatasetV',
        ann_file=
        'data/bdd/labels/box_track_20/box_track_val_remap_cocofmt.json',
        img_prefix='data/bdd/images/track/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1296, 720),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='VideoCollect', keys=['img'])
                ])
        ]),
    test=dict(
        type='BDDSelfSupervisionDatasetV',
        ann_file=
        'data/bdd/labels/box_track_20/box_track_val_remap_cocofmt.json',
        img_prefix='data/bdd/images/track/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1296, 720),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='VideoCollect', keys=['img'])
                ])
        ]))
optimizer = dict(type='SGD', lr=0.02/8, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[8, 11])
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
evaluation = dict(metric=['bbox', 'track'], interval=12)
find_unused_parameters = False
work_dir = './work_dirs/test'
gpu_ids = range(0, 1)
