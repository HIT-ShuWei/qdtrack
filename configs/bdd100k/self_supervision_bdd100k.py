_base_ = './qdtrack-frcnn_r50_fpn_12e_bdd100k.py'

model = dict(
    type='VPMTrack',
    roi_head=dict(
        type="SelfSupervisionRoIHead",
        track_roi_extractor=dict(
            roi_layer=dict(type='RoIAlign', output_size=28, sampling_ratio=0),
        ),
        track_head=dict(
            type='SelfSupervisionEmbedHead',
            roi_feat_size=28,
        ),
    ),
)

dataset_type = 'BDDSelfSupervisionDataset'
data_root = 'data/bdd/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    # dict(type='SeqLoadAnnotations', with_bbox=True, with_ins_id=True, with_loc_map=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_ins_id=True, with_loc_map=True),
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
data = dict(
    train = [
        dict(
            type=dataset_type,
            devide_w = 2,
            devide_h = 2,
            only_holistic = True,
            load_as_video = False,
            ann_file=data_root + 'labels/det_20/det_train_cocofmt_add_atri.json',
            img_prefix=data_root + 'images/100k/train/',
            pipeline=train_pipeline
        )
    ]
)