_base_ = './qdtrack-frcnn_r50_fpn_12e_bdd100k.py'

model = dict(
    roi_head=dict(
        type='NewInfRoIHead',
        devide=(1,3),
        track_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
    ),
    tracker=dict(
        type='NewMatchEmbedTracker',
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
        match_metric='cosine',
        boundary_sift=True,
        memo_part_momentum=0.4),
)