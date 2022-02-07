_base_ = './qdtrack-frcnn_r50_fpn_12e_bdd100k.py'
model = dict(
    roi_head = dict(
        track_head=dict(
        type='PartLevelEmbedHead',
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
        part=6)),

    )

data = dict(
    samples_per_gpu = 2
)
optimizer = dict(type='SGD', lr=0.02/2, momentum=0.9, weight_decay=0.0001)

