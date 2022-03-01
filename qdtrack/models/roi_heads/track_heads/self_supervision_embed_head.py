import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models import HEADS, build_loss

from qdtrack.core import cal_similarity


@HEADS.register_module()
class SelfSupervisionEmbedHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 num_fcs=0,
                 num_regions=4,
                 roi_feat_size=7,
                 in_channels=256,
                 conv_out_channels=256,
                 embed_channels=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 softmax_temp=-1,
                 loss_track=dict(
                     type='MultiPosCrossEntropyLoss', loss_weight=0.25),
                 loss_track_aux=dict(
                     type='L2Loss',
                     sample_ratio=3,
                     margin=0.3,
                     loss_weight=1.0,
                     hard_mining=True)):
        super(SelfSupervisionEmbedHead, self).__init__()
        self.num_convs = num_convs
        self.num_regions = num_regions
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.embed_channels = embed_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.relu = nn.ReLU(inplace=True)
        self.convs, last_layer_dim = self._add_conv_branch(
            self.num_convs, self.in_channels)
        self.classifer, last_layer_dim = self._add_classifier_branch(
            last_layer_dim, self.num_regions
        )
        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_embed = nn.Linear(last_layer_dim, embed_channels)
        
        self.softmax_temp = softmax_temp
        self.loss_track = build_loss(loss_track)
        if loss_track_aux is not None:
            self.loss_track_aux = build_loss(loss_track_aux)
        else:
            self.loss_track_aux = None

    def _add_conv_branch(self, num_convs, in_channels):
        last_layer_dim = in_channels
        # add branch specific conv layers
        convs = nn.ModuleList()
        if num_convs > 0:
            for i in range(num_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels

        return convs, last_layer_dim
        
        
    def _add_classifier_branch(self, in_channels, out_channels):

        classifier = nn.ModuleList()
        classifier.append(
            ConvModule(
                in_channels,
                out_channels,
                1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=None,
                act_cfg=None))
        last_layer_dim = out_channels
        return classifier, last_layer_dim

        

    def init_weights(self):
        nn.init.normal_(self.fc_embed.weight, 0, 0.01)
        nn.init.constant_(self.fc_embed.bias, 0)

    def forward(self, x):

        # 3*3 conv layers
        if self.num_convs > 0:
            for i, conv in enumerate(self.convs):
                x = conv(x)

        # 1*1 conv layer + softmax
        location_maps = self.classifer[0](x)
        location_maps = self.softmax(location_maps)

        # get embedding and visible-scores
        embeds = []
        scores = []
        for i in range(self.num_regions):
            location_map = location_maps[:,i,:,:].unsqueeze(1)
            emb = torch.mul(location_map, x)
            emb = self.avgpool(emb)
            embeds.append(emb.squeeze().unsqueeze(1))
            
            score = torch.sum(location_map, (2,3))
            scores.append(score.unsqueeze(1))

        embeds = torch.cat(embeds, dim=1)
        scores = torch.cat(scores, dim=1)

        # print("location_maps:{}".format(location_maps.size()))
        # print("embeds:{}".format(embeds.size()))
        # print('scores:{}'.format(scores.size()))

        return location_maps, embeds, scores

    def get_track_targets(self, gt_match_indices, key_sampling_results,
                          ref_sampling_results):
        track_targets = []
        track_weights = []
        for _gt_match_indices, key_res, ref_res in zip(gt_match_indices,
                                                       key_sampling_results,
                                                       ref_sampling_results):
            targets = _gt_match_indices.new_zeros(
                (key_res.pos_bboxes.size(0), ref_res.bboxes.size(0)),
                dtype=torch.int)
            _match_indices = _gt_match_indices[key_res.pos_assigned_gt_inds]
            pos2pos = (_match_indices.view(
                -1, 1) == ref_res.pos_assigned_gt_inds.view(1, -1)).int()
            targets[:, :pos2pos.size(1)] = pos2pos
            weights = (targets.sum(dim=1) > 0).float()
            track_targets.append(targets)
            track_weights.append(weights)
        return track_targets, track_weights

    def match(self, key_embeds, ref_embeds, key_sampling_results,
              ref_sampling_results):
        num_key_rois = [res.pos_bboxes.size(0) for res in key_sampling_results]
        key_embeds = torch.split(key_embeds, num_key_rois)
        num_ref_rois = [res.bboxes.size(0) for res in ref_sampling_results]
        ref_embeds = torch.split(ref_embeds, num_ref_rois)

        dists, cos_dists = [], []
        for key_embed, ref_embed in zip(key_embeds, ref_embeds):
            dist = cal_similarity(
                key_embed,
                ref_embed,
                method='dot_product',
                temperature=self.softmax_temp)
            dists.append(dist)
            if self.loss_track_aux is not None:
                cos_dist = cal_similarity(
                    key_embed, ref_embed, method='cosine')
                cos_dists.append(cos_dist)
            else:
                cos_dists.append(None)
        return dists, cos_dists

    def loss(self, dists, cos_dists, targets, weights):
        losses = dict()

        loss_track = 0.
        loss_track_aux = 0.
        for _dists, _cos_dists, _targets, _weights in zip(
                dists, cos_dists, targets, weights):
            loss_track += self.loss_track(
                _dists, _targets, _weights, avg_factor=_weights.sum())
            if self.loss_track_aux is not None:
                loss_track_aux += self.loss_track_aux(_cos_dists, _targets)
        losses['loss_track'] = loss_track / len(dists)

        if self.loss_track_aux is not None:
            losses['loss_track_aux'] = loss_track_aux / len(dists)

        return losses

    @staticmethod
    def random_choice(gallery, num):
        """Random select some elements from the gallery.

        It seems that Pytorch's implementation is slower than numpy so we use
        numpy to randperm the indices.
        """
        assert len(gallery) >= num
        if isinstance(gallery, list):
            gallery = np.array(gallery)
        cands = np.arange(len(gallery))
        np.random.shuffle(cands)
        rand_inds = cands[:num]
        if not isinstance(gallery, np.ndarray):
            rand_inds = torch.from_numpy(rand_inds).long().to(gallery.device)
        return gallery[rand_inds]
