import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models import HEADS, build_loss

from qdtrack.core import cal_similarity, cal_weighted_similarity

import math 


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
                 loss_loc=dict(type='CrossEntropyLoss', use_mask=True),
                 loss_loc_ref = dict(type='CrossEntropyLoss', use_mask=True,loss_weight=0.001),
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
        self.loss_loc = build_loss(loss_loc)
        self.loss_loc_ref = build_loss(loss_loc_ref)

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

        # embeds = embeds.view(embeds.size(0), -1)
        
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
              ref_sampling_results, key_scores, ref_scores):
        num_key_rois = [res.pos_bboxes.size(0) for res in key_sampling_results]
        key_embeds = torch.split(key_embeds, num_key_rois)
        key_scores = torch.split(key_scores, num_key_rois)
        num_ref_rois = [res.bboxes.size(0) for res in ref_sampling_results]
        ref_embeds = torch.split(ref_embeds, num_ref_rois)
        ref_scores = torch.split(ref_scores, num_ref_rois)

        dists, cos_dists = [], []
        for key_embed, ref_embed, key_score, ref_score in \
        zip(key_embeds, ref_embeds, key_scores, ref_scores):
            dist = cal_weighted_similarity(
                key_embed,
                ref_embed,
                key_score,
                ref_score,
                method='dot_product',
                temperature=self.softmax_temp)
            dists.append(dist)
            if self.loss_track_aux is not None:
                cos_dist = cal_weighted_similarity(
                    key_embed, ref_embed, key_score, ref_score, method='cosine')
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

    def loss_location(self, key_location_maps, gt_location_maps, key_sampling_results):
        """
        loss of location map----> CE Loss
        
        Args:
        key_location_maps (Tensor): [batch_inds, region_inds, h, w]
        gt_location_maps (Tensor): [batch_inds, region_inds, h, w]
        
        Return:

        loss: CE loss
        """
        losses = dict()

        loss_loc = 0.

        num_key_rois = [res.pos_bboxes.size(0) for res in key_sampling_results]
        key_location_maps = torch.split(key_location_maps, num_key_rois)
        gt_location_maps = torch.split(gt_location_maps, num_key_rois)

        for _key_loc, _gt_loc in zip(key_location_maps, gt_location_maps):
            loss_loc += self.loss_loc(_key_loc, _gt_loc)
        
        losses['loss_loc'] = loss_loc / len(key_location_maps)

        return losses

    def loss_location_ref(self, ref_location_maps, gt_location_maps, ref_sampling_results):
        """
        loss of location map----> CE Loss
        
        Args:
        key_location_maps (Tensor): [batch_inds, region_inds, h, w]
        gt_location_maps (Tensor): [batch_inds, region_inds, h, w]
        
        Return:

        loss: CE loss
        """
        losses = dict()

        loss_loc = 0.

        num_ref_rois = [res.bboxes.size(0) for res in ref_sampling_results]
        ref_location_maps = torch.split(ref_location_maps, num_ref_rois)
        gt_location_maps = torch.split(gt_location_maps, num_ref_rois)

        for _ref_loc, _gt_loc in zip(ref_location_maps, gt_location_maps):
            loss_loc += self.loss_loc_ref(_ref_loc, _gt_loc)
        
        losses['loss_loc_ref'] = loss_loc / len(ref_location_maps)

        return losses


    def get_loc_maps(self, gt_bboxes, key_sampling_results):
        """
        generate the self-supervision label for location layer
        Args:
            gt_bboxes(list(Tensor)) : ground_truth bounding box 
            key_sampling_results(list(SamplingResult)) : Sampling result after proposal and assign

        Returns:
            gt_location_maps(Tensor) : [batch_inds, num_regiouns, roi_feat_size, roi_feat_size] 
        """
        key_bboxes = [res.pos_bboxes for res in key_sampling_results]
        key_is_gts = [res.pos_is_gt for res in key_sampling_results]
        key_gt_inds = [res.pos_assigned_gt_inds for res in key_sampling_results]

        
        gt_location_maps = []

        for _gt_bbox, _key_bbox, _key_is_gt, _key_gt_ind in zip(
            gt_bboxes, key_bboxes, key_is_gts, key_gt_inds):
            
            loc_map = self.generate(_gt_bbox, _key_bbox, _key_is_gt, _key_gt_ind)
            
            gt_location_maps.append(loc_map)            

        gt_location_maps = torch.cat(gt_location_maps, dim=0)
        
        return gt_location_maps


    def generate(self, gt_bboxes, key_bboxes, key_is_gt, key_gt_ind):
        """
        Get the self-supervision label

        """
        bs = key_bboxes.size(0)
        loc_map = torch.zeros((bs, self.num_regions, self.roi_feat_size, self.roi_feat_size)).to(key_bboxes.device)

        for ind in range(key_bboxes.size(0)):
            if key_is_gt[ind]:
                # if the box is gt, the map label just devide it into num_regions 
                for ind_region in range(self.num_regions):
                    begin = ind_region * math.floor(self.roi_feat_size / self.num_regions)
                    end = (ind_region+1) * math.floor(self.roi_feat_size / self.num_regions)
                    loc_map[ind, ind_region, :, begin:end] = 1
            else:
                # the box is a proposal, cal the overlap on gt
                gt_ind = key_gt_ind[ind]
                gt_bbox = gt_bboxes[gt_ind].clone()
                key_bbox = key_bboxes[ind].clone()
                unit_w = (gt_bbox[2] - gt_bbox[0]) / self.num_regions
                x1, y1, x2, y2 = gt_bbox[:4].clone().detach()
                x1_k, y1_k, x2_k, y2_k = key_bbox[:4].clone().detach()
                w_k, h_k = x2_k-x1_k, y2_k-y1_k
                w_ratio, h_ratio = w_k / self.roi_feat_size, h_k / self.roi_feat_size

                for ind_region in range(self.num_regions):
                    if (x1_k < x1 + ind_region*unit_w and x2_k < x1 + ind_region*unit_w) or \
                    (x1_k > x1 + (ind_region+1)*unit_w and x2_k > x1 + (ind_region+1)*unit_w): 
                        # key_bbox doesn't has the ind_region
                        # loc_map[ind, ind_region, :, :] = 0
                        continue

                    else:
                        # key_bbox has this region overlap with gt_bbox
                        r_x1 = max(x1_k, x1 + ind_region*unit_w)
                        r_x2 = min(x2_k, x1 + (ind_region+1)*unit_w)
                        r_y1 = max(y1_k, y1)
                        r_y2 = min(y2_k, y2)
                        # down sampling
                        r_x1_ds = math.floor((r_x1-x1_k) / w_ratio)
                        r_x2_ds = math.floor((r_x2-x1_k) / w_ratio)
                        r_y1_ds = math.floor((r_y1-y1_k) / h_ratio)
                        r_y2_ds = math.floor((r_y2-y1_k) / h_ratio)
                        # self-supervision labeling
                        
                        loc_map[ind, ind_region, r_y1_ds:r_y2_ds, r_x1_ds:r_x2_ds] = 1
                        # torch.set_printoptions(profile = 'full')
                        # print(loc_map[ind, ind_region, :, :])

        return loc_map


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
