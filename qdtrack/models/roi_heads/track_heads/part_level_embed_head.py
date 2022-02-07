import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from mmcv.cnn import ConvModule
from mmdet.models import HEADS, build_loss

from qdtrack.core import cal_similarity
from qdtrack.models.roi_heads.track_heads.quasi_dense_embed_head import QuasiDenseEmbedHead


@HEADS.register_module()
class PartLevelEmbedHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 num_fcs = 1,
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
                     hard_mining=True),
                 part=6):
        super(PartLevelEmbedHead, self).__init__()
        
        self.softmax_temp = softmax_temp
        self.loss_track = build_loss(loss_track)
        if loss_track_aux is not None:
            self.loss_track_aux = build_loss(loss_track_aux)
        else:
            self.loss_track_aux = None
        # different from PCB, nn...((self.part, 1)), 
        # because the vehicle is not a tall but a long obj
        self.part = part
        self.avgpool = nn.AdaptiveAvgPool2d((1, self.part))

        # TODO add dropout and compare result
        self.dropout = nn.Dropout(p=0.5)

        # before avgpooling conv
        self.conv_out_channels = conv_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.convs_before_avgpool, before_layer_dim  = self._add_conv_branch(
            self.num_convs, self.in_channels
        )
        
        # after avgpooling conv+fc
        self.layers_after_avgpool = nn.ModuleList()
        for i in range(self.part):
            self.layers_after_avgpool.append(ClassBlock(
                input_dim=before_layer_dim, num_bottleneck=embed_channels
            ))

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


    def forward(self, x):
        if self.num_convs > 0:
            for i, conv in enumerate(self.convs_before_avgpool):
                x = conv(x)
        # print('x size = {}'.format(x.size()))
        x = self.avgpool(x)
        # print('after pool size = {}'.format(x.size()))
        x = self.dropout(x)
        
        part = {}
        predict = {}

        for i in range(self.part):
            part[i] = x[:, :, :, i]
            part[i] = torch.unsqueeze(part[i], 3)
            predict[i] = self.layers_after_avgpool[i](part[i])
            if len(predict[i].size()) == 1:
                predict[i].unsqueeze_(0)

        y = []
        for i in range(self.part):
            y.append(predict[i])
        
        # x = torch.cat(y,1)
        # print('y size = ',len(y))
        # print('predict size =',predict[0].size())
        # print('x size = ',x.size())
        # emb = x.view(x.size(0), x.size(1), x.size(2))
        # print('emb size',emb.size())
        # return 
        
        emb = torch.cat(y,1)
        return emb

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


    def init_weights(self):
        return


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.constant(m.weight.data, 1)
        init.constant(m.bias.data, 0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim,  relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []

        add_block += [nn.Conv2d(input_dim, num_bottleneck, kernel_size=1, bias=False)]
        add_block += [nn.BatchNorm2d(num_bottleneck)]
        if relu:
            #add_block += [nn.LeakyReLU(0.1)]
            add_block += [nn.ReLU(inplace=True)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        self.add_block = add_block
        
    def forward(self, x):
        x = self.add_block(x)
        x = torch.squeeze(x)

        return x

    