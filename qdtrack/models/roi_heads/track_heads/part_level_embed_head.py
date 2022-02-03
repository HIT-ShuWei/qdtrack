import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from mmcv.cnn import ConvModule
from mmdet.models import HEADS, build_loss

from qdtrack.core import cal_similarity
from qdtrack.models.roi_heads.track_heads.quasi_dense_embed_head import QuasiDenseEmbedHead


@HEADS.register_module()
class PartLevelEmbedHead(QuasiDenseEmbedHead):

    def __init__(self,
                 num_convs=4,
                 num_fcs=1,
                 roi_feat_size=7,
                 in_channels=256,
                 conv_out_channels=256,
                 fc_out_channels=1024,
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
        super(PartLevelEmbedHead, self).__init__(num_convs, num_fcs, roi_feat_size, in_channels,
                                                 conv_out_channels, fc_out_channels, embed_channels,
                                                 conv_cfg, norm_cfg, softmax_temp, loss_track,
                                                 loss_track_aux)
        
        
        
        
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
        self.convs_before_avgpool, _, before_layer_dim  = self._add_conv_fc_branch(
            self.num_convs, self.num_fcs, self.in_channels
        )
        
        # after avgpooling conv+fc
        self.layers_after_avgpool = nn.ModuleList()
        for i in range(self.parts):
            self.layers_after_avgpool.append(ClassBlock(
                input_dim=before_layer_dim, num_bottleneck=embed_channels
            ))




    def forward(self, x):
        x = self.convs_before_avgpool(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        
        part = {}
        predict = {}

        for i in range(self.part):
            part[i] = x[:, :, i, :]
            part[i] = torch.unsqueeze(part[i], 3)
            predict[i] = self.layers_after_avgpool[i](part[i])

        y = []
        for i in range(self.part):
            y.append(predict[i])
        
        x = torch.cat(y,2)
        print('x size = ',x.size)
        emb = x.view(x.size(0), x.size(1), x.size(2))
        print('emb size',emb.size)
        return emb


    


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