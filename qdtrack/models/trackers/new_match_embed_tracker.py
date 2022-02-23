import torch
import torch.nn.functional as F
import torch.nn as nn
from mmdet.core import bbox_overlaps


from ..builder import TRACKERS
from .quasi_dense_embed_tracker import QuasiDenseEmbedTracker

@TRACKERS.register_module()
class NewMatchEmbedTracker(QuasiDenseEmbedTracker):

    def __init__(self, init_score_thr=0.8, obj_score_thr=0.5, match_score_thr=0.5, memo_tracklet_frames=10, memo_backdrop_frames=1, memo_momentum=0.8, nms_conf_thr=0.5, nms_backdrop_iou_thr=0.3, nms_class_iou_thr=0.7, with_cats=True, match_metric='bisoftmax', boundary_sift=True, memo_part_momentum=0.2):
        super().__init__(init_score_thr, obj_score_thr, match_score_thr, memo_tracklet_frames, memo_backdrop_frames, memo_momentum, nms_conf_thr, nms_backdrop_iou_thr, nms_class_iou_thr, with_cats, match_metric)
        self.boundary_sift = boundary_sift
        self.memo_part_momentum = memo_part_momentum

    def match(self, bboxes, labels, track_feats, frame_id, asso_tau=-1):
        
        # sort according to the score
        _, inds = bboxes[:, -1].sort(descending=True)
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        embeds = track_feats[inds, :]

        # duplicate removal for potential backdrops and cross classes
        valids = bboxes.new_ones((bboxes.size(0)))
        ious = bbox_overlaps(bboxes[:, :-1], bboxes[:, :-1])
        for i in range(1, bboxes.size(0)):
            thr = self.nms_backdrop_iou_thr if bboxes[
                i, -1] < self.obj_score_thr else self.nms_class_iou_thr
            if (ious[i, :i] > thr).any():
                valids[i] = 0
        valids = valids == 1
        bboxes = bboxes[valids, :]
        labels = labels[valids]
        embeds = embeds[valids, :]
        
        # init ids container
        ids = torch.full((bboxes.size(0), ), -1, dtype=torch.long)
        valid_parts = torch.full((bboxes.size(0), ), 0, dtype=torch.long)
        # match if buffer is not empty
        if bboxes.size(0) > 0 and not self.empty:
            (memo_bboxes, memo_labels, memo_embeds, memo_ids,
             memo_vs, memo_frames) = self.memo

            # print('memo_bbox_before:{}'.format(memo_bboxes.size()))
            if self.boundary_sift:
                # if the tracklet vannished in the boundary,
                # don't match such tracklet with the key obj not in boundary
                valids_boundary = memo_bboxes.new_ones((memo_bboxes.size(0)))
                left_thr = 20
                right_thr = 1280 - 20
                for i in range(1, memo_bboxes.size(0)):
                    if (memo_bboxes[i,0] < left_thr or memo_bboxes[i,2] > right_thr) \
                    and (frame_id - memo_frames[i]> 5):
                        valids_boundary[i] = 0
                valids_boundary = valids_boundary == 1
                memo_bboxes = memo_bboxes[valids_boundary, :]
                memo_labels = memo_labels[valids_boundary]
                memo_embeds = memo_embeds[valids_boundary, :]
            # print('memo_bbox_after:{}'.format(memo_bboxes.size()))

            scores, valid_part = self._get_dist(embeds, memo_embeds, metric=self.match_metric)

            # if self.match_metric == 'bisoftmax':
            #     feats = torch.mm(embeds, memo_embeds.t())
            #     d2t_scores = feats.softmax(dim=1)
            #     t2d_scores = feats.softmax(dim=0)
            #     scores = (d2t_scores + t2d_scores) / 2
            # elif self.match_metric == 'softmax':
            #     feats = torch.mm(embeds, memo_embeds.t())
            #     scores = feats.softmax(dim=1)
            # elif self.match_metric == 'cosine':
            #     scores = torch.mm(
            #         F.normalize(embeds, p=2, dim=1),
            #         F.normalize(memo_embeds, p=2, dim=1).t())
            # else:
            #     raise NotImplementedError
            
            if self.with_cats:
                cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
                scores *= cat_same.float().to(scores.device)
            

            for i in range(bboxes.size(0)):
                conf, memo_ind = torch.max(scores[i, :], dim=0)
                id = memo_ids[memo_ind]
                if conf > self.match_score_thr:
                    if id > -1:
                        valid_parts[i] = valid_part[i, memo_ind]
                        if bboxes[i, -1] > self.obj_score_thr:
                            ids[i] = id
                            scores[:i, memo_ind] = 0
                            scores[i + 1:, memo_ind] = 0
                        else:
                            if conf > self.nms_conf_thr:
                                ids[i] = -2
                        
        new_inds = (ids == -1) & (bboxes[:, 4] > self.init_score_thr).cpu()
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(
            self.num_tracklets,
            self.num_tracklets + num_news,
            dtype=torch.long)
        self.num_tracklets += num_news

        self.update_memo(ids, bboxes, embeds, labels, frame_id, valid_parts)
        

        return bboxes, labels, ids

    def _get_dist(self, _embeds, _memo_embeds, metric='bisoftmax'):
        '''
        calculate the holistic emb from _embeds([0:256]) with every emb in _memo_embed
        get the best dist for each obj
        
        Args:
        _embeds(torch.Tensor): [batch_size, embedings(holistic, p1, p2, ...)]
        _memo_embeds(torch.Tensor): [tracklet_nums, embeddings(h, p1, p2, ...)]
        _metric(str): in ['bisoftmax', 'cosine', 'softmax']
        Outputs:
        _scores(torch.Tensor): [batch_size, tracklet_nums] 
                              the distances between holistic key obj and each part emb of tracklets obj
        '''
        bs, t_nums = _embeds.size()[0], _memo_embeds.size()[0]
        parts_per_obj = int(_embeds.size()[1] / 256)
        score_list = []
        
        # get dist between holistic key-obj and each part in tracklets
        for i in range(parts_per_obj):
            if self.match_metric == 'bisoftmax':
                feat = torch.mm(_embeds[:, 0:256], _memo_embeds[:, i*256:(i+1)*256].t())
                d2t_scores = feat.softmax(dim=1)
                t2d_scores = feat.softmax(dim=0)
                score = (d2t_scores + t2d_scores) / 2
            elif self.match_metric == 'softmax':
                feat = torch.mm(_embeds[:, 0:256], _memo_embeds[:, i*256:(i+1)*256].t())
                score = feat.softmax(dim=1)
            elif self.match_metric == 'cosine':
                score = torch.mm(
                    F.normalize(_embeds[:, 0:256], p=2, dim=1),
                    F.normalize(_memo_embeds[:, i*256:(i+1)*256], p=2, dim=1).t())
            else:
                raise NotImplementedError
            score_list.append(score)
        _scores = torch.cat(score_list, dim=1).view(bs, parts_per_obj, t_nums).transpose(1,2)

        (scores_best, best_inds) = torch.max(_scores, 2)
        # maxpool = nn.MaxPool1d(parts_per_obj)
        # scores_best = maxpool(_scores)
        # scores_best = torch.squeeze(scores_best, dim=2)
        return scores_best, best_inds

    @property
    def memo(self):
        memo_embeds = []
        memo_ids = []
        memo_bboxes = []
        memo_labels = []
        memo_vs = []
        memo_frames = []
        for k, v in self.tracklets.items():
            memo_bboxes.append(v['bbox'][None, :])
            memo_embeds.append(v['embed'][None, :])
            memo_ids.append(k)
            memo_labels.append(v['label'].view(1, 1))
            memo_vs.append(v['velocity'][None, :])
            memo_frames.append(torch.tensor(v['last_frame']).view(1, 1))
        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)

        for backdrop in self.backdrops:
            backdrop_ids = torch.full((1, backdrop['embeds'].size(0)),
                                      -1,
                                      dtype=torch.long)
            backdrop_vs = torch.zeros_like(backdrop['bboxes'])
            memo_bboxes.append(backdrop['bboxes'])
            memo_embeds.append(backdrop['embeds'])
            memo_ids = torch.cat([memo_ids, backdrop_ids], dim=1)
            memo_labels.append(backdrop['labels'][:, None])
            memo_vs.append(backdrop_vs)
            memo_frames.append(torch.tensor(backdrop['last_frame'][:, None]))


        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        memo_vs = torch.cat(memo_vs, dim=0)
        memo_frames = torch.cat(memo_frames, dim=0).squeeze(1)


        return memo_bboxes, memo_labels, memo_embeds, memo_ids.squeeze(
            0), memo_vs, memo_frames

    def update_memo(self, ids, bboxes, embeds, labels, frame_id, valid_parts=None):
        tracklet_inds = ids > -1

        # update memo
        for id, bbox, embed, label, valid_part in zip(ids[tracklet_inds],
                                          bboxes[tracklet_inds],
                                          embeds[tracklet_inds],
                                          labels[tracklet_inds],
                                          valid_parts[tracklet_inds]):
            id = int(id)
            valid_part = int(valid_part)
            if id in self.tracklets.keys():
                velocity = (bbox - self.tracklets[id]['bbox']) / (
                    frame_id - self.tracklets[id]['last_frame'])
                self.tracklets[id]['bbox'] = bbox
                if valid_part == 0:
                    self.tracklets[id]['embed'] = (
                    1 - self.memo_momentum
                ) * self.tracklets[id]['embed'] + self.memo_momentum * embed
                elif valid_part != 0:
                    self.tracklets[id]['embed'] = (
                        1 - self.memo_part_momentum
                    ) * self.tracklets[id]['embed'] + self.memo_part_momentum * embed
                self.tracklets[id]['last_frame'] = frame_id
                self.tracklets[id]['label'] = label
                self.tracklets[id]['velocity'] = (
                    self.tracklets[id]['velocity'] *
                    self.tracklets[id]['acc_frame'] + velocity) / (
                        self.tracklets[id]['acc_frame'] + 1)
                self.tracklets[id]['acc_frame'] += 1
            else:
                self.tracklets[id] = dict(
                    bbox=bbox,
                    embed=embed,
                    label=label,
                    last_frame=frame_id,
                    velocity=torch.zeros_like(bbox),
                    acc_frame=0)

        backdrop_inds = torch.nonzero(ids == -1, as_tuple=False).squeeze(1)
        ious = bbox_overlaps(bboxes[backdrop_inds, :-1], bboxes[:, :-1])
        for i, ind in enumerate(backdrop_inds):
            if (ious[i, :ind] > self.nms_backdrop_iou_thr).any():
                backdrop_inds[i] = -1
        backdrop_inds = backdrop_inds[backdrop_inds > -1]

        self.backdrops.insert(
            0,
            dict(
                bboxes=bboxes[backdrop_inds],
                embeds=embeds[backdrop_inds],
                labels=labels[backdrop_inds],
                last_frame=torch.tensor([frame_id for i in range(len(backdrop_inds))])))

        # pop memo
        invalid_ids = []
        for k, v in self.tracklets.items():
            if frame_id - v['last_frame'] >= self.memo_tracklet_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracklets.pop(invalid_id)

        if len(self.backdrops) > self.memo_backdrop_frames:
            self.backdrops.pop()