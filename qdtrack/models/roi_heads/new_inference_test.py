import torch
from mmdet.models import HEADS
from mmdet.core import bbox2roi
from .quasi_dense_roi_head import QuasiDenseRoIHead

@HEADS.register_module()
class NewInfRoIHead(QuasiDenseRoIHead):
    def __init__(self, track_roi_extractor=None, track_head=None, track_train_cfg=None, devide=(1,1) , *args, **kwargs):
        super().__init__(track_roi_extractor, track_head, track_train_cfg, *args, **kwargs)
        self.devide = devide

    def simple_test(self, x, img_metas, proposal_list, rescale):
        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        # TODO: support batch inference
        det_bboxes = det_bboxes[0]
        det_labels = det_labels[0]

        if det_bboxes.size(0) == 0:
            return det_bboxes, det_labels, None

        track_bboxes = det_bboxes[:, :-1] * torch.tensor(
            img_metas[0]['scale_factor']).to(det_bboxes.device)
        track_feats = self._track_forward(x, [track_bboxes])
        bs = det_bboxes.size()[0]
        track_feats = self._concat_feats(track_feats, bs)

        return det_bboxes, det_labels, track_feats
    
    def _concat_feats(self, feats, bs):
        ''''
        concat the seperate part emb into a single feat  
        (bs*part, emb)--->(bs, part*emb)
        '''
        _feats = feats.view(bs, -1)

        return _feats


    def _track_forward(self, x, bboxes):
        """Track head forward function used in both training and testing."""
        rois = bbox2roi(bboxes)
        part_rois = self._devide_rois(rois, m=self.devide[0], n=self.devide[1])
        track_feats = self.track_roi_extractor(
            x[:self.track_roi_extractor.num_inputs], part_rois)
        track_feats = self.track_head(track_feats)
        return track_feats

    def _devide_rois(self, rois, m, n):
        '''
        Devide each object into (m,n) parts
        
        Args:
            rois(torch.Tensor): a Tensor shaped (bs,5), which is [batch_inds, x1, y1, x2, y2]

        Returns:
            part_rois(torch.Tensor): shape(bs*(m*n+1), 5), which is also like [batch_inds, x1, y1, x2, y2]
                                    [holistic_roi, part_roi_1, part_roi_2, ...]
        '''
        part_rois_list = []
        for roi in rois:
            _ind = roi[0]
            x1, y1, x2, y2 = roi[1], roi[2], roi[3], roi[4]
            w, h = x2-x1, y2-y1
            part_rois_list.append(torch.unsqueeze(roi, 0))
            for i in range(m):
                for j in range(n):
                    _x1, _y1, _x2, _y2 = x1+j*(w/n), y1+i*(h/m), x1+(j+1)*(w/n), y1+(i+1)*(h/m)
                    part_roi = torch.Tensor([_ind, _x1, _y1, _x2, _y2]).to(roi.device)
                    part_roi = torch.unsqueeze(part_roi, 0)
                    # print('part_roi:',part_roi.size())
                    part_rois_list.append(part_roi)
        _part_rois = torch.cat(part_rois_list)
        # print('part_rois :',_part_rois.size())
        return _part_rois