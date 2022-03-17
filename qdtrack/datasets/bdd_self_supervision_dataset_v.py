from mmdet.datasets import DATASETS
import numpy as np


from .coco_video_dataset import CocoVideoDataset


@DATASETS.register_module()
class BDDSelfSupervisionDatasetV(CocoVideoDataset):

    CLASSES = ('car', 'bus', 'truck')

    # CLASSES = ('pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 
    #            'motorcycle', 'bicycle')
    def __init__(self, devide_w=4, devide_h=1 ,only_holistic=False, *args, **kwargs):
        self.devide_w = devide_w
        self.devide_h = devide_h
        self.only_holistic = only_holistic
        super().__init__(*args, **kwargs)


    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_instance_ids = []
        # gt_location_maps = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            if self.only_holistic and (ann['truncated'] or ann['occluded']):
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            # if self.only_holistic:
            #     location_map = np.zeros((h, w), dtype=np.int64)
            #     location_label = 0
            #     meta_h = h // self.devide_h
            #     meta_w = w // self.devide_w
            #     for i in range(self.devide_w):
            #         for j in range(self.devide_h):
            #             location_label += 1
            #             if i == self.devide_w -1 and img_info['width'] % self.devide_w != 0:
            #                 location_map[j*meta_h:(j+1)*meta_h, i*meta_w:img_info['width']] = location_label
            #                 continue
            #             if j == self.devide_h -1 and img_info['height'] % self.devide_h != 0:
            #                 location_map[j*meta_h:img_info['height'], i*meta_w:(i+1)*meta_w] = location_label
            #                 continue
            #             location_map[j*meta_h:(j+1)*meta_h, i*meta_w:(i+1)*meta_w] = location_label
            #     pass
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                if ann.get('segmentation', False):
                    gt_masks_ann.append(ann['segmentation'])
                instance_id = ann.get('instance_id', None)
                if instance_id is not None:
                    gt_instance_ids.append(ann['instance_id'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        # if self.only_holistic:
        #     location_map = np.zeros((img_info['height'], img_info['width']), dtype=np.int64)
        #     location_label = 0
        #     meta_h = img_info['height'] // self.devide_h
        #     meta_w = img_info['width'] // self.devide_w
        #     for i in range(self.devide_w):
        #         for j in range(self.devide_h):
        #             location_label += 1
        #             if i == self.devide_w -1 and img_info['width'] % self.devide_w != 0:
        #                 location_map[j*meta_h:(j+1)*meta_h, i*meta_w:img_info['width']] = location_label
        #                 continue
        #             if j == self.devide_h -1 and img_info['height'] % self.devide_h != 0:
        #                 location_map[j*meta_h:img_info['height'], i*meta_w:(i+1)*meta_w] = location_label
        #                 continue
        #             location_map[j*meta_h:(j+1)*meta_h, i*meta_w:(i+1)*meta_w] = location_label

        seg_map = img_info['filename'].replace('jpg', 'png')
        
        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        if self.load_as_video:
            ann['instance_ids'] = np.array(gt_instance_ids).astype(np.int)
        else:
            ann['instance_ids'] = np.arange(len(gt_labels))

        return ann