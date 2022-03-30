import numpy as np
import torch

def loc2result(loc_maps, labels, ids, num_classes):
    valid_inds = ids > -1
    labels = labels[valid_inds]
    ids = ids[valid_inds]
    loc_maps = loc_maps[valid_inds]

    if loc_maps.shape[0] == 0:
        return [np.zeros((0, 3, 27, 27), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(loc_maps, torch.Tensor):
            loc_maps = loc_maps.cpu().numpy()
            labels = labels.cpu().numpy()
            ids = ids.cpu().numpy()
        return [
            loc_maps[labels == i, :, :, :]  for i in range(num_classes)
        ]



def track2result(bboxes, labels, ids, num_classes):
    valid_inds = ids > -1
    bboxes = bboxes[valid_inds]
    labels = labels[valid_inds]
    ids = ids[valid_inds]

    if bboxes.shape[0] == 0:
        return [np.zeros((0, 6), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()
            labels = labels.cpu().numpy()
            ids = ids.cpu().numpy()
        return [
            np.concatenate((ids[labels == i, None], bboxes[labels == i, :]),
                           axis=1) for i in range(num_classes)
        ]


def restore_result(result, return_ids=False):
    labels = []
    for i, bbox in enumerate(result):
        labels.extend([i] * bbox.shape[0])
    bboxes = np.concatenate(result, axis=0).astype(np.float32)
    labels = np.array(labels, dtype=np.int64)
    if return_ids:
        ids = bboxes[:, 0].astype(np.int64)
        bboxes = bboxes[:, 1:]
        return bboxes, labels, ids
    else:
        return bboxes, labels

def restore_loc_result(result):
    loc_maps = np.concatenate(result, axis=0).astype(np.float32)
    return loc_maps
