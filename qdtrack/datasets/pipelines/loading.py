from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile


@PIPELINES.register_module()
class LoadMultiImagesFromFile(LoadImageFromFile):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            # print('img.shape={}'.format(_results['img'].shape))
            # print('img.type={}'.format(type(_results['img'])))

            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqLoadAnnotations(LoadAnnotations):

    def __init__(self, with_loc_map=False ,with_ins_id=False, *args, **kwargs):
        # TODO: name
        super().__init__(*args, **kwargs)
        self.with_ins_id = with_ins_id
        self.with_loc_map = with_loc_map

    def _load_ins_ids(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_match_indices'] = results['ann_info'][
            'match_indices'].copy()

        return results

    def _load_loc_map(self, results):

        results['location_maps'] = results['ann_info']['location_map'].copy()

        return results

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            if self.with_loc_map:
                _results = self._load_loc_map(_results)
            if self.with_ins_id:
                _results = self._load_ins_ids(_results)
            outs.append(_results)

        return outs
