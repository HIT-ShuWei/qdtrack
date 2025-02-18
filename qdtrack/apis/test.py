import os.path as osp
import shutil
import tempfile
import time
from collections import defaultdict

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

from mmcv.image import tensor2imgs
import os

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = defaultdict(list)
    dataset = data_loader.dataset
    prev_img_meta = None
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        for k, v in result.items():
            results[k].append(v)

        # if show or out_dir:
        #     pass  # TODO

        batch_size = data['img'][0].size(0)
        #-----------------------------------------------#
        # [Zhangsw]
        # Visulization like mmtracking[https://github.com/open-mmlab/mmtracking/blob/8016b903aceb21e3e6a97d72b6f17efef1c86562/mmtrack/apis/test.py#L51]
        #-----------------------------------------------#
        if show or out_dir:
            assert batch_size == 1, 'Only support batch_size=1 when testing.'
            img_tensor = data['img'][0]
            img_meta = data['img_metas'][0].data[0][0]
            img = tensor2imgs(img_tensor, **img_meta['img_norm_cfg'])[0]

            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            ori_h, ori_w = img_meta['ori_shape'][:-1]
            img_show = mmcv.imresize(img_show, (ori_w, ori_h))

            if out_dir:
                out_file = osp.join(out_dir, img_meta['ori_filename'])
            else:
                out_file = None

            model.module.show_result(
                img_show,
                result,
                show=show,
                out_file=out_file,
                score_thr=show_score_thr)

            # Whether need to generate a video from images.
            # The frame_id == 0 means the model starts processing
            # a new video, therefore we can write the previous video.
            # There are two corner cases.
            # Case 1: prev_img_meta == None means there is no previous video.
            # Case 2: i == len(dataset) means processing the last video
            need_write_video = (
                prev_img_meta is not None and img_meta['frame_id'] == 0
                or i == len(dataset))
            if out_dir and need_write_video:
                prev_img_prefix, prev_img_name = prev_img_meta[
                    'ori_filename'].rsplit('/', 1)
                # prev_img_prefix: 图片存储的文件夹
                # prev_img_name: 图片名
                prev_img_idx, prev_img_type = prev_img_name.split('.')
                # prev_img_idx: 图片名去掉后缀
                # prev_img_type: 图片名后缀
                prev_img_idx = prev_img_idx.split('-')[-1]
                prev_filename_tmpl = '{:0' + str(len(prev_img_idx)) + 'd}.' + prev_img_type      # {:025d}.jpg
                prev_img_dirs = f'{out_dir}/{prev_img_prefix}'      # 输出图片路径
                prev_img_names = sorted(os.listdir(prev_img_dirs))
                # prev_start_frame_id = int(prev_img_names[0].split('.')[0])
                # prev_end_frame_id = int(prev_img_names[-1].split('.')[0])
                prev_start_frame_id = int(prev_img_names[0].split('.')[0].split('-')[-1])
                prev_end_frame_id = int(prev_img_names[-1].split('.')[0].split('-')[-1])
                mmcv.frames2video(
                    prev_img_dirs,
                    f'{prev_img_dirs}/out_video.mp4',
                    fps=6,
                    fourcc='mp4v',
                    filename_tmpl=prev_filename_tmpl,
                    start=prev_start_frame_id,
                    end=prev_end_frame_id,
                    show_progress=False)

            prev_img_meta = img_meta
        #---------------------------------#
        #[Zhangsw] Code Modify End 
        #---------------------------------#

        
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = defaultdict(list)
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        for k, v in result.items():
            results[k].append(v)

        if rank == 0:
            batch_size = (
                len(data['img_meta']._data)
                if 'img_meta' in data else data['img'][0].size(0))
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        raise NotImplementedError
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = defaultdict(list)
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_file = mmcv.load(part_file)
            for k, v in part_file.items():
                part_list[k].extend(v)
        shutil.rmtree(tmpdir)
        return part_list
