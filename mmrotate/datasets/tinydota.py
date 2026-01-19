# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from terminaltables import AsciiTable
from mmcv.utils import print_log
import glob
import os
import os.path as osp
import re
import tempfile
import time
import zipfile
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
import torch.multiprocessing as mp
from tqdm import tqdm
from .eval.tinydota_eval import TinyDOTAeval
import mmcv
import numpy as np
import torch
from mmcv.ops import nms_rotated
from mmdet.datasets.custom import CustomDataset

from mmrotate.core import obb2poly_np, poly2obb_np
from .builder import ROTATED_DATASETS
import random
INF = 1e8

@ROTATED_DATASETS.register_module()
class TinyDOTADataset(CustomDataset):
    """DOTA dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        version (str, optional): Angle representations. Defaults to 'oc'.
        difficulty (bool, optional): The difficulty threshold of GT.
    """
    CLASSES = ('plane', 'bridge', 'small-vehicle', 'large-vehicle', 
               'ship', 'storage-tank', 'swimming-pool', 'helicopter')

    PALETTE = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 ori_ann_file,
                 version='le90',
                 difficulty=100,
                 filter_dense=INF,
                 **kwargs):
        self.ori_ann_file = ori_ann_file
        self.version = version
        self.difficulty = difficulty
        self.filter_dense = filter_dense
        self.ori_data_infos = self.load_ori_annotations(ori_ann_file)
        self.cat_ids = self._get_cat_ids()

        super(TinyDOTADataset, self).__init__(ann_file, pipeline, **kwargs)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)
    
    def _get_cat_ids(self):
        cat_ids = dict()
        for idx, cat in enumerate(self.CLASSES):
            cat_ids[idx] = cat
        return cat_ids

    def load_annotations(self, ann_folder):
        """
            Args:
                ann_folder: folder that contains DOTA v1 annotations txt files
        """
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        ann_files = glob.glob(ann_folder + '/*.txt')

        # if self.test_mode is False:
        #     random.seed(42)
        #     sample_size = int(len(ann_files) * 0.1) 
        #     ann_files = random.sample(ann_files, sample_size)


        data_infos = []
        if not ann_files:  # test phase
            ann_files = glob.glob(ann_folder + '/*.png')
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.png'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.png'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []

                if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    if len(s) > self.filter_dense:
                        continue
                    for si in s:
                        if si.split()[-2] in self.CLASSES:
                            bbox_info = si.split()
                            poly = np.array(bbox_info[:8], dtype=np.float32)
                            try:
                                x, y, w, h, a = poly2obb_np(poly, self.version)
                            except:  # noqa: E722
                                continue
                            cls_name = bbox_info[8]
                            difficulty = int(bbox_info[9])
                            label = cls_map[cls_name]
                            if difficulty > self.difficulty:
                                pass
                            else:
                                gt_bboxes.append([x, y, w, h, a])
                                gt_labels.append(label)
                                gt_polygons.append(poly)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(
                        gt_polygons, dtype=np.float32)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8),
                                                            dtype=np.float32)

                if gt_polygons_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.array(
                        gt_polygons_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.zeros(
                        (0, 8), dtype=np.float32)

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos

    def load_ori_annotations(self, ann_folder):
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        ann_files = glob.glob(ann_folder + '/*.txt')
        data_infos = []
        if not ann_files:  # test phase
            ann_files = glob.glob(ann_folder + '/*.png')
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.png'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.png'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []

                if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        elem = si.split()
                        if len(elem) < 2:
                            continue
                        if elem[-2] in self.CLASSES:
                            bbox_info = si.split()
                            poly = np.array(bbox_info[:8], dtype=np.float32)
                            try:
                                x, y, w, h, a = poly2obb_np(poly, self.version)
                            except:  # noqa: E722
                                continue
                            cls_name = bbox_info[8]
                            difficulty = int(bbox_info[9])
                            label = cls_map[cls_name]
                            if difficulty > self.difficulty:
                                pass
                            else:
                                gt_bboxes.append([x, y, w, h, a])
                                gt_labels.append(label)
                                gt_polygons.append(poly)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(
                        gt_polygons, dtype=np.float32)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8),
                                                            dtype=np.float32)

                if gt_polygons_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.array(
                        gt_polygons_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.zeros(
                        (0, 8), dtype=np.float32)

                data_infos.append(data_info)

        self.ori_img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos

    def get_ori_ann_info(self, idx):
        return self.ori_data_infos[idx]['ann']

    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if (not self.filter_empty_gt
                    or data_info['ann']['labels'].size > 0):
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def translate(self, bboxes, x, y):
        translated = bboxes.copy()
        translated[..., :2] = translated[..., :2] + \
                              np.array([x, y], dtype=np.float32)
        return translated

    def merge_det(self,
                  results,
                  with_merge=True,
                  nms_iou_thr=0.5,
                  nproc=10,
                  save_dir=None,
                  **kwargs):
        if mmcv.is_list_of(results, tuple):
            dets, segms = results
        else:
            dets = results

        if not with_merge:
            results = [(data_info['id'], result)
                       for data_info, result in zip(self.data_infos, results)]
            if save_dir is not None:
                pass  # TODO:
            return results

        if mp.get_start_method(allow_none=True) != 'spawn':
            print("INFO: Multiprocessing start method is not 'spawn'. Attempting to set it...")
            try:
                mp.set_start_method('spawn', force=True)
                print("SUCCESS: Multiprocessing start method set to 'spawn'.")
            except RuntimeError as e:
                print(f"WARNING: Could not set start method to 'spawn': {e}")
                print(
                    "WARNING: This may lead to CUDA errors. Please consider setting it at the start of your main script.")

        print('\n>>> Merge detected results of patch for whole image evaluating...')
        start_time = time.time()

        # Use a list to store intermediate results, then concatenate
        collector = defaultdict(list)

        # Pre-process detections for each patch
        for data_info, result in tqdm(zip(self.data_infos, dets), total=len(self.data_infos),
                                      desc="Collecting patch results"):
            filename = data_info['filename']
            # Improved file name parsing: Use a more robust split, or even better,
            # store coordinates in `data_info` during data loading.
            try:
                parts = filename.split('___')
                x_start, y_start = \
                    int(parts[0].split('__')[-1]), \
                    int(parts[-1].split('.')[0])
                ori_name = filename.split('__')[0]
            except IndexError:
                print(f"Warning: Could not parse coordinates from filename: {filename}. Skipping.")
                continue

            # This part is optimized for better array concatenation
            all_bboxes_for_patch = []
            for i, res in enumerate(result):
                if res.size > 0:
                    bboxes, scores = res[:, :-1], res[:, [-1]]
                    bboxes = self.translate(bboxes, x_start, y_start)
                    labels = np.full((bboxes.shape[0], 1), i, dtype=np.float32)
                    all_bboxes_for_patch.append(np.concatenate(
                        [labels, bboxes, scores], axis=1
                    ))

            if all_bboxes_for_patch:
                new_result = np.concatenate(all_bboxes_for_patch, axis=0)
                collector[ori_name].append(new_result)

        # NMS device is default to 'cuda:0' otherwise 'cpu'
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Using device for NMS: {device}")

        # Use functools.partial for better function wrapping
        merge_func = partial(merge_function, CLASSES=self.CLASSES, iou_thr=nms_iou_thr, device=device)

        # Process results in parallel or sequentially
        if nproc > 1:
            try:
                with Pool(nproc) as pool:
                    merged_results = pool.map(merge_func, list(collector.items()))
            except Exception as e:
                print(f"Error during multiprocessing: {e}. Falling back to sequential processing.")
                merged_results = list(map(merge_func, list(collector.items())))
        else:
            merged_results = list(map(merge_func, list(collector.items())))

        if save_dir is not None:
            pass

        stop_time = time.time()
        print('Merge results completed, it costs %.1f seconds.' % (stop_time - start_time))
        return merged_results



    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 metric_items=None,
                 scale_ranges=None,
                 nproc=8):
        txtPth = "./work_dirs/evalRes.txt"
        txtFile = open(txtPth, 'a+')
        txtFile.writelines(f"{'-' * 30}Overall Evaluation{'-' * 30}")
        txtFile.writelines('\n')
        txtFile.close()
        time.sleep(2)

        nproc = 4
        merged_results = self.merge_det(results, nproc=nproc)
        merge_idx = [self.ori_img_ids.index(res[0]) for res in merged_results]
        results = [res[1] for res in merged_results]  # exclude `id` for evaluation

        iou_thr = np.linspace(.5, 0.75, int(np.round((0.75 - .5) / .25)) + 1, endpoint=True)
        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ori_ann_info(i) for i in merge_idx]

        eval_results = {}
        TinyDOTAEval = TinyDOTAeval(annotations, results, numCats=len(self.CLASSES), nproc=nproc)
        TinyDOTAEval.params.iouThrs = iou_thr

        # mapping of cocoEval.stats
        Tinydota_metric_names = {
            'AP': 0,
            'AP_75': 1,
            'AP_eS': 2,
            'AP_rS': 3,
            'AP_gS': 4,
            'AP_Normal': 5,
            'AR': 6,
            'AR_eS': 7,
            'AR_rS': 8,
            'AR_gS': 9,
            'AR_Normal': 10
        }
        TinyDOTAEval.evaluate()
        TinyDOTAEval.accumulate()
        TinyDOTAEval.summarize()

        classwise = True
        if classwise:  # Compute per-category AP
            # Compute per-category AP
            # from https://github.com/facebookresearch/detectron2/
            precisions = TinyDOTAEval.eval['precision']
            # precision: (iou, recall, cls, area range, max dets)
            assert len(self.cat_ids) == precisions.shape[2]

            results_per_category = []
            for catId, catName in self.cat_ids.items():
                # area range index 0: all area ranges
                # max dets index -1: typically 20000 per image
                precision = precisions[:, :, catId, 0, -1]
                precision = precision[precision > -1]
                if precision.size:
                    ap = np.mean(precision)
                else:
                    ap = float('nan')
                results_per_category.append(
                    (f'{catName}', f'{float(ap):0.3f}'))

            num_columns = min(6, len(results_per_category) * 2)
            results_flatten = list(
                itertools.chain(*results_per_category))
            headers = ['category', 'AP'] * (num_columns // 2)
            results_2d = itertools.zip_longest(*[
                results_flatten[i::num_columns]
                for i in range(num_columns)
            ])
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            print_log('\n' + table.table, logger=logger)

        # TODO: proposal evaluation
        if metric_items is None:
            metric_items = [
                'AP', 'AP_75', 'AP_eS',
                'AP_rS', 'AP_gS', 'AP_Normal'
            ]

        for metric_item in metric_items:
            key = f'{metric}_{metric_item}'
            val = float(
                f'{TinyDOTAEval.stats[Tinydota_metric_names[metric_item]]:.3f}'
            )
            eval_results[key] = val
        ap = TinyDOTAEval.stats[:6]
        eval_results[f'{metric}_mAP_copypaste'] = (
            f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} '
            f'{ap[3]:.3f} {ap[4]:.3f} {ap[5]:.3f}'
        )

        return eval_results



    def _results2submission(self, id_list, dets_list, out_folder=None):
        """Generate the submission of full images.

        Args:
            id_list (list): Id of images.
            dets_list (list): Detection results of per class.
            out_folder (str, optional): Folder of submission.
        """
        if osp.exists(out_folder):
            raise ValueError(f'The out_folder should be a non-exist path, '
                             f'but {out_folder} is existing')
        os.makedirs(out_folder)

        files = [
            osp.join(out_folder, 'Task1_' + cls + '.txt')
            for cls in self.CLASSES
        ]
        file_objs = [open(f, 'w') for f in files]
        for img_id, dets_per_cls in zip(id_list, dets_list):
            for f, dets in zip(file_objs, dets_per_cls):
                if dets.size == 0:
                    continue
                bboxes = obb2poly_np(dets, self.version)
                for bbox in bboxes:
                    txt_element = [img_id, str(bbox[-1])
                                   ] + [f'{p:.2f}' for p in bbox[:-1]]
                    f.writelines(' '.join(txt_element) + '\n')

        for f in file_objs:
            f.close()

        target_name = osp.split(out_folder)[-1]
        with zipfile.ZipFile(
                osp.join(out_folder, target_name + '.zip'), 'w',
                zipfile.ZIP_DEFLATED) as t:
            for f in files:
                t.write(f, osp.split(f)[-1])

        return files

    def format_results(self, results, submission_dir=None, nproc=4, **kwargs):
        """Format the results to submission text (standard format for DOTA
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            submission_dir (str, optional): The folder that contains submission
                files. If not specified, a temp folder will be created.
                Default: None.
            nproc (int, optional): number of process.

        Returns:
            tuple:

                - result_files (dict): a dict containing the json filepaths
                - tmp_dir (str): the temporal directory created for saving \
                    json files when submission_dir is not specified.
        """
        nproc = min(nproc, os.cpu_count())
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            f'The length of results is not equal to '
            f'the dataset len: {len(results)} != {len(self)}')
        if submission_dir is None:
            submission_dir = tempfile.TemporaryDirectory()
        else:
            tmp_dir = None

        print('\nMerging patch bboxes into full image!!!')
        start_time = time.time()
        id_list, dets_list = self.merge_det(results, nproc)
        stop_time = time.time()
        print(f'Used time: {(stop_time - start_time):.1f} s')

        result_files = self._results2submission(id_list, dets_list,
                                                submission_dir)

        return result_files, tmp_dir


def merge_function(info, CLASSES, iou_thr, device='cpu'):
    """
    Merging function for a single image  with GPU support
    """
    img_id, label_dets = info
    # Concatenate all detection results for the current image in one go
    label_dets = np.concatenate(label_dets, axis=0)

    if len(label_dets) == 0:
        return img_id, [np.empty((0, 5), dtype=np.float32) for _ in CLASSES]

    # Extract labels and detections
    labels = label_dets[:, 0]
    dets = label_dets[:, 1:]

    ori_img_results = []

    # Convert numpy arrays to torch tensors once, and move to device
    all_bboxes = torch.from_numpy(dets[:, :-1]).to(device).contiguous()
    all_scores = torch.from_numpy(dets[:, -1]).to(device).contiguous()
    all_labels = torch.from_numpy(labels).to(device).to(torch.int64)

    # Sanity check for bounding box format
    if all_bboxes.shape[1] == 4:
        print("Warning: Bounding box format is (N, 4). `nms_rotated` expects (N, 5).")
        pass
    elif all_bboxes.shape[1] == 5:
        pass
    else:
        raise ValueError(f"Unsupported bounding box shape: {all_bboxes.shape[1]}")

    for i in range(len(CLASSES)):
        # Filter detections for the current class efficiently using boolean indexing
        cls_mask = (all_labels == i)

        # If no detections for this class, add an empty array
        if not torch.any(cls_mask):
            ori_img_results.append(np.empty((0, dets.shape[1]), dtype=np.float32))
            continue

        cls_bboxes = all_bboxes[cls_mask]
        cls_scores = all_scores[cls_mask]

        # Apply NMS on the selected detections using the MMCV CUDA operator
        # This is where the major speed-up comes from.
        results, inds = nms_rotated(cls_bboxes, cls_scores, iou_thr)

        # Move results back to CPU and convert to numpy for returning
        results_np = results.cpu().numpy()
        ori_img_results.append(results_np)

    return img_id, ori_img_results