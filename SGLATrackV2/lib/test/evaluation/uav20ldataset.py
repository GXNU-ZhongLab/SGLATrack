import os
import json
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class UAV20LDataset(BaseDataset):
    """ UAV20L dataset.

    Publication:
        A Benchmark and Simulator for UAV Tracking.
        Matthias Mueller, Neil Smith and Bernard Ghanem
        ECCV, 2016
        https://ivul.kaust.edu.sa/Documents/Publications/2016/A%20Benchmark%20and%20Simulator%20for%20UAV%20Tracking.pdf

    Download the dataset from https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx
    """
    def __init__(self, attribute=None):
        super().__init__()
        self.base_path = '/home/zhongbineng/dataset/UAV123/data_seq/UAV123'
        self.sequence_info_list = self._get_sequence_info_list()

        self.att_dict = None

        if attribute is not None:
            self.sequence_info_list = self._filter_sequence_info_list_by_attribute(attribute, self.sequence_info_list)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path, 
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'uav20l', ground_truth_rect[init_omit:,:],
                        object_class=sequence_info['object_class'])

    def get_attribute_names(self, mode='short'):
        if self.att_dict is None:
            self.att_dict = self._load_attributes()

        names = self.att_dict['att_name_short'] if mode == 'short' else self.att_dict['att_name_long']
        return names

    def _load_attributes(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'dataset_attribute_specs', 'UAV123_attributes.json'), 'r') as f:
            att_dict = json.load(f)
        return att_dict

    def _filter_sequence_info_list_by_attribute(self, att, seq_list):
        if self.att_dict is None:
            self.att_dict = self._load_attributes()

        if att not in self.att_dict['att_name_short']:
            if att in self.att_dict['att_name_long']:
                att = self.att_dict['att_name_short'][self.att_dict['att_name_long'].index(att)]
            else:
                raise ValueError('\'{}\' attribute invalid.')

        # return [s for s in seq_list if att in self.att_dict[s['name'][4:]]]  # UAV123在pytracking中加了前缀uav_
        return [s for s in seq_list if att in self.att_dict[s['name'][0:]]]

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "bike1", "path": "bike1", "startFrame": 1, "endFrame": 3085, "nz": 6,
             "ext": "jpg", "anno_path": "../../anno/UAV20L/bike1.txt",
             "object_class": "vehicle"},

            {"name": "bird1", "path": "bird1", "startFrame": 1, "endFrame": 2437, "nz": 6,
             "ext": "jpg", "anno_path": "../../anno/UAV20L/bird1.txt",
             "object_class": "bird"},

            {"name": "car1", "path": "car1", "startFrame": 1, "endFrame": 2629, "nz": 6,
             "ext": "jpg", "anno_path": "../../anno/UAV20L/car1.txt",
             "object_class": "car"},

            {"name": "car16", "path": "car16", "startFrame": 1, "endFrame": 1993, "nz": 6,
             "ext": "jpg", "anno_path": "../../anno/UAV20L/car16.txt",
             "object_class": "car"},

            {"name": "car3", "path": "car3", "startFrame": 1, "endFrame": 1717, "nz": 6,
             "ext": "jpg", "anno_path": "../../anno/UAV20L/car3.txt",
             "object_class": "car"},

            {"name": "car6", "path": "car6", "startFrame": 1, "endFrame": 4861, "nz": 6,
             "ext": "jpg", "anno_path": "../../anno/UAV20L/car6.txt",
             "object_class": "car"},

            {"name": "car8", "path": "car8", "startFrame": 1, "endFrame": 2575, "nz": 6,
             "ext": "jpg", "anno_path": "../../anno/UAV20L/car8.txt",
             "object_class": "car"},

            {"name": "car9", "path": "car9", "startFrame": 1, "endFrame": 1879, "nz": 6,
             "ext": "jpg", "anno_path": "../../anno/UAV20L/car9.txt",
             "object_class": "car"},
            {"name": "group1", "path": "group1", "startFrame": 1, "endFrame": 4873, "nz": 6,
             "ext": "jpg", "anno_path": "../../anno/UAV20L/group1.txt",
             "object_class": "person"},

            {"name": "group2", "path": "group2", "startFrame": 1, "endFrame": 2683, "nz": 6,
             "ext": "jpg", "anno_path": "../../anno/UAV20L/group2.txt",
             "object_class": "person"},

            {"name": "group3", "path": "group3", "startFrame": 1, "endFrame": 5527, "nz": 6,
             "ext": "jpg", "anno_path": "../../anno/UAV20L/group3.txt",
             "object_class": "person"},

            {"name": "person14", "path": "person14", "startFrame": 1, "endFrame": 2923, "nz": 6,
             "ext": "jpg", "anno_path": "../../anno/UAV20L/person14.txt",
             "object_class": "person"},

            {"name": "person17", "path": "person17", "startFrame": 1, "endFrame": 2347, "nz": 6,
             "ext": "jpg", "anno_path": "../../anno/UAV20L/person17.txt",
             "object_class": "person"},

            {"name": "person19", "path": "person19", "startFrame": 1, "endFrame": 4357, "nz": 6,
             "ext": "jpg", "anno_path": "../../anno/UAV20L/person19.txt",
             "object_class": "person"},

            {"name": "person2", "path": "person2", "startFrame": 1, "endFrame": 2623, "nz": 6,
             "ext": "jpg", "anno_path": "../../anno/UAV20L/person2.txt",
             "object_class": "person"},

            {"name": "person20", "path": "person20", "startFrame": 1, "endFrame": 1783, "nz": 6,
             "ext": "jpg", "anno_path": "../../anno/UAV20L/person20.txt",
             "object_class": "person"},

            {"name": "person4", "path": "person4", "startFrame": 1, "endFrame": 2743, "nz": 6,
             "ext": "jpg", "anno_path": "../../anno/UAV20L/person4.txt",
             "object_class": "person"},

            {"name": "person5", "path": "person5", "startFrame": 1, "endFrame": 2101, "nz": 6,
             "ext": "jpg", "anno_path": "../../anno/UAV20L/person5.txt",
             "object_class": "person"},

            {"name": "person7", "path": "person7", "startFrame": 1, "endFrame": 2065, "nz": 6,
             "ext": "jpg", "anno_path": "../../anno/UAV20L/person7.txt",
             "object_class": "person"},

            {"name": "uav1", "path": "uav1", "startFrame": 1, "endFrame": 3469, "nz": 6,
             "ext": "jpg", "anno_path": "../../anno/UAV20L/uav1.txt",
             "object_class": "aircraft"}
        ]

        return sequence_info_list
