import glob

import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class UAVTrackDataset(BaseDataset):
    """ GOT-10k dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """
    def __init__(self):
        super().__init__()

        self.base_path = os.path.join(self.env_settings.uavtrack_path)
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):

        anno_path = '{}/{}/{}.txt'.format(self.base_path, 'anno_l',sequence_name)

        ground_truth_rect = np.loadtxt(anno_path, delimiter=',')

        frames_path = '{}/{}/{}'.format(self.base_path, 'data_seq',sequence_name)
        frames_list = sorted(glob.glob(os.path.join(frames_path, '*.jpg')))
        return Sequence(sequence_name, frames_list, 'uavtrack', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        seq_path = os.path.join(self.base_path,'anno_l')
        seqs = os.listdir(seq_path)
        name_list = []
        for seqname in seqs:
            name_list.append(seqname[:-4])
        return name_list
