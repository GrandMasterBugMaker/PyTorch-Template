#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
# @Author:      Bai Lingnan
# @Project:     Pytorch-Template
# @Filename:    custom_dataset.py
# @Time:        2020/3/12 10:48
"""
import os
import random
import swifter
import numpy as np
import numba as nb
from torch.utils.data import Dataset
"""
tricks:
1.torch-optimizer:实现了最新的一些优化器.
2.fastai:有一些评价指标
3.numba:import numba as nb,纯python或numpy加速,加@nb.njit或@nb.jit(nopython=True)
4.swifter:df.apply()→·df.swifter.apply()，加速pandas
5.captum:可解释性
6.cupy:加速pandas,1000万以上数据更快
7.modin:import modin.pandas as mdpd,用mdpd代替pd即可，加速pandas,加载数据和查询数据更快,统计方法pandas更快
"""

class DataProcessor(object):
    """
    Base class for data converters for sequence classification data sets.
    """
    def __init__(self, opt):
        self.opt = opt
    def get_train_examples(self):
        return self._create_example(
            self._read_data(os.path.join(self.opt.data_dir, "train"))
        )

    def get_dev_examples(self):
        return self._create_example(
            self._read_data(os.path.join(self.opt.data_dir, 'dev'))
        )

    def get_test_examples(self):
        return self._create_example(
            self._read_data(os.path.join(self.opt.data_dir, "test"))
        )


    def _read_data(self, input_file):
        """
        此处填写代码
        """
        return data

    def _create_example(self, data):

        return list(zip(x1s, x2s, x3s)

class ModelDataSet(Dataset):

    def __init__(self, opt,case='train'):
        self.opt = opt
        self.data_processer = DataProcessor(opt)
        
        self.x=self.load_data(case)

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)
    def load_data(self, case):
        if case == 'train':
            return self.data_processer.get_train_examples()
        elif case == 'dev':
            return self.data_processer.get_dev_examples()

        return self.data_processer.get_test_examples()



