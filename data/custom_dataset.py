#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Author:      Bai Lingnan
# @Project:     Pytorch-Template
# @Filename:    custom_dataset.py
# @Time:        2020/3/12 10:48
"""
"""
tricks:
1.torch-optimizer:实现了最新的一些优化器.
2.numba:import numba as nb,纯python或numpy加速,加@nb.njit或@nb.jit(nopython=True)
3.swifter:df.apply()→·df.swifter.apply()，加速pandas
4.cupy:1000万以上数据更快
5.modin:import modin.pandas as mdpd,用mdpd代替pd即可，加速pandas,加载数据和查询数据更快,统计方法pandas更快
"""
import os
import random
import swifter
import numpy as np
import numba as nb
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader


# class DataProcessor(object):
#     """
#     Base class for data converters for sequence classification data sets.
#     """
#     def __init__(self, opt):
#         self.opt = opt
#     def get_train_examples(self):
#         return self._create_example(
#             self._read_data(os.path.join(self.opt.data_dir, "train"))
#         )
#
#     def get_dev_examples(self):
#         return self._create_example(
#             self._read_data(os.path.join(self.opt.data_dir, 'dev'))
#         )
#
#     def get_test_examples(self):
#         return self._create_example(
#             self._read_data(os.path.join(self.opt.data_dir, "test"))
#         )
#
#
#     def _read_data(self, input_file):
#         """
#         此处填写代码
#         """
#         return data
#
#     def _create_example(self, data):
#
#         return list(zip(x1s, x2s, x3s)

# class ModelDataSet(Dataset):
#
#     def __init__(self, opt,case='train'):
#         self.opt = opt
#         self.data_processer = DataProcessor(opt)
#
#         self.x=self.load_data(case)
#
#     def __getitem__(self, idx):
#         assert idx < len(self.x)
#         return self.x[idx]
#
#     def __len__(self):
#         return len(self.x)
#     def load_data(self, case):
#         if case == 'train':
#             return self.data_processer.get_train_examples()
#         elif case == 'dev':
#             return self.data_processer.get_dev_examples()
#
#         return self.data_processer.get_test_examples()
"""
示例一：
data_df = pd.read_csv("xxx.csv")
feature_col = [
    "No",
    "year",
    "month",
    "day",
    "hour",
    "DWEP",
    "TEMP",
    "PRES",
    "Iws",
    "Is",
    "Ir",
]
data_df_x = data_df.loc[:127, feature_col]
label_col = ["pm2.5"]
data_df_y = data_df.loc[:127, label_col]

data_numpy_x=data_df_x.values
data_numpy_y=data_df_y.values

X=torch.from_numpy(data_numpy_x)
Y=torch.from_numpy(data_numpy_y)

dataset=TensorDataset(X,Y)
dataloader=DataLoader(dataset=dataset,batch_size=64,shuffle=True,)

利用TensorDataset的时候传入的应该是tensor类型，如果是df需要先转换成numpy.array在转换成tensor，输出的也是tensor，事情其实可以分为以下三步：

1.加载数据，提取出feature和label，并转换成tensor
2. 传入TensorDataset中，实例化TensorDataset为datsset
3. 再将dataset传入到Dataloader中，最后通过enumerate输出我们想要的经过shuffle的bachsize大小的feature和label数据
"""


class MyDataset(Dataset):
    def __init__(self, path):
        data_df = pd.read_csv("path")
        feature_col = [
            "No",
            "year",
            "month",
            "day",
            "hour",
            "DWEP",
            "TEMP",
            "PRES",
            "Iws",
            "Is",
            "Ir",
        ]
        data_df_x = data_df.loc[:127, feature_col]
        label_col = ["pm2.5"]
        data_df_y = data_df.loc[:127, label_col]

        data_numpy_x = data_df_x.values
        data_numpy_y = data_df_y.values

        self.X = torch.from_numpy(data_numpy_x)
        self.Y = torch.from_numpy(data_numpy_y)

        self.len = data_numpy_x.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len
