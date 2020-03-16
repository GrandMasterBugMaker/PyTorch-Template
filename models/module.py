#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Author:      Bai Lingnan
# @Project:     Pytorch-Template
# @Filename:    module.py
# @Time:        2020/3/12 10:12
"""
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
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt

    def forward(self, x):
        pass
