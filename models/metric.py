#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
# @Author:      Bai Lingnan
# @Project:     Pytorch-Template
# @Filename:    metric.py
# @Time:        2020/3/12 10:15
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
# #自定义损失函数
    # class CustomLoss(nn.Module):
    #
    #     def __init__(self):
    #         super(CustomLoss, self).__init__()
    #
    #     def forward(self, x, y):
    #         loss = torch.mean((x - y) ** 2)
    #         return loss

