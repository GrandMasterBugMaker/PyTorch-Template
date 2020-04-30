import os
import sys
import argparse
import time
import random
import wandb
from tqdm import tqdm
import numpy as np
import numba as nb
import pandas as pd
import torch
import hiddenlayer as hl
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from models.module import Model
from data.custom_dataset import MyDataset

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


def test():
    last = time.time()
    torch.cuda.empty_cache()
    test_losses = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_losses.append(loss.item())
    val_loss = np.mean(np.mean(test_losses))


if __name__ == "__main__":
    # #取每个 GPU 的剩余显存数，并存放到 tmp 文件中
    # os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
    # memory_gpu = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    # torch.cuda.set_device(np.argmax(memory_gpu))
    # os.system("rm tmp")  # 删除临时生成的 tmp 文件
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    ##命令行执行
    # CUDA_VISIBLE_DEVICES=0,1 python train.py
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(description="Train a network for ...")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--path_to_checkpoint", type=str, default="../checkpoint")

    opt = parser.parse_args()
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    wandb.init(project="my-project")
    wandb.config.xxx = opt.xxx
    # 准备数据
    test_dataset = MyDataset('test_dataset_path')  # 定义的数据集
    test_data_loader = data.DataLoader(
        test_dataset, batch_size=128, shuffle=True, drop_last=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device_ids = [0, 1]

    model = Model(opt)
    ckpt = torch.load(
        opt.path_to_checkpoint + "lowest_val_loss_model.pt"
    )  # custom method for loading last checkpoint
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # 并行运算，如果需要的话
    # model = nn.DataParallel(model, device_ids=device_ids).to(device)
    # summary(model, input_size=(channels, H, W))
    # hl.build_graph(model, torch.zeros([1, 2, 3]))

    # loss function， 比如交叉熵
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    wandb.watch(model, criterion)

    writer = SummaryWriter("runs/models")

    test()
