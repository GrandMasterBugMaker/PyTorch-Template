import os
import sys
import argparse
import time
import random
import wandb
from tqdm import tqdm
import numpy as np
import torch
import hiddenlayer as hl
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from networks import MyModel, weight_init
from utils.util import EarlyStopping

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
# 取每个 GPU 的剩余显存数，并存放到 tmp 文件中
os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
memory_gpu = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
torch.cuda.set_device(np.argmax(memory_gpu))
os.system("rm tmp")  # 删除临时生成的 tmp 文件
##命令行执行
# CUDA_VISIBLE_DEVICES=0,1 python train.py
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(description="Train a network for ...")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--path_to_checkpoint", type=str, default="../checkpoint/")

    opt = parser.parse_args()
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    wandb.init(project="my-project")
    wandb.config.xxx = opt.xxx
    # 准备数据
    train_dataset = xxxxxx  # 定义的数据集
    validation_dataset = xxx
    train_data_loader = data.DataLoader(
        train_dataset,
        batch_size=xxxx,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    validation_data_loader = data.DataLoader(
        validation_dataset,
        batch_size=xxxx,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device_ids = [0, 1]

    # 自定义的model
    model = MyModel(opt)
    model.to(device)
    model.apply(weight_init)

    # 并行运算，如果需要的话
    model = nn.DataParallel(model, device_ids=device_ids).to(device)
    # summary(model, input_size=(channels, H, W))
    # hl.build_graph(model, torch.zeros([1, 2, 3]))

    # loss function， 比如交叉熵
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    wandb.watch(model, criterion)

    # optimizer，比如Adam
    optimizer = optim.Adam(model.parameters())
    # 调整学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    # 训练
    num_epochs = opt.epochs

    start_epoch = 0
    if opt.resume:
        ckpt = torch.load(
            opt.path_to_checkpoint + "models.pt"
        )  # custom method for loading last checkpoint
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        model = wandb.restore("mymodel.h5")

        print("last checkpoint restored")
        ...
    # default `log_dir`is "runs" - we'll be more specific here
    writer = SummaryWriter("runs/models")

    def train(start_epoch, num_epochs):
        early_stopping = EarlyStopping(opt, verbose=True)
        for epoch in tqdm(range(start_epoch, num_epochs)):
            torch.cuda.empty_cache()
            print("Epoch: %d" % epoch)
            train_losses = []
            model.train()
            past = time.time()
            for inputs, targets in train_data_loader:

                # models.zero_grad()
                # optimizer.zero_grad()#当optimizer=optim.Optimizer(models.parameters())时，两者等效
                # 梯度清零
                optimizer.zero_grad()

                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                # if (batch_idx + 1) % num == 0:
                # 	print(batch_idx + 1, len(dataloader), 'Loss: %.3f' % (train_loss / num))
                # 	train_loss = 0
            now = time.time()
            train_loss = np.mean(np.array(train_losses))
            print(epoch, "loss:%.3f,time:%.2fs" % (train_loss, now - past))
            writer.add_scalar("train_loss", train_loss, epoch)
            wandb.log({"train_loss": train_loss}, step=epoch)
            train_loss = 0
            # checkpoint = {
            #     "model_state_dict": models.module.state_dict(),
            #     "opt_state_dict": optimizer.state_dict(),
            #     "epoch": epoch,
            # }

            scheduler.step()
            # the end of one epoch
            model.eval()
            val_losses = []
            with torch.no_grad():
                for inputs, targets in validation_data_loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_losses.append(loss.item())
            val_loss = np.mean(np.array(val_losses))
            wandb.log({"val_loss": val_loss}, step=epoch)

            #####some testing#####
            print("xxxxxxx".format(xxxxxxx))
            #####some logging#####
            prefix = opt.path_to_checkpoint + opt.hidden_size + "_"
            file = prefix + "xxx_xxx_xxx.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                file
            )
            wandb.save("mymodel.h5")
            early_stopping(val_loss, model, opt)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    train(start_epoch, num_epochs)
