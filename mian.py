import os
import sys
import argparse
import time
import random
import numpy as np
import torch
import hiddenlayer as hl
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from networks import MyModel, weight_init

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
    parser.add_argument("--path_to_checkpoint", type=str, default="../checkpoint")

    opt = parser.parse_args()
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    # 准备数据
    train_dataset = xxxxxx  # 定义的数据集
    validation_dataset = xxx
    train_data_loader = data.DataLoader(train_dataset, batch_size=xxxx, shuffle=True)
    validation_data_loader = data.DataLoader(
        validation_dataset, batch_size=xxxx, shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device_ids = [0, 1]

    # 自定义的model
    model = MyModel()
    model.to(device)
    model.apply(weight_init)
    # 并行运算，如果需要的话
    model = nn.DataParallel(model, device_ids=device_ids).to(device)
    summary(model, input_size=(channels, H, W))
    hl.build_graph(model, torch.zeros([1, 2, 3]))

    # loss function， 比如交叉熵
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    # #自定义损失函数
    # class CustomLoss(nn.Module):
    #
    #     def __init__(self):
    #         super(CustomLoss, self).__init__()
    #
    #     def forward(self, x, y):
    #         loss = torch.mean((x - y) ** 2)
    #         return loss

    # optimizer，比如Adam
    optimizer = optim.Adam(model.parameters())
    # 调整学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    # 训练
    num_epochs = opt.epochs

    start_epoch = 0
    if opt.resume:
        ckpt = torch.load(
            opt.path_to_checkpoint + "model.pt"
        )  # custom method for loading last checkpoint
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print("last checkpoint restored")
        ...
    # default `log_dir`is "runs" - we'll be more specific here
    writer = SummaryWriter("runs/model")

    def train(start_epoch, num_epochs):
        # num = 20  # 隔多少个batch打印一次
        for epoch in range(start_epoch, num_epochs):
            torch.cuda.empty_cache()
            print("Epoch: %d" % epoch)
            train_batch = 0
            train_loss = 0
            model.train()
            past = time.time()
            for batch_idx, (inputs, targets) in train_data_loader:

                # model.zero_grad()
                # optimizer.zero_grad()#当optimizer=optim.Optimizer(model.parameters())时，两者等效
                # 梯度清零
                optimizer.zero_grad()

                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                train_batch += 1
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # if (batch_idx + 1) % num == 0:
                # 	print(batch_idx + 1, len(dataloader), 'Loss: %.3f' % (train_loss / num))
                # 	train_loss = 0
            now = time.time()
            print(
                epoch, "loss:%.3f,time:%.2fs" % (train_loss / train_batch, now - past)
            )
            writer.add_scalar("train_loss", train_loss / train_batch, epoch)
            train_loss = 0
            # checkpoint = {
            #     "model_state_dict": model.module.state_dict(),
            #     "opt_state_dict": optimizer.state_dict(),
            #     "epoch": epoch,
            # }

            scheduler.step()
            # the end of one epoch
            model.eval()
            with torch.no_grad():
                for batch_idx, (inputs, targets) in validation_data_loader:
                    pass

            #####some testing#####
            print("xxxxxxx".format(xxxxxxx))
            #####some logging#####
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                opt.path_to_checkpoint + "model.pt",
            )

    train(start_epoch, num_epochs)
