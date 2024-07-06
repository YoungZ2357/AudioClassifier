# -*- coding: utf-8 -*-
# @Time    : 2024/6/10 23:45
# @Author  : Qingyang Zhang
# @File    : main.py
# @Project : AudioClassifier
from preprocess import *
from network import *
from build_dataloader import *
from trainer import *

root_dir = r"data"

# hyper parameters
EPOCHS = 100
LR = .01

# params for dataloader
batch_size = 32

# params for neural network
desired_length = get_desired_length(root_dir)
resnet_layers = [2, 2, 2, 2]
block_expansion = 2
rnn_hidden = 64
tmp_size = 256
n_classes = 2


if __name__ == '__main__':

    train_loader, test_loader = get_loaders(root_dir, batch_size)

    model = ResnetGRUNet(resnet_layers, block_expansion, rnn_hidden, desired_length, tmp_size, n_classes)
    train_val_classifier(model, train_loader, test_loader)
    pass
