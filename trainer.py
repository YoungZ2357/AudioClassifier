# -*- coding: utf-8 -*-
# @Time    : 2024/6/10 23:52
# @Author  : Qingyang Zhang
# @File    : trainer.py
# @Project : AudioClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR


def train_val_classifier(
        # model and DataLoaders
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        # optimizer and scheduler
        optimizer: str = "adam",
        lr_decay: float = .95,
        decay_gap: int = 50,
        # device
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        # hyperparameters
        epochs: int = 300,
        lr: float = .001,
        # other settings
        show_info: bool = True
) -> None:
    model = model.to(device)

    size = len(train_loader.dataset)
    batch = len(train_loader)
    optim_handler = {
        "adam": optim.Adam(model.parameters(), lr=lr)
    }
    optimizer = optim_handler[optimizer]
    criterion = nn.CrossEntropyLoss()
    # update parameters
    scheduler = ExponentialLR(optimizer, gamma=lr_decay)
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        model.train()
        for mel_spec, wav, label in train_loader:
            # print(mel_spec.shape)
            # print(type(label))
            # print(label)
            label = torch.tensor(label)
            mel_spec, wav, label = mel_spec.to(device), wav.to(device), label.to(device)

            optimizer.zero_grad()
            y_hat = model(mel_spec, wav)
            loss = criterion(y_hat, label)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss += loss.item()
                train_acc += (y_hat.argmax(1) == label).type(torch.float).sum().item()
        train_loss /= size
        train_acc /= batch

        if epoch % decay_gap == 0:
            scheduler.step()
        # if show_info:
        #     print(f"EPOCH:{epoch + 1}: Begin validation")
        val_loss = 0
        val_acc = 0
        model.eval()
        size = len(val_loader.dataset)
        batch = len(val_loader)
        for mel_spec_val, wav_val, label_val in val_loader:
            label_val = torch.tensor(label_val)
            mel_spec_val, wav_val, label_val = mel_spec_val.to(device), wav_val, label_val.to(device)
            val_hat = model(mel_spec_val, wav_val)
            loss_val = criterion(val_hat, label_val)
            with torch.no_grad():
                val_loss += loss_val.item()
                val_acc += (val_hat.argmax(1) == label_val).type(torch.float).sum().item()
        val_loss /= batch
        val_acc /= size
        if show_info:
            template = f"{'=' * 20}EPOCH:{epoch + 1:^11}{'=' * 20}\n" \
                       f"{f'train loss:{train_loss}':<12}|{f'train acc:{train_acc}':<12}|" \
                       f"{f'val loss:{val_loss}':<12}|{f'val acc:{val_acc}':<12}\n" \

            print(template)
