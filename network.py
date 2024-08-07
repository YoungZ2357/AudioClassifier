# -*- coding: utf-8 -*-
# @Time    : 2024/6/10 23:46
# @Author  : Qingyang Zhang
# @File    : network.py
# @Project : AudioClassifier
import torch.nn as nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=1):
        super(ResidualBlock, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels * expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * expansion)

        self.downsample = None
        if stride != 1 or in_channels != out_channels * expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * expansion)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNetCNN(nn.Module):
    def __init__(self, block, layers, output_size, expansion=1):
        super(ResNetCNN, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0], expansion=expansion)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2, expansion=expansion)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2, expansion=expansion)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2, expansion=expansion)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * expansion, output_size)

    def make_layer(self, block, out_channels, blocks, stride=1, expansion=1):
        layers = [block(self.in_channels, out_channels, stride, expansion)]
        self.in_channels = out_channels * expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, expansion=expansion))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class GRURNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRURNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


class ResnetGRUNet(nn.Module):
    def __init__(
            self,
            # resenet setting
            resnet_layers: list[int],
            res_block_expasion: int,
            # gru setting
            rnn_hidden: int,
            seq_len: int,
            # tensor concation setting
            tmp_size: int,
            # classifier setting
            n_classes: int
    ):
        super(ResnetGRUNet, self).__init__()
        self.cnn = ResNetCNN(ResidualBlock, resnet_layers, tmp_size, res_block_expasion)
        self.rnn = GRURNN(seq_len, rnn_hidden, tmp_size)
        self.classifier = nn.Sequential(
            nn.Linear(tmp_size * 2, tmp_size),
            nn.Dropout(p=.5),
            nn.SiLU(),
            nn.Linear(tmp_size, int(tmp_size / 2)),
            nn.Dropout(p=.5),
            nn.SiLU(),
            nn.Linear(int(tmp_size/2), n_classes)
        )

    def forward(self, mel_spec, audio_tensor):
        cnn_out = self.cnn(mel_spec)
        rnn_out = self.rnn(audio_tensor)
        # print(f"cnn out shape: {cnn_out.shape}")
        # print(f"rnn out shape: {rnn_out.shape}")
        tmp = torch.cat([cnn_out, rnn_out], dim=1)
        # print(f"tmp shape: {tmp.shape}")
        result = self.classifier(tmp)
        return result

# planning to implement classification based on given feature kinds and its weights

