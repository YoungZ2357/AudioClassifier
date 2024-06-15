# -*- coding: utf-8 -*-
# @Time    : 2024/6/10 23:53
# @Author  : Qingyang Zhang
# @File    : tmp.py
# @Project : AudioClassifier
import torchaudio
import matplotlib.pyplot as plt
import torchaudio.transforms as transforms
from network import *


f = r"data/ev_h_-5_0006.wav"
w, s = torchaudio.load(f)

mel_spectrogram = transforms.MelSpectrogram(
    sample_rate=s,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=64
)

mel_spec = mel_spectrogram(w)
db_transform = transforms.AmplitudeToDB()
log_mel_spec = db_transform(mel_spec)
# print(log_mel_spec.shape)
# print(log_mel_spec)
# plt.imshow(log_mel_spec[0].detach().numpy(), cmap="viridis")
# plt.colorbar(format='%+2.0f dB')
# plt.show()


model = ResNetCNN(ResidualBlock, [2, 2, 2, 2], 128, 1)
model2 = GRURNN(1, 128, 128)
print(log_mel_spec.shape)
y_hat = model.forward(log_mel_spec.unsqueeze(0))
