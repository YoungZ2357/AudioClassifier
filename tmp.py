# -*- coding: utf-8 -*-
# @Time    : 2024/6/10 23:53
# @Author  : Young Zhang
# @File    : tmp.py
# @Project : VoiceClassifier
import torchaudio
import matplotlib.pyplot as plt
import torchaudio.transforms as transforms

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
print(log_mel_spec.shape)
print(log_mel_spec)
plt.imshow(log_mel_spec[0].detach().numpy(), cmap="viridis")
plt.colorbar(format='%+2.0f dB')
plt.show()
