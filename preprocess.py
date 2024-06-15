# -*- coding: utf-8 -*-
# @Time    : 2024/6/10 23:52
# @Author  : Qingyang Zhang
# @File    : preprocess.py
# @Project : AudioClassifier
import os
import torch

import torchaudio
from torchaudio.transforms import TimeStretch
import torchaudio.transforms as transforms


def get_desired_length(wav_dir) -> int:
    """get the largets length in the dataset

    :param wav_dir:
    :return: largest length
    """
    l_list = []
    tmp = os.listdir(wav_dir)
    for elem in tmp:
        template = f"data/{elem}"
        wav, n_s = torchaudio.load(template)
        l_list.append(wav.shape[-1])
    return max(l_list)


def pad_wav(single_wav: torch.Tensor, desired_length) -> torch.Tensor:
    """reshape audio tensor into a desired length by padding or slicing

    :param single_wav: audio tensor variable
    :param desired_length: target length of the tensor, usually the largest length of the dataset
    :return: reshaped audio tensor
    """
    audio_length = single_wav.shape[-1]
    if audio_length < desired_length:
        padding = torch.zeros(1, desired_length - single_wav.shape[-1])
        result = torch.cat([single_wav, padding], dim=1)
    else:
        result = single_wav[:, :desired_length]
    return result


def stretch_wav(single_wav: torch.Tensor, desired_length, sample_rate) -> torch.Tensor:
    """reshape audio tensor into a desired length by time strenching
    !!!DEPRECATED!!!
    !!!Planing to implement
    :param single_wav: audio tensor variable
    :param desired_length: desired_length: target length of the tensor
    :param sample_rate: sample rate of the audio
    :return: reshaped audio tensor
    """
    audio_length = single_wav.shape[-1]
    stretch_factor = desired_length / audio_length
    transform = TimeStretch(n_freq=sample_rate // 2, fixed_rate=stretch_factor)
    result = transform(single_wav, stretch_factor)
    return result


def to_melspec(
        wav: torch.Tensor,
        sample_rate: int,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 64,
        log_transform: bool = True
) -> torch.Tensor:
    """

    :param wav:
    :param sample_rate:
    :param n_fft:
    :param win_length:
    :param hop_length:
    :param n_mels:
    :param log_transform:
    :return:
    """
    mel_spec_t = transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spec = mel_spec_t(wav)
    if log_transform:
        mel_spec = transforms.AmplitudeToDB()(mel_spec)
    return mel_spec
