# -*- coding: utf-8 -*-
# @Time    : 2024/6/11 15:48
# @Author  : Young Zhang
# @File    : build_dataloader.py
# @Project : VoiceClassifier

from torch.utils.data import Dataset, DataLoader
from preprocess import *


class AudioDataset(Dataset):
    def __init__(self, root_dir, mel_spec: bool = True):
        self.root_dir = root_dir
        self.file_list = [file for file in os.listdir(root_dir) if file.endswith(".wav")]
        self.max_length = get_desired_length(root_dir)
        self.mel_spec = mel_spec

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """Name of the file should be: {identifier}_{label}.wav
        The type of the label should be string or int

        :param idx:
        :return: audio tensor, label of the audio
        """
        name = self.file_list[idx]
        # process X value
        file_path = os.path.join(self.root_dir, name)
        wav, sample_rate = torchaudio.load(file_path)
        wav = pad_wav(single_wav=wav, desired_length=self.max_length)
        if self.mel_spec:
            wav = to_melspec(wav, sample_rate=sample_rate)
        # process label
        label = name.split("_")[-1].strip(".wav")
        return wav, label
