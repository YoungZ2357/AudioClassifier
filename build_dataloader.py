# -*- coding: utf-8 -*-
# @Time    : 2024/6/11 15:48
# @Author  : Qingyang Zhang
# @File    : build_dataloader.py
# @Project : AudioClassifier

from torch.utils.data import Dataset, DataLoader
from preprocess import *


class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = [file for file in os.listdir(root_dir) if file.endswith(".wav")]
        self.max_length = get_desired_length(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """Name of the file should be: {identifier}_{label}.wav
        The type of the label should be string ,int or one-hot encoding

        :param idx:
        :return: (1d audio tensor, mel spectrotram, label)
        """
        name = self.file_list[idx]
        # process X value
        file_path = os.path.join(self.root_dir, name)
        wav, sample_rate = torchaudio.load(file_path)
        wav = pad_wav(single_wav=wav, desired_length=self.max_length)
        mel_spec = to_melspec(wav, sample_rate=sample_rate)
        # process label
        label = name.split("_")[-1].strip(".wav")
        return wav, mel_spec, label


def get_loader(
        root_dir: str,
        batch_size: int,
        num_workers: int = 1,
        shuffle: bool = True
) -> DataLoader:
    dataset = AudioDataset(root_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )