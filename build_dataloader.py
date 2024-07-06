# -*- coding: utf-8 -*-
# @Time    : 2024/6/11 15:48
# @Author  : Qingyang Zhang
# @File    : build_dataloader.py
# @Project : AudioClassifier

from torch.utils.data import Dataset, DataLoader, random_split
from preprocess import *


class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = [file for file in os.listdir(root_dir) if file.endswith(".wav")]
        self.max_length = get_desired_length(root_dir)
        self.label_handler = self._get_label_mapper()

    def __len__(self):
        return len(self.file_list)

    def _get_label_mapper(self) -> dict:
        """ turn str type labels into int

        :return:
        """
        labels = [file.split("_")[0] for file in self.file_list]
        unique = sorted(set(labels))
        handler = {
            label: idx for idx, label in enumerate(unique)
        }
        return handler

    def show_label_mapping(self):
        for elem in self.label_handler:
            print(elem)

    def __getitem__(self, idx):
        """Change the methods of processing labels accroding to the name of your files

        :param idx:
        :return: (1d audio tensor, mel spectrotram, label)
        """
        name = self.file_list[idx]

        # process input value
        file_path = os.path.join(self.root_dir, name)
        wav, sample_rate = torchaudio.load(file_path)
        wav = pad_wav(single_wav=wav, desired_length=self.max_length)
        mel_spec = to_melspec(wav, sample_rate=sample_rate)
        # process label
        label = name.split("_")[0]
        label = self.label_handler[label]
        return mel_spec, wav, label


def get_loaders(
        root_dir: str,
        batch_size: int,
        test_split: float = 0.2,
        num_workers: int = 1,
        shuffle: bool = True
) -> tuple[DataLoader, DataLoader]:
    """
    Splits the dataset into training and testing sets and returns the corresponding data loaders.

    :param root_dir: Directory where audio files are stored.
    :param batch_size: Batch size for the data loaders.
    :param test_split: Fraction of the dataset to be used as the test set.
    :param num_workers: Number of worker processes for data loading.
    :param shuffle: Whether to shuffle the dataset before splitting.
    :return: A tuple containing the training data loader and the test data loader.
    """
    dataset = AudioDataset(root_dir)
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader
