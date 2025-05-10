"""Google speech commands dataset."""
__author__ = 'Yuan Xu'

import os
import numpy as np

import librosa

from torch.utils.data import Dataset
import pandas as pd

__all__ = ['SpeechCommandsDataset', 'BackgroundNoiseDataset' ]

# CLASSES = 'unknown, silence, yes, no, up, down, left, right, on, off, stop, go'.split(', ')

class SpeechCommandsDataset(Dataset):
    """Google speech commands dataset. Only 'yes', 'no', 'up', 'down', 'left',
    'right', 'on', 'off', 'stop' and 'go' are treated as known classes.
    All other classes are used as 'unknown' samples.
    See for more information: https://www.kaggle.com/c/tensorflow-speech-recognition-challenge
    """

    def __init__(self, df_data_path, transform=None, silence_percentage=0.1, class_c=None):
        df = pd.read_csv(df_data_path)
        all_classes = np.flip(np.sort(df['class'].unique()))
        #for c in classes[2:]:
        #    assert c in all_classes

        # class_to_idx = {classes[i]: i for i in range(len(classes))}
        class_to_idx = {}
        for n, c in enumerate(all_classes):
            class_to_idx[c] = n


        data = []
        for c in all_classes:
            temp_df = df.loc[df['class'] == c]
            target = class_to_idx[c]
            for f in temp_df['file']:
                path = f'../data/{f}'
                data.append((path, target))

        # add silence
        # target = class_to_idx['silence']
        # data += [('', target)] * int(len(data) * silence_percentage)

        self.classes = list(class_to_idx.keys())
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.data[index]
        data = {'path': path, 'target': target}

        if self.transform is not None:
            data = self.transform(data)
        # print(index)

        return data

    def make_weights_for_balanced_classes(self):
        """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

        nclasses = len(self.classes)
        count = np.zeros(nclasses)
        for item in self.data:
            count[item[1]] += 1

        N = float(sum(count))
        weight_per_class = N / count
        weight = np.zeros(len(self))
        for idx, item in enumerate(self.data):
            weight[idx] = weight_per_class[item[1]]
        return weight

class BackgroundNoiseDataset(Dataset):
    """Dataset for silence / background noise."""

    def __init__(self, folder, transform=None, sample_rate=16000, sample_length=1, classes=None):
        audio_files = [d for d in os.listdir(folder) if os.path.isfile(os.path.join(folder, d)) and d.endswith('.wav')]
        samples = []
        for f in audio_files:
            path = os.path.join(folder, f)
            s, sr = librosa.load(path, sr=sample_rate)
            samples.append(s)

        samples = np.hstack(samples)
        c = int(sample_rate * sample_length)
        r = len(samples) // c
        self.samples = samples[:r*c].reshape(-1, c)
        self.sample_rate = sample_rate
        self.classes = classes
        self.transform = transform
        self.path = folder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = {'samples': self.samples[index], 'sample_rate': self.sample_rate, 'target': 1, 'path': self.path}

        if self.transform is not None:
            data = self.transform(data)

        return data
