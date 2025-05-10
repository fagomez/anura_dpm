import torch
import os
from functools import partial

os.chdir('../3_classifier')

from model_utils import (LoadAudio, FixAudioLength, 
                         ToMelSpectrogram, ToTensor, SpeechCommandsDataset)
from torchvision.transforms import Compose
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
from scipy import linalg
from tqdm import tqdm


def get_intermediate_output(model, layer_name, input_tensor):
  output = input_tensor  
  for name, module in list(model._modules.items())[0][1]._modules.items():
      output = module(output)
      if name == layer_name:
          return output

def get_embedding(model, dataset_dir, class_c, batch_processing=1, sample_size=np.inf):
    n_mels = 32
    feature_transform = Compose([
        ToMelSpectrogram(n_mels=n_mels), 
        ToTensor('mel_spectrogram', 'input')])
    transform = Compose([
        LoadAudio(), 
        FixAudioLength(), 
        feature_transform])

    test_dataset = SpeechCommandsDataset(dataset_dir, transform, silence_percentage=0, class_c=class_c)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_processing)

    partial_function = partial(get_intermediate_output, model, 'stage_3')

    pbar = tqdm(test_dataloader, unit="audios", unit_scale=test_dataloader.batch_size)
    matrix = None

    paths = []
    for batch in pbar:
        paths += batch['path']
        inputs = batch['input']
        inputs = torch.unsqueeze(inputs, 1)

        with torch.no_grad():
            inputs = Variable(inputs)
            emb = partial_function(inputs)
            x = F.avg_pool2d(emb, 8, 1)
            x = x.view(-1, model.module.stages[3]).numpy()

            if matrix is None:
                matrix = x
            else:
                matrix = np.vstack((matrix, x))  
        if matrix.shape[0] >= sample_size:
            break

    return matrix, paths


def calculate_fid(real_sample, generated_sample, verbose=False):
    """Calculates the Fréchet Inception Distance (FID)"""
    mu1, sigma1 = real_sample.mean(axis=0), np.cov(real_sample, rowvar=False)
    mu2, sigma2 = generated_sample.mean(axis=0), np.cov(generated_sample, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    term_1 = ssdiff
    term_2 = np.trace(sigma1 + sigma2 - 2.0 * covmean)
    if verbose:
        print("FID (termino 1): ", term_1)
        print("FID (termino 2): ", term_2)
    fid = term_1 + term_2
    return fid

