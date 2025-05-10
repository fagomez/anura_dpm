import os
import shutil
import sys
import json

import argparse
import models
from sampling.sampling_utils import get_vae_sample, get_diffusion_sample

sys.path.append(os.path.abspath("../model_utils"))
from model_utils.diffusion_utils import calc_diffusion_hyperparams

import soundfile as sf
import torch

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", choices=models.available_models, default=models.available_models[0], help='model of GEN AI')
parser.add_argument("--number-samples", type=int, default=10, help='number of samples')
parser.add_argument("--folder-generated-data", type=str, default='../data/', help='path of target folder')
parser.add_argument("--config", type=str, default='./config.json', help='path of difussion config')
args = parser.parse_args()
model_name = args.model
n_samples = args.number_samples

with open(args.config, 'r') as f:
    config = json.load(f)
    sr = config['audio_config']['sample_rate']

with open(args.config, 'r') as f:
    config = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
generation_folder = f'{args.folder_generated_data}/generated/{model_name}/BOAFAB'

if os.path.exists(generation_folder):
    shutil.rmtree(generation_folder)
    os.makedirs(generation_folder)
else:
    os.makedirs(generation_folder)

if model_name == 'VAE':
    model = torch.load(f'{model_name}_model.pth', map_location=device).to(device)
    samples = get_vae_sample(model, n_samples)
elif model_name == 'DIFFUSION':
    diffusion_config = config['diffusion_config']
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)
    model = torch.load(f'{model_name}_model.pth', map_location=device).to(device)
    samples = get_diffusion_sample(model, (n_samples, 1, sr), diffusion_hyperparams)

for n, s in enumerate(samples):
    s = s[0].cpu()
    to_save_audio = sf.write(f'{generation_folder}/gen_{n}.wav', s, sr)