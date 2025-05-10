import sys
import os
from functools import partial

sys.path.append(os.path.abspath("../model_utils"))
from model_utils.diffusion_utils import calc_diffusion_hyperparams

from .vae import compute_vae_training_loss
from .diffusion import compute_diffusion_training_loss
available_models = [
    'VAE', 'DIFFUSION'
]

def get_loss_train(model_name, general_config=None):
    '''
    Get the loss training function for the model

    Args:
    model_name (str): The model name
    general_config (dict): The configuration dictionary

    Returns:
    loss_training_function (function): The loss training function
    '''
    if model_name == "VAE":
        loss_training_function = compute_vae_training_loss
    elif model_name == "DIFFUSION":
        diffusion_config = general_config['diffusion_config']
        diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)
        loss_training_function = partial(
            compute_diffusion_training_loss,
            diffusion_hyperparams=diffusion_hyperparams)

    return loss_training_function