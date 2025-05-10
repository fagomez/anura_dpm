from .vae import VAE
from .WaveNet import WaveNet_Speech_Commands as WaveNet

available_models = [
    'VAE', 'DIFFUSION'
]

def create_model(model_name, config=None):
    '''
    Create the model

    Args:
    model_name (str): The model name
    config (dict): The configuration dictionary

    Returns:
    model: The model
    '''
    if model_name == "VAE":
        sr = config['audio_config']['sample_rate']
        secs = config['audio_config']['time_sample_sec']
        in_out_dim = sr * secs

        vae_config = config['VAE_configuraciones']
        model = VAE(in_out_dim=in_out_dim, **vae_config)
    elif model_name == "DIFFUSION":
        net_config = config['wavenet_config']
        model = WaveNet(**net_config)

    return model