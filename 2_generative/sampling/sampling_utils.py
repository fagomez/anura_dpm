import torch
from tqdm import tqdm

@torch.no_grad()
def get_vae_sample(model, num_samples):
    device = next(model.parameters()).device 
    with torch.no_grad():
        z = torch.randn(num_samples, 1, model.latent_dim).to(device)
        samples = model.decode(z)

        samples = (samples * 2) - 1 # TODO: Revisar que sea necesaria esta inversion

    return samples

@torch.no_grad()
def get_diffusion_sample(model, size, diffusion_hyperparams):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated,
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    print('begin sampling, total number of reverse steps = %s' % T)
    device = next(model.parameters()).device 
    x = std_normal(size).to(device)
    with torch.no_grad():
        for t in tqdm(range(T-1, -1, -1), desc='Reverse process'):
            diffusion_steps = (t * torch.ones((size[0], 1))).to(device)  # use the corresponding reverse step
            epsilon_theta = model((x, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
            if t > 0:
                x = x + Sigma[t] * std_normal(size).to(device)  # add the variance term to x_{t-1}
    return x

@torch.no_grad()
def get_diffusion_sample_interim_steps(model, size, diffusion_hyperparams, list_steps=[199, 100, 50, 0]):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated,
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    print('begin sampling, total number of reverse steps = %s' % T)
    res = {}
    device = next(model.parameters()).device 
    x = std_normal(size).to(device)
    with torch.no_grad():
        for t in tqdm(range(T-1, -1, -1), desc='Reverse process'):
            diffusion_steps = (t * torch.ones((size[0], 1))).to(device)  # use the corresponding reverse step
            epsilon_theta = model((x, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
            if t > 0:
                x = x + Sigma[t] * std_normal(size).to(device)  # add the variance term to x_{t-1}
            if t in list_steps:
                res[t] = x
    return res

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size)