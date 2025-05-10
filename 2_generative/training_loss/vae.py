import torch
import torch.nn.functional as F


def compute_vae_training_loss(model, batch):
    recon_batch, mu, logvar = model(batch)
    loss = get_loss(recon_batch, batch, mu, logvar)

    return loss


def get_loss(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print(BCE)
    # print(KLD)
    return BCE + KLD