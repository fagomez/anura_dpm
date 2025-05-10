import torch 
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, in_out_dim=16000, **kwargs):
        super(VAE, self).__init__()

        latent_dim = kwargs['latent_dim']
        encode_dims = kwargs['encode_dims']
        encode_dims *= int(kwargs['expand_encode_dims'])
        decode_dims = kwargs['decode_dims']
        decode_dims *= int(kwargs['expand_decode_dims'])
        
        self.latent_dim = latent_dim
        # Encoder
        self.fc1 = nn.Linear(in_out_dim, encode_dims[0])  # Input (flattened image) to hidden
        self.fc2_n = nn.ModuleList([
            nn.Linear(encode_dims[i - 1], encode_dims[i]) for i in range(1, len(encode_dims))
        ])
        self.fc21 = nn.Linear(encode_dims[-1], latent_dim)  # Hidden to mean
        self.fc22 = nn.Linear(encode_dims[-1], latent_dim)  # Hidden to log-variance

        # Decoder
        self.fc3 = nn.Linear(latent_dim, decode_dims[0])  # Latent to hidden
        self.fc3_n = nn.ModuleList([
            nn.Linear(decode_dims[i - 1], decode_dims[i]) for i in range(1, len(decode_dims))
        ])

        self.fc4 = nn.Linear(decode_dims[-1], in_out_dim)  # Hidden to output (reconstruction)

    def encode(self, x):
        h1 = F.relu(self.fc1(x)) 
        for linear_layer in self.fc2_n:
            h1 = F.relu(linear_layer(h1))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Sample from standard normal distribution
        return eps * std + mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        for linear_layer in self.fc3_n:
            h3 = F.relu(linear_layer(h3))
        return self.fc4(h3)  # Sigmoid for image pixel values

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar