from typing import Any, Callable, List

import torch
import pandas as pd
import utils.utils as utils

class LinearVAE(torch.nn.Module):
    def __init__(self,
                 in_size: int | tuple[int, int],
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 activation_func: Callable = torch.nn.GELU,
                 **kwargs
                 ) -> None:
        super(LinearVAE, self).__init__()

         # input shape handling (supports rectangular HxW)
        if isinstance(in_size, tuple):
            in_h, in_w = in_size
        else:
            in_h, in_w = in_size, in_size
        self.in_h = in_h
        self.in_w = in_w
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.activation_func = activation_func
        self.flat_dim = in_h * in_w * in_channels
        if hidden_dims is None:
            hidden_dims = [512, 256, latent_dim]
        self.device = kwargs["device"]  
        encoder_layers = []
        prev_dim = self.flat_dim
        for h_dim in hidden_dims:
            encoder_layers.append(torch.nn.Linear(prev_dim, h_dim))
            encoder_layers.append(torch.nn.ReLU())
            prev_dim = h_dim
        self.encoder = torch.nn.Sequential(*encoder_layers)

        self.fc_mu = torch.nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = torch.nn.Linear(prev_dim, latent_dim)

        decoder_layers = []
        hidden_dims_rev = hidden_dims[::-1]
        hidden_dims_rev.pop(0)  # remove latent_dim
        prev_dim = latent_dim
        for h_dim in hidden_dims_rev:
            decoder_layers.append(torch.nn.Linear(prev_dim, h_dim))
            decoder_layers.append(torch.nn.LeakyReLU())
            prev_dim = h_dim
        decoder_layers.append(torch.nn.Linear(prev_dim, self.flat_dim))
        # decoder_layers.append(torch.nn.Sigmoid())
        self.decoder = torch.nn.Sequential(*decoder_layers) 
    

    def forward(self, input: torch.Tensor, **kwargs):
        mu, logvar = self.encode(input.flatten(1))
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z).unflatten(1, (self.in_channels, self.in_h, self.in_w))
        return decoded, input, mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=self.device)
        return mu + std * eps
    
   
    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        h = self.encoder(input)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        return self.decoder(input)
    
    def loss(self, input: torch.Tensor, output: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, kld_weight: float) -> dict:
        reconstruction_loss = torch.nn.functional.mse_loss(input, output)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
        loss = reconstruction_loss + kld_weight * kld_loss
        return {"loss" : loss, "reconstruction" : reconstruction_loss.detach(), "kld" : kld_loss.detach()}
    
    def sample(self, num_samples: int, return_latent: bool = False) -> torch.Tensor:
        """Sample decoded tensors from the LinearVAE latent space.

        Args:
            num_samples: number of samples to draw from N(0, I)
            return_latent: if True, return a tuple (decoded, z) where z is the sampled latent tensor

        Returns:
            decoded tensors with shape (num_samples, in_channels, in_h, in_w), or
            (decoded, z) if return_latent is True where z has shape (num_samples, latent_dim)
        """
        z = torch.randn(num_samples, self.latent_dim, device=getattr(self, "device", None) or torch.device("cpu"))
        with torch.no_grad():
            decoded_flat = self.decoder(z)
            decoded = decoded_flat.unflatten(1, (self.in_channels, self.in_h, self.in_w))
        if return_latent:
            return decoded, z
        return decoded