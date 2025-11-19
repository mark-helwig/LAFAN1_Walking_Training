from typing import Any, Callable, List

import torch
import pandas as pd
import utils

class VAE(torch.nn.Module):
    def __init__(self,
                 in_size: int | tuple[int, int],
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 activation_func: Callable = torch.nn.GELU,
                 **kwargs
                 ) -> None:
        super(VAE, self).__init__()

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
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]
        self.device = kwargs["device"]
        
        # encoder
        encoder_layers = []
        for h_dim in hidden_dims:
            encoder_layers.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=1, padding=1, bias=False, device=self.device),
                    torch.nn.BatchNorm2d(h_dim, device=self.device),
                    self.activation_func()  
                )
            )
            in_channels = h_dim
        encoder_layers.append(torch.nn.Flatten())

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.encoder.apply(self.init_weights)

        # with kernel=3, stride=1, padding=1 spatial dims remain the same across conv layers
        self.enc_h = utils.get_encoder_size(self.in_h, hidden_dims, kernel=3, stride=1, padding=1)
        self.enc_w = utils.get_encoder_size(self.in_w, hidden_dims, kernel=3, stride=1, padding=1)

        self.mu = torch.nn.Linear(hidden_dims[-1] * self.enc_h * self.enc_w, latent_dim, device=self.device)
        self.mu.apply(self.init_weights)

        self.logvar = torch.nn.Linear(hidden_dims[-1] * self.enc_h * self.enc_w, latent_dim, device=self.device)
        self.logvar.apply(self.init_weights)

        # decoder
        hidden_dims.reverse()
        decoder_layers = [
            torch.nn.Linear(latent_dim, hidden_dims[0] * self.enc_h * self.enc_w, device=self.device),
            torch.nn.Unflatten(-1, (hidden_dims[0], self.enc_h, self.enc_w))
        ]
        for i in range(len(hidden_dims)-1):
            decoder_layers.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, stride=1, padding=1, bias=False, device=self.device),
                    torch.nn.BatchNorm2d(hidden_dims[i+1], device=self.device),
                    self.activation_func()
                )
            )
        decoder_layers.append(
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=1, padding=1, device=self.device),
                torch.nn.BatchNorm2d(hidden_dims[-1], device=self.device),
                self.activation_func(),
                torch.nn.Conv2d(hidden_dims[-1], self.in_channels, kernel_size=3, stride=1, padding=1, device=self.device),
                torch.nn.Tanh()
            )
        )

        self.decoder = torch.nn.Sequential(*decoder_layers)
        self.decoder.apply(self.init_weights)

    def init_weights(self, l: Any) -> None:
        if isinstance(l, torch.nn.Linear) or isinstance(l, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(l.weight)
            if isinstance(l, torch.nn.Linear): l.bias.data.fill_(0.01)

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        latent = self.encoder(input)
        mu = self.mu(latent)
        logvar = self.logvar(latent)
        return [mu, logvar]

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        return self.decoder(input)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=self.device)
        return mu + std * eps

    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return [self.decode(z), input, mu, logvar]
    
    def sample(self, num_samples: int, **kwargs) -> torch.Tensor:
        """Sample decoded tensors from the latent space.

        Returns shape: (num_samples, in_channels, in_h, in_w)
        """
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        with torch.no_grad():
            x_gen = self.decoder(z)
        return x_gen

    def generate_sequence(self, seq_len: int, filepath: str, dataset) -> None:
        """
        Generate a naive motion sequence by sampling independent frames and
        taking the last time step across the height dimension (temporal axis).

        NOTE: This is a simplistic sampler for VAEs trained on (time, features)
        patches. For coherent sequences, consider an autoregressive model.
        """
        frames = self.sample(seq_len)  # (seq_len, C=1, H=time, W=features)
        samples_np = frames.cpu().numpy().reshape(-1, dataset.df.shape[1])
        # Take the last time step (H-1) to get a single 36-dim vector per sample
        samples_denorm = dataset.denormalize(samples_np)

        # last_step = frames[:, 0, -1, :]  # (seq_len, in_w)
        df_generated = pd.DataFrame(samples_denorm)         
        df_generated.to_csv(filepath, index=False)  
        print(f"Generated sequence saved to {filepath}")

    def generate(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(input, **kwargs)[0]

    def loss(self, input: torch.Tensor, output: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, kld_weight: float) -> dict:
        reconstruction_loss = torch.nn.functional.mse_loss(input, output)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
        loss = reconstruction_loss + kld_weight * kld_loss
        return {"loss" : loss, "reconstruction" : reconstruction_loss.detach(), "kld" : kld_loss.detach()}
    


class PhaseVAE(torch.nn.Module):
    def __init__(self, input_dim=36, seq_in=20, seq_out=2,
                 latent_dim=16, hidden_channels=32, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.latent_dim = latent_dim
        self.device = kwargs["device"]

        # ---------- Encoder ----------
        # Input: (B,1,seq_in,input_dim)
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, hidden_channels, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_channels, hidden_channels * 2, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 3, stride=2, padding=1),
            torch.nn.ReLU()
        )
        self.flat_dim = hidden_channels * 4 * (seq_in // 4) * (input_dim // 4)
        self.fc_mu = torch.nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = torch.nn.Linear(self.flat_dim, latent_dim)

        # ---------- Decoder ----------
        # phase adds 2 extra channels (sinϕ, cosϕ)
        self.fc_decode = torch.nn.Linear(latent_dim + 2, hidden_channels * 4 * (seq_out // 2) * (input_dim // 4))
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_channels, 1, 3, padding=1)  # final projection
)

        self.to(self.device)

    # ---- Reparameterization ----
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device = self.device)
        return mu + eps * std

    # ---- Forward ----
    def forward(self, x, phase, z_override=None):
        """
        x: (B, seq_in, input_dim)
        phase: (B,) in radians
        """
        

        if x.dim() == 3:
            x = x.unsqueeze(1)  
        x = x.to(self.device)        # (B,1,seq_in,input_dim)

        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        if z_override is not None:
            z = z_override
        else:
            z = self.reparameterize(mu, logvar)

        # append sinϕ, cosϕ to latent
        phase_feat = torch.stack([torch.sin(phase), torch.cos(phase)], dim=1)
        z_phase = torch.cat([z, phase_feat], dim=1)

        # decode
        h_dec = self.fc_decode(z_phase)
        h_dec = h_dec.view(x.size(0), -1, self.seq_out // 2, self.input_dim // 4)
        out = self.decoder(h_dec)
        out = torch.nn.functional.interpolate(out, size=(self.seq_out, self.input_dim), mode="bilinear", align_corners=False)
        return out.squeeze(1), mu, logvar  # (B, seq_out, input_dim)
    
    def loss(self, input: torch.Tensor, output: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, kld_weight: float, smooth_weight: float) -> dict:
        recon_loss = torch.nn.functional.mse_loss(output, input)

    # ---- KL divergence ----
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # ---- Temporal smoothness (optional) ----
        if output.size(1) > 1:  # only if more than 1 frame predicted
            vel_pred = output[:, 1:, :] - output[:, :-1, :]
            vel_true = input[:, 1:, :] - input[:, :-1, :]
            smooth_loss = torch.nn.functional.mse_loss(vel_pred, vel_true)
        else:
            smooth_loss = torch.tensor(0.0, device=output.device)

        # ---- Combine ----
        total_loss = recon_loss + kld_weight * kl_loss + smooth_weight * smooth_loss

        return {"loss" : total_loss, "reconstruction" : recon_loss.detach(), "kld" : kl_loss.detach(), "smooth" : smooth_loss.detach()}


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
        reconstruction_loss = torch.functional.mse_loss(input, output)
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