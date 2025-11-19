import torch
import torch.nn as nn

# Residual Latent Dynamics Model
# This model predicts the subsequent latent state given the current latent state.
# The latent dynamics of the VAE are hopefully smooth, so a residual connection is used.
# μ_t+1​=μ_t​+A*μ_t, (log_σt+1)^2​=(log_σt)^2​+B*(log_σt)^2​

class ResidualLatentDynamics(nn.Module):
    def __init__(self, latent_dim=128, device="cpu", **kwargs):
        super().__init__()
        self.mu_linear = nn.Linear(latent_dim, latent_dim)
        self.logvar_linear = nn.Linear(latent_dim, latent_dim)

    def forward(self, mu_t, logvar_t):
        delta_mu     = self.mu_linear(mu_t)
        delta_logvar = self.logvar_linear(logvar_t)
        return mu_t + delta_mu, logvar_t + delta_logvar

    # Gaussian negative log-likelihood loss
    def loss(self, true_z, pred_mu, pred_logvar):
        return 0.5 * torch.mean(
            pred_logvar + (true_z - pred_mu)**2 / torch.exp(pred_logvar)
        )

class LatentMLPDynamics(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256, device="cpu", **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t):
        delta = self.net(z_t)
        return z_t + delta  # residual update
    
    def loss(self, z_true, z_pred):
        return nn.MSELoss()(z_pred, z_true)