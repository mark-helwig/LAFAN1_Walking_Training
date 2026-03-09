import torch
import torch.nn as nn


class PredictionModel():
    def train_step(self, x_batch, y_batch, encoder, optimizer):
        raise NotImplementedError

# Residual Latent Dynamics Model
# This model predicts the subsequent latent state given the current latent state.
# The latent dynamics of the VAE are hopefully smooth, so a residual connection is used.
# μ_t+1​=μ_t​+A*μ_t, (log_σt+1)^2​=(log_σt)^2​+B*(log_σt)^2​

class LatentMLP(PredictionModel, nn.Module):
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
    
    def train_step(self, x_batch, y_batch, encoder, optimizer):
        # Build a target window with the same temporal length as x_batch.
        # For non-overlap datasets, y_batch is pure future of length K.
        # We construct the shifted window: [x[:, K:], y[:, :K]].
        hist_len = x_batch.size(1)
        pred_len = y_batch.size(1)

        if pred_len > hist_len:
            raise ValueError(
                f"Prediction length ({pred_len}) cannot exceed history length ({hist_len}) for latent MLP training."
            )

        y_shifted = torch.cat((x_batch[:, pred_len:, :], y_batch[:, :pred_len, :]), dim=1)

        # VAE expects NCHW
        x_in = x_batch.unsqueeze(1)
        y_in = y_shifted.unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        encoded_x = encoder.encode(x_in.flatten(1))
        encoded_y = encoder.encode(y_in.flatten(1))

        z_pred = self.forward(encoded_x[0])
        loss = self.loss(encoded_y[0], z_pred)
        loss.backward()
        optimizer.step()

        return loss.detach()