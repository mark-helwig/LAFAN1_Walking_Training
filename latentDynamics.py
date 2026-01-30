import torch
import torch.nn as nn

# Residual Latent Dynamics Model
# This model predicts the subsequent latent state given the current latent state.
# The latent dynamics of the VAE are hopefully smooth, so a residual connection is used.
# μ_t+1​=μ_t​+A*μ_t, (log_σt+1)^2​=(log_σt)^2​+B*(log_σt)^2​

class PredictionModel():
    def train_step(self, x_batch, y_batch, encoder, optimizer):
        raise NotImplementedError

class ResidualLatentDynamics(PredictionModel, nn.Module):
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
    
    def train_step(self, x_batch, y_batch, encoder, optimizer):
        # VAE expects NCHW: add channel dim
        x_in = x_batch.unsqueeze(1)  # (B, 1, time=15, features=36)
        y_in = y_batch.unsqueeze(1)  # (B, 1, time=15, features=36)

        optimizer.zero_grad(set_to_none=True)
        encoded_x = encoder.encode(x_in.flatten(1))
        encoded_y = encoder.encode(y_in.flatten(1))

        mu, logvar = self.forward(encoded_x[0], encoded_x[1])
        z_true = encoder.reparameterize(encoded_y[0], encoded_y[1])

        loss = self.loss(z_true, mu, logvar)
        loss.backward()
        optimizer.step()

        return loss.detach()



class LatentMLPDynamics(PredictionModel, nn.Module):
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
        # VAE expects NCHW: add channel dim
        x_in = x_batch.unsqueeze(1)  # (B, 1, time=15, features=36)
        y_in = y_batch.unsqueeze(1)  # (B, 1, time=15, features=36)

        optimizer.zero_grad(set_to_none=True)
        encoded_x = encoder.encode(x_in.flatten(1))
        encoded_y = encoder.encode(y_in.flatten(1))

        z_pred = self.forward(encoded_x[0])
        loss = self.loss(encoded_y[0], z_pred)
        loss.backward()
        optimizer.step()

        return loss.detach()

class LatentTransformer(PredictionModel, nn.Module):
    def __init__(self, latent_dim=128, num_heads=8):
        super().__init__()

        # Split 128 into 16 chunks (tokens) of 8 dims each
        self.num_tokens = 16 
        self.token_dim = latent_dim // self.num_tokens # 8
        
        # Learned position embeddings for the 16 "feature slots"
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, self.token_dim))
        
        # d_model must be divisible by num_heads (8 % 8 == 0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=num_heads,
            dim_feedforward=self.token_dim * 4,
            batch_first=True,
            norm_first=True # Better stability for robotics
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Map the attended features back to the 128-dim output
        self.output_head = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        # x: (Batch, 128)
        b = x.shape[0]
        
        # 1. Reshape to (Batch, 16, 8)
        x = x.view(b, self.num_tokens, self.token_dim)
        
        # 2. Add 'Feature Position' info
        x = x + self.pos_embedding
        
        # 3. Apply Multi-Head Self-Attention across the 16 tokens
        # Each 'token' (feature group) attends to every other 'token'
        x = self.transformer(x)
        
        # 4. Collapse back to (Batch, 128)
        x = x.reshape(b, -1)    
        
        # 5. Final projection to next state
        return self.output_head(x)