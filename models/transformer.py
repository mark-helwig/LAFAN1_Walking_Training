import torch
import torch.nn as nn
from typing import Optional, Callable, List


class InputProjection(nn.Module):
    """1-layer MLP to project from joint space (36) to latent space (latent_dim)."""
    def __init__(self, input_dim: int = 36, latent_dim: int = 128, **kwargs):
        super().__init__()
        self.linear = nn.Linear(input_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 36) or (batch, 36)
        
        Returns:
            (batch, seq_len, latent_dim) or (batch, latent_dim)
        """
        return self.linear(x)


class OutputProjection(nn.Module):
    """Project from latent space back to joint space (36) for loss computation."""
    def __init__(self, latent_dim: int = 128, output_dim: int = 36, **kwargs):
        super().__init__()
        self.linear = nn.Linear(latent_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, latent_dim) or (batch, latent_dim)
        
        Returns:
            (batch, seq_len, 36) or (batch, 36)
        """
        return self.linear(x)


class TransformerPredictor(nn.Module):
    """
    Combined Transformer encoder-predictor model.
    
    Takes history frames in joint space, projects to latent space, applies transformer
    layers to predict future latent representations, and decodes back to joint space.
    
    Args:
        history_len: number of input frames
        pred_len: number of frames to predict
        joint_dim: dimension of joint space (default 36)
        latent_dim: dimension of latent space (default 128)
        num_encoder_layers: number of transformer encoder layers
        num_decoder_layers: number of transformer decoder layers
        nhead: number of attention heads
        dim_feedforward: dimension of feedforward network in transformer
        dropout: dropout rate
        activation: activation function in transformer
        device: device to use
    """
    
    def __init__(
        self,
        history_len: int = 24,
        pred_len: int = 8,
        joint_dim: int = 36,
        latent_dim: int = 128,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "relu",
        device: str = "cpu",
        **kwargs
    ):
        super().__init__()
        
        self.history_len = history_len
        self.pred_len = pred_len
        self.joint_dim = joint_dim
        self.latent_dim = latent_dim
        self.device = device
        
        # Input projection: 36 -> latent_dim (1-layer MLP)
        self.input_projection = InputProjection(input_dim=joint_dim, latent_dim=latent_dim)
        
        # Positional encoding
        self.register_buffer(
            "pos_encoding",
            self._create_positional_encoding(history_len + pred_len, latent_dim)
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )
        
        # Learnable decoder input tokens
        self.decoder_tokens = nn.Parameter(torch.randn(1, pred_len, latent_dim))
        
        # Output projection: latent_dim -> 36
        self.output_projection = OutputProjection(latent_dim=latent_dim, output_dim=joint_dim)
        
        self.to(device)
    
    def _create_positional_encoding(
        self, 
        seq_len: int, 
        d_model: int
    ) -> torch.Tensor:
        """Create positional encoding for sequence."""
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div_term)
        
        return pe.unsqueeze(0)  # (1, seq_len, d_model)
    
    def forward(
        self, 
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, history_len, 36) - history frames in joint space
            src_mask: optional mask for encoder
            tgt_mask: optional mask for decoder (causal mask)
        
        Returns:
            (batch, pred_len, 36) - predicted future frames in joint space
        """
        batch_size = x.size(0)
        
        # Project input to latent space: (batch, history_len, 36) -> (batch, history_len, latent_dim)
        latent_history = self.input_projection(x)
        
        # Add positional encoding
        latent_history = latent_history + self.pos_encoding[:, :self.history_len, :]
        
        # Encode history through transformer encoder
        encoded = self.transformer_encoder(latent_history, src_key_padding_mask=src_mask)
        
        # Prepare decoder input: expand learnable tokens to batch size
        decoder_input = self.decoder_tokens.expand(batch_size, -1, -1)
        
        # Add positional encoding to decoder input (offset by history_len)
        decoder_input = decoder_input + self.pos_encoding[:, self.history_len:self.history_len + self.pred_len, :]
        
        # Decode: generate latent predictions for future frames
        latent_predictions = self.transformer_decoder(
            decoder_input,
            encoded,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_mask,
        )
        
        # Project latent predictions back to joint space
        predictions = self.output_projection(latent_predictions)
        
        return predictions
    
    def loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> dict:
        """
        Compute MSE loss between predictions and targets.
        
        Args:
            predictions: (batch, pred_len, 36) - model predictions in joint space
            targets: (batch, pred_len, 36) - ground truth future frames in joint space
        
        Returns:
            dict containing loss and detached loss value
        """
        mse_loss = nn.MSELoss()(predictions, targets)
        return {
            "loss": mse_loss,
            "mse": mse_loss.detach(),
        }
    
    def train_step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        """
        Single training step.
        
        Args:
            x_batch: (batch, history_len, 36) - input history
            y_batch: (batch, pred_len, 36) - target future frames
            optimizer: optimizer to use
        
        Returns:
            loss value (detached)
        """
        optimizer.zero_grad(set_to_none=True)
        
        predictions = self.forward(x_batch)
        loss_dict = self.loss(predictions, y_batch)
        
        loss_dict["loss"].backward()
        optimizer.step()
        
        return loss_dict["loss"].detach()


class TransformerPredictorAutoregressive(nn.Module):
    """
    Alternative autoregressive Transformer predictor.
    
    Uses encoder-only architecture with autoregressive generation.
    At prediction time, generates one frame at a time, feeding previous predictions
    as input for the next frame.
    """
    
    def __init__(
        self,
        history_len: int = 24,
        pred_len: int = 8,
        joint_dim: int = 36,
        latent_dim: int = 128,
        num_layers: int = 4,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "relu",
        device: str = "cpu",
        **kwargs
    ):
        super().__init__()
        
        self.history_len = history_len
        self.pred_len = pred_len
        self.joint_dim = joint_dim
        self.latent_dim = latent_dim
        self.device = device
        
        # Input projection: 36 -> latent_dim (1-layer MLP)
        self.input_projection = InputProjection(input_dim=joint_dim, latent_dim=latent_dim)
        
        # Positional encoding
        self.register_buffer(
            "pos_encoding",
            self._create_positional_encoding(history_len + pred_len, latent_dim)
        )
        
        # Transformer encoder with causal mask for autoregressive generation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Output projection: latent_dim -> 36
        self.output_projection = OutputProjection(latent_dim=latent_dim, output_dim=joint_dim)
        
        self.to(device)
    
    def _create_positional_encoding(
        self, 
        seq_len: int, 
        d_model: int
    ) -> torch.Tensor:
        """Create positional encoding for sequence."""
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div_term)
        
        return pe.unsqueeze(0)  # (1, seq_len, d_model)
    
    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, history_len, 36) - history frames in joint space
        
        Returns:
            (batch, pred_len, 36) - predicted future frames in joint space
        """
        batch_size = x.size(0)
        
        # Project input to latent space
        latent_history = self.input_projection(x)
        
        # Add positional encoding
        latent_history = latent_history + self.pos_encoding[:, :self.history_len, :]
        
        # Process history through transformer encoder
        encoded = self.transformer_encoder(latent_history)
        
        # Autoregressive generation
        predictions = []
        current_input = latent_history
        
        for step in range(self.pred_len):
            # Create causal mask
            seq_len = current_input.size(1)
            causal_mask = self._create_causal_mask(seq_len)
            
            # Process through encoder
            hidden = self.transformer_encoder(current_input, src_key_padding_mask=causal_mask)
            
            # Get last frame's latent representation and project to joint space
            last_latent = hidden[:, -1, :]
            next_pred = self.output_projection(last_latent)  # (batch, 36)
            predictions.append(next_pred)
            
            # Project next prediction to latent space and add positional encoding
            next_latent = self.input_projection(next_pred)
            pos_idx = self.history_len + step + 1
            if pos_idx < self.pos_encoding.size(1):
                next_latent = next_latent + self.pos_encoding[:, pos_idx, :]
            
            # Append to input for next step
            current_input = torch.cat([current_input, next_latent.unsqueeze(1)], dim=1)
        
        # Stack predictions: list of (batch, 36) -> (batch, pred_len, 36)
        predictions = torch.stack(predictions, dim=1)
        
        return predictions
    
    def loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> dict:
        """
        Compute MSE loss between predictions and targets.
        
        Args:
            predictions: (batch, pred_len, 36) - model predictions in joint space
            targets: (batch, pred_len, 36) - ground truth future frames in joint space
        
        Returns:
            dict containing loss and detached loss value
        """
        mse_loss = nn.MSELoss()(predictions, targets)
        return {
            "loss": mse_loss,
            "mse": mse_loss.detach(),
        }
    
    def train_step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        """
        Single training step.
        
        Args:
            x_batch: (batch, history_len, 36) - input history
            y_batch: (batch, pred_len, 36) - target future frames
            optimizer: optimizer to use
        
        Returns:
            loss value (detached)
        """
        optimizer.zero_grad(set_to_none=True)
        
        predictions = self.forward(x_batch)
        loss_dict = self.loss(predictions, y_batch)
        
        loss_dict["loss"].backward()
        optimizer.step()
        
        return loss_dict["loss"].detach()
