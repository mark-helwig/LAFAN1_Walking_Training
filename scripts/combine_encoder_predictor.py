import torch
import argparse
import os
from models.vae import LinearVAE
from models.mlp import LatentMLPDynamics, ResidualLatentDynamics


class EncoderPredictorCombined(torch.nn.Module):
    """
    Combined encoder and predictor model.
    Takes raw motion data and returns both latent representations and predictions.
    """
    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, channels, time, features)
        
        Returns:
            latents: (batch, latent_dim) - latent representations
            preds: (batch, latent_dim) - predicted next latent state
        """
        # Encode the input
        x_flat = x.flatten(1)  # (batch, channels * time * features)
        mu, logvar = self.encoder.encode(x_flat)
        latents = mu
        
        # Predict next latent state
        preds = self.predictor(latents)
        
        return latents, preds


def combine_encoder_predictor(encoder_pt_path, predictor_pt_path, output_path=None, 
                               input_frames=24, latent_dim=128, batch_size=15, 
                               predictor_type="mlp", hidden_dim=256):
    """
    Combine encoder and predictor models into a single JIT-compiled model.
    
    Args:
        encoder_pt_path: Path to encoder .pt file
        predictor_pt_path: Path to predictor .pt file
        output_path: Path to save the combined JIT model (optional)
        input_frames: Number of input frames (default: 24)
        latent_dim: Latent dimension (default: 128)
        batch_size: Batch size for example input (default: 15)
        predictor_type: Type of predictor ('mlp' or 'residual')
        hidden_dim: Hidden dimension for MLP predictor (default: 256)
    """
    # Determine device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    # Model parameters
    in_channels = 1
    in_size = (input_frames, 36)  # (time, features)
    n_classes = 0
    
    # Load encoder
    print(f"Loading encoder from {encoder_pt_path}...")
    encoder = LinearVAE(
        in_size=in_size,
        in_channels=in_channels,
        latent_dim=latent_dim,
        context_dim=n_classes,
        device=DEVICE
    )
    encoder_state = torch.load(encoder_pt_path, map_location=DEVICE, weights_only=True)
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    encoder.to(DEVICE)
    
    # Load predictor
    print(f"Loading predictor from {predictor_pt_path}...")
    if predictor_type.lower() == "mlp":
        predictor = LatentMLPDynamics(latent_dim=latent_dim, hidden_dim=hidden_dim, device=DEVICE)
    elif predictor_type.lower() == "residual":
        predictor = ResidualLatentDynamics(latent_dim=latent_dim, device=DEVICE)
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")
    
    predictor_state = torch.load(predictor_pt_path, map_location=DEVICE, weights_only=True)
    predictor.load_state_dict(predictor_state)
    predictor.eval()
    predictor.to(DEVICE)
    
    # Create combined model
    print("Creating combined model...")
    combined = EncoderPredictorCombined(encoder, predictor)
    combined.eval()
    combined.to(DEVICE)
    
    # Create example input: (batch, channels, time, features)
    example_input = torch.randn(batch_size, in_channels, input_frames, 36, device=DEVICE)
    
    # Trace the combined model
    print("Tracing combined model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(combined, example_input, check_trace=False)
    
    # Determine output path
    if output_path is None:
        output_path = "models/combined/encoder_predictor_combined_jit.pt"
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save JIT model
    print(f"Saving combined JIT model to {output_path}...")
    torch.jit.save(traced_model, output_path)
    print("Done!")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine encoder and predictor models into single JIT model")
    parser.add_argument("encoder_pt", type=str, help="Path to encoder .pt file")
    parser.add_argument("predictor_pt", type=str, help="Path to predictor .pt file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path for combined JIT model")
    parser.add_argument("--input-frames", type=int, default=24, help="Number of input frames (default: 24)")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension (default: 128)")
    parser.add_argument("--batch-size", type=int, default=15, help="Batch size for tracing (default: 15)")
    parser.add_argument("--predictor-type", type=str, default="mlp", choices=["mlp", "residual"],
                        help="Type of predictor (default: mlp)")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension for MLP predictor (default: 256)")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.encoder_pt):
        print(f"Error: Encoder file not found: {args.encoder_pt}")
        exit(1)
    if not os.path.exists(args.predictor_pt):
        print(f"Error: Predictor file not found: {args.predictor_pt}")
        exit(1)
    
    combine_encoder_predictor(
        args.encoder_pt,
        args.predictor_pt,
        args.output,
        args.input_frames,
        args.latent_dim,
        args.batch_size,
        args.predictor_type,
        args.hidden_dim
    )
