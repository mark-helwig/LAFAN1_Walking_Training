import torch
import argparse
import os
from vae import LinearVAE

def convert_pt_to_jit(pt_file_path, output_path=None, input_frames=24, latent_dim=128, batch_size=15):
    """
    Convert a .pt model file to JIT compiled version.
    
    Args:
        pt_file_path: Path to the .pt model file
        output_path: Path to save the JIT compiled model (optional, defaults to same name with .pt -> _jit.pt)
        input_frames: Number of input frames (default: 24)
        latent_dim: Latent dimension (default: 128)
        batch_size: Batch size for example input (default: 15)
    """
    # Determine device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    # Model parameters (should match the training configuration)
    in_channels = 1
    in_size = (input_frames, 36)  # (time, features)
    n_classes = 0
    
    # Initialize model
    print("Initializing model...")
    net = LinearVAE(
        in_size=in_size,
        in_channels=in_channels,
        latent_dim=latent_dim,
        context_dim=n_classes,
        device=DEVICE
    )
    
    # Load state dict
    print(f"Loading model from {pt_file_path}...")
    state_dict = torch.load(pt_file_path, map_location=DEVICE, weights_only=True)
    net.load_state_dict(state_dict)
    
    # Set to eval mode
    net.eval()
    net.to(DEVICE)
    
    # Create example input
    example_input = torch.randn(batch_size, in_channels, input_frames, 36).to(DEVICE)
    
    # Trace the model
    print("Tracing model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(net, example_input, check_trace=False)
    
    # Determine output path
    if output_path is None:
        base_name = os.path.splitext(pt_file_path)[0]
        output_path = f"{base_name}_jit.pt"
    
    # Save JIT model
    print(f"Saving JIT model to {output_path}...")
    torch.jit.save(traced_model, output_path)
    print("Done!")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .pt encoder model to JIT compiled version")
    parser.add_argument("pt_file", type=str, help="Path to the .pt model file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path for JIT model (optional)")
    parser.add_argument("--input-frames", type=int, default=24, help="Number of input frames (default: 24)")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension (default: 128)")
    parser.add_argument("--batch-size", type=int, default=15, help="Batch size for example input (default: 15)")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.pt_file):
        print(f"Error: File not found: {args.pt_file}")
        exit(1)
    
    convert_pt_to_jit(
        args.pt_file,
        args.output,
        args.input_frames,
        args.latent_dim,
        args.batch_size
    )
