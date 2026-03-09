import os
import sys
import argparse
import time
from contextlib import nullcontext
from datetime import datetime
import pandas as pd
import torch

if __name__ == "__main__":
    # Set up paths first
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from models.transformer import TransformerPredictor, TransformerPredictorAutoregressive
    from datasets.RobotMovementDataset import RobotMovementDataset
    
    # Set device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    
    # Argument parser
    parser = argparse.ArgumentParser(description="Train a Transformer model for robot motion prediction")
        
    # Optional arguments
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--history-len", type=int, default=24, help="Number of history frames (default: 24)")
    parser.add_argument("--pred-len", type=int, default=8, help="Number of frames to predict (default: 8)")
    parser.add_argument("--latent-dim", type=int, default=128, help="Dimension of latent space (default: 128)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training (default: 64)")
    parser.add_argument("--num-encoder-layers", type=int, default=1, help="Number of transformer encoder layers (default: 2)")
    parser.add_argument("--num-decoder-layers", type=int, default=1, help="Number of transformer decoder layers (default: 2)")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads (default: 4)")
    parser.add_argument("--dim-feedforward", type=int, default=256, help="Dimension of feedforward network (default: 256)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--model-type", type=str, default="autoregressive", choices=["encoder-decoder", "autoregressive"], help="Model type: 'encoder-decoder' or 'autoregressive' (default: encoder-decoder)")
    parser.add_argument("--normalize", action="store_true", help="Normalize dataset (default: False)")
    parser.add_argument("--teacher-forcing", action="store_true", help="Enable teacher forcing for autoregressive model (default: disabled)")
    parser.add_argument("--device", type=str, default=DEVICE, choices=["cuda", "cpu"], help=f"Device to use (default: {DEVICE})")
    parser.add_argument("--data-folder", type=str, default="LAFAN1_Retargeting_Dataset/g1/", help="Path to data folder (default: LAFAN1_Retargeting_Dataset/g1/)")
    parser.add_argument("--subject", type=int, default=None, help="Subject to use from dataset (default: all subjects)")
    parser.add_argument("--num-workers", type=int, default=16, help="DataLoader worker processes (default: 16)")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch factor per worker (default: 4)")
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True, help="Keep DataLoader workers alive across epochs (default: enabled)")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Enable automatic mixed precision on CUDA (default: enabled)")
    parser.add_argument("--amp-dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"], help="AMP dtype to use on CUDA (default: bfloat16)")
    
    args = parser.parse_args()
    
    # Override DEVICE if specified
    DEVICE = args.device
    amp_enabled = args.amp and DEVICE == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

    if args.grad_accum_steps < 1:
        print("Error: --grad-accum-steps must be >= 1")
        sys.exit(1)
    
    # Get filenames
    def get_walking_filenames_in_folder(folder_path, subject):
        try:
            filenames = os.listdir(folder_path)
            filenames = [f for f in filenames if f.startswith("walk") and os.path.isfile(os.path.join(folder_path, f))]
            if subject is not None:
                filenames = [f for f in filenames if f"subject{subject}" in f]
            return filenames
        except FileNotFoundError:
            print(f"Error: Folder not found at '{folder_path}'")
            return []
        except Exception as e:
            print(f"An error occurred: {e}")
            return []
    
    filenames = get_walking_filenames_in_folder(args.data_folder, args.subject)
    if not filenames:
        print(f"No walking files found in {args.data_folder}")
        sys.exit(1)
    
    print(f"Found {len(filenames)} walking files")
    
    # Create dataset
    dataset = RobotMovementDataset(
        filenames=filenames,
        input_len=args.history_len,
        output_len=args.pred_len,
        device="cpu",
        normalize=args.normalize,
        reconstruct=False,
    )
    
    # Create dataloader
    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": (DEVICE == "cuda"),
        "drop_last": True,
    }
    if args.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
        dataloader_kwargs["persistent_workers"] = args.persistent_workers
    trainloader = torch.utils.data.DataLoader(**dataloader_kwargs)
    
    print(f"Dataset batches: {len(trainloader)}")
    print(f"Training with {len(filenames)} files, {len(dataset)} samples")
    print(
        f"Loader config: workers={args.num_workers}, prefetch={args.prefetch_factor if args.num_workers > 0 else 'n/a'}, "
        f"persistent={args.persistent_workers if args.num_workers > 0 else 'n/a'}"
    )
    
    # Create model
    if args.model_type == "encoder-decoder":
        model = TransformerPredictor(
            history_len=args.history_len,
            pred_len=args.pred_len,
            joint_dim=36,
            latent_dim=args.latent_dim,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            device=DEVICE,
        )
        print("Using TransformerPredictor (encoder-decoder)")
    else:
        model = TransformerPredictorAutoregressive(
            history_len=args.history_len,
            pred_len=args.pred_len,
            joint_dim=36,
            latent_dim=args.latent_dim,
            num_layers=args.num_encoder_layers + args.num_decoder_layers,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            use_teacher_forcing= args.teacher_forcing,
            device=DEVICE,
        )
        teacher_forcing_status = "enabled" if args.teacher_forcing else "disabled"
        print(f"Using TransformerPredictorAutoregressive (teacher forcing: {teacher_forcing_status})")
    
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"AMP enabled: {amp_enabled} (dtype={args.amp_dtype if amp_enabled else 'n/a'})")
    print(f"Gradient accumulation steps: {args.grad_accum_steps}")
    print()
    
    # Create run-specific folder before training
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = (
        f"generated_models/transformer/{timestamp}_"
        f"{args.model_type}_h{args.history_len}_p{args.pred_len}_d{args.latent_dim}"
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving checkpoints to {run_dir}\n")
    
    # Training loop
    error_graph = []
    print("Starting training...")
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        batch_start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        
        for i, (x_batch, y_batch) in enumerate(trainloader):
            # Move batches to device
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)
            
            # For transformer, only use the first pred_len frames from the target
            # (dataset returns overlapping windows needed for VAE/MLP)
            y_batch = y_batch[:, :args.pred_len, :]
            
            amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_enabled else nullcontext()
            with amp_context:
                if args.model_type == "autoregressive" and args.teacher_forcing:
                    predictions = model.forward_teacher_forcing(x_batch, y_batch)
                else:
                    predictions = model(x_batch)

                loss_dict = model.loss(predictions, y_batch)
                loss = loss_dict["loss"]

            loss_for_backward = loss / args.grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            if (i + 1) % args.grad_accum_steps == 0 or (i + 1) == len(trainloader):
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.detach().item()
        
        n_batches = i + 1 if 'i' in locals() else 1
        avg_loss = running_loss / n_batches
        error_graph.append(avg_loss)
        
        epoch_time = time.time() - batch_start_time
        gpu_mem = ""
        if DEVICE == "cuda":
            gpu_mem = f" | GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB"
        print(f"[Epoch {epoch + 1}/{args.epochs}] loss: {avg_loss:.6f} | time: {epoch_time:.1f}s{gpu_mem}")
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_filename = f"{run_dir}/checkpoint_epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_filename)
            print(f"  Checkpoint saved: {checkpoint_filename}")
        
        # Clear GPU cache to prevent memory buildup
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    
    print("Training complete.")
    
    # Save model and metrics
    model.eval()
    
    model_filename = f"{run_dir}/model.pt"
    error_filename = f"{run_dir}/error.csv"
    
    # Save state dict
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")
    
    # Try to save JIT traced model
    try:
        example_input = torch.randn(args.batch_size, args.history_len, 36, device=DEVICE)
        traced_model = torch.jit.trace(model, example_input, check_trace=False)
        jit_filename = model_filename.replace(".pt", "_jit.pt")
        torch.jit.save(traced_model, jit_filename)
        print(f"JIT model saved to {jit_filename}")
    except Exception as e:
        print(f"Warning: Could not save JIT model: {e}")
    
    # Save error graph
    error_df = pd.DataFrame(error_graph, columns=["epoch_loss"])
    error_df.to_csv(error_filename, index=False)
    print(f"Error graph saved to {error_filename}")
    
    # Print training summary
    print("\n" + "="*50)
    print("Training Summary:")
    print(f"  Model type: {args.model_type}")
    print(f"  History length: {args.history_len}")
    print(f"  Prediction length: {args.pred_len}")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Final loss: {error_graph[-1]:.6f}")
    print("="*50)