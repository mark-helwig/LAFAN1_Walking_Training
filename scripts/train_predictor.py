import torch
import torchvision
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os
from datetime import datetime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(DEVICE)
if DEVICE == "cuda": torch.backends.cudnn.benchmark = True # enables cuDNN auto-tuner
torch.manual_seed(0)

# from LAFAN1_VAE_Experiment import ROOT_DIR
from models.vae import LinearVAE
from models.mlp import LatentMLP
import utils
from datasets.RobotMovementDataset import RobotMovementDataset

def get_walking_filenames_in_folder(folder_path):
    try:
        filenames = os.listdir(folder_path)
        # You can optionally filter for files only, excluding subdirectories
        filenames = [f for f in filenames if f.startswith("walk") and os.path.isfile(os.path.join(folder_path, f))]
        return filenames
    except FileNotFoundError:
        print(f"Error: Folder not found at '{folder_path}'")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def run_training_loop(net, trainloader, encoder, optimizer, epochs, DEVICE, error_graph, run_dir):
    for epoch in range(epochs):
        encoder.to(DEVICE)
        running_loss, running_recons, running_kld = 0, 0, 0
        for i, (x_batch, y_batch) in enumerate(trainloader):
            # DataLoader returns CPU tensors; move per-batch for correct device placement
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            loss = net.train_step(x_batch, y_batch, encoder, optimizer)

            # # VAE expects NCHW: add channel dim
            # x_in = x_batch.unsqueeze(1)  # (B, 1, time=15, features=36)
            # y_in = y_batch.unsqueeze(1)  # (B, 1, time=15, features=36)

            # optimizer.zero_grad(set_to_none=True)
            # encoded_x = encoder.encode(x_in.flatten(1))
            # encoded_y = encoder.encode(y_in.flatten(1))

            # mu, logvar = net.forward(encoded_x[0], encoded_x[1])
            # # z_pred = net.forward(encoded_x[0])
            # z_true = encoder.reparameterize(encoded_y[0], encoded_y[1])

            # loss = net.loss(z_true, mu, logvar)
            # # loss = net.loss(encoded_y[0], z_pred)
            # loss.backward()
            # optimizer.step()

            running_loss += loss

        print(
            "[{epoch}, {batch}%] loss: {loss:.4f}".format(
                epoch=epoch+1,
                batch=100,
                loss=loss.item(),
            )
        )   
        n_batches = i + 1 if 'i' in locals() else 1
        print(
            "[{epoch}, train] loss: {loss:.4f}".format(
                epoch=epoch+1,
                loss=(running_loss / n_batches).item(),
            )
        )
        error_graph.append((running_loss / n_batches).item())

        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_filename = f"{run_dir}/checkpoint_epoch_{epoch + 1}.pt"
            torch.save(net.state_dict(), checkpoint_filename)
            print(f"  Checkpoint saved: {checkpoint_filename}")
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train latent-space predictor (MLP) model")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--history-len", type=int, default=24, help="Number of history frames")
    parser.add_argument("--pred-len", type=int, default=8, help="Number of prediction frames")
    parser.add_argument("--latent-dim", type=int, default=128, help="Dimension of latent space")
    parser.add_argument("--batch-size", type=int, default=15, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--normalize", action="store_true", help="Normalize dataset")
    parser.add_argument("--device", type=str, default=DEVICE, choices=["cuda", "cpu"], help=f"Device to use (default: {DEVICE})")
    parser.add_argument("--data-folder", type=str, default="LAFAN1_Retargeting_Dataset/g1/", help="Path to data folder")
    parser.add_argument("--encoder-path", type=str, default="generated_models/encoder/model_fallback_1000_epochs_20260202_205355.pt", help="Path to trained encoder weights")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension for predictor MLP")
    args = parser.parse_args()

    DEVICE = args.device

    filenames = get_walking_filenames_in_folder(args.data_folder)
    if not filenames:
        print(f"No walking files found in {args.data_folder}")
        raise SystemExit(1)
    
    encoder_filepath = args.encoder_path

    normalize = args.normalize
    reconstruct = False

    motions = filenames
    classes = []
    n_classes = len(classes)
    in_channels = 1

    input_frames = args.history_len
    pred_length = args.pred_len
    in_size = (input_frames, 36)
    latent_dim = args.latent_dim
    epochs = args.epochs
    batch_size = args.batch_size

    error_graph = []

    # Create run-specific folder before training
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"generated_models/predictor/{run_timestamp}_mlp_h{input_frames}_p{pred_length}_d{latent_dim}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving checkpoints to {run_dir}\n")


    dataset = RobotMovementDataset(filenames=motions, input_len=input_frames, output_len=pred_length, device=DEVICE, normalize=normalize, reconstruct=reconstruct)
    # make dataset
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=2, pin_memory=True, drop_last=True)   
    
    kld_weight_train = 1/len(trainloader)
    print(f"Dataset batches: {len(trainloader)}")
    
    # define network
    encoder = LinearVAE(in_size=in_size, in_channels=in_channels, latent_dim=latent_dim, context_dim=n_classes, device=DEVICE)
    state_dict = torch.load(encoder_filepath, weights_only=True)
    encoder.load_state_dict(state_dict)
    encoder.to(DEVICE)
    encoder.eval()
    net = LatentMLP(latent_dim=latent_dim, hidden_dim=args.hidden_dim, device=DEVICE)
    net.to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    print(net)
    print()

    # run training loop
    run_training_loop(net, trainloader, encoder, optimizer, epochs, DEVICE, error_graph, run_dir)

    net.eval()
    example_input = torch.randn(batch_size, latent_dim, device=DEVICE)
    traced_model = torch.jit.trace(net, example_input, check_trace=False)

    fallback_filepath = f"{run_dir}/model.pt"
    error_filepath = f"{run_dir}/error.csv"
    model_filepath = f"{run_dir}/model_jit.pt"
    
    torch.save(net.state_dict(), fallback_filepath)
    try:
        torch.jit.save(traced_model, model_filepath)
    except Exception as e:
        print(f"Error saving traced model: {e}")
    error_graph = pd.DataFrame(error_graph, columns=['epoch_loss'])
    error_graph.to_csv(error_filepath, index=False)
    print(f"Error graph saved to {error_filepath}")
