import os
from datetime import datetime
import argparse

import torch
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
torch.manual_seed(0)

# from LAFAN1_VAE_Experiment import ROOT_DIR
from models.vae import LinearVAE
import utils
from datasets.RobotMovementDataset import RobotMovementDataset


def get_filenames_in_folder(folder_path):
    """
    Retrieves a list of filenames within a specified folder.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        list: A list containing the names of files in the folder.
              Returns an empty list if the folder does not exist or is empty.
    """
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train encoder (VAE) model")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--history-len", type=int, default=24, help="Number of input frames")
    parser.add_argument("--latent-dim", type=int, default=128, help="Dimension of latent space")
    parser.add_argument("--batch-size", type=int, default=15, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--normalize", action="store_true", help="Normalize dataset")
    parser.add_argument("--device", type=str, default=DEVICE, choices=["cuda", "cpu"], help=f"Device to use (default: {DEVICE})")
    parser.add_argument("--data-folder", type=str, default="LAFAN1_Retargeting_Dataset/g1/", help="Path to data folder")
    args = parser.parse_args()

    DEVICE = args.device

    filenames = get_filenames_in_folder(args.data_folder)
    if not filenames:
        print(f"No walking files found in {args.data_folder}")
        raise SystemExit(1)

    normalize = args.normalize
    reconstruct = True

    motions = filenames
    classes = []
    n_classes = len(classes)
    in_channels = 1

    input_frames = args.history_len
    in_size = (input_frames, 36)
    latent_dim = args.latent_dim
    epochs = args.epochs
    batch_size = args.batch_size

    error_graph = []

    # Create run-specific folder before training
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"generated_models/encoder/{run_timestamp}_vae_h{input_frames}_d{latent_dim}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving checkpoints to {run_dir}\n")


    dataset = RobotMovementDataset(
        filenames=motions,
        input_len=input_frames,
        output_len=input_frames,
        device="cpu",
        normalize=normalize,
        reconstruct=reconstruct,
    )
    # make dataset
    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
        drop_last=True,
    )
    
    kld_weight_train = 1/len(trainloader)
    print(f"Dataset batches: {len(trainloader)}")
    # kld_weight_train = 1/(2*len(trainloader))
    print("kld_weight_train:", kld_weight_train)
    
    # define network
    net = LinearVAE(in_size=in_size, in_channels=in_channels, latent_dim=latent_dim, context_dim=n_classes, device=DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    print(net)
    print()


    for epoch in range(epochs):
        net.train()
        net.to(DEVICE)
        running_loss, running_recons, running_kld = 0, 0, 0
        for i, (x_batch, y_batch) in enumerate(trainloader):
            # DataLoader returns CPU tensors; move per-batch for correct device placement
            x_batch = x_batch.to(DEVICE)
            # y_batch not used by vanilla VAE loss

            # VAE expects NCHW: add channel dim
            x_in = x_batch.unsqueeze(1)  # (B, 1, time=10, features=36)

            optimizer.zero_grad(set_to_none=True)
            generated, src, mu, logvar = net(x_in)
            

            loss_dict = net.loss(src, generated, mu, logvar, kld_weight_train)
            loss_dict["loss"].backward()
            optimizer.step()

            running_loss += loss_dict["loss"].detach()
            running_recons += loss_dict["reconstruction"].detach()
            running_kld += loss_dict["kld"].detach()

        print(
            "[{epoch}, {batch}%] loss: {loss:.4f} reconstruction loss: {recons:.4f} kld loss: {kld:.4f}".format(
                epoch=epoch+1,
                batch=100,
                loss=loss_dict["loss"].item(),
                recons=loss_dict["reconstruction"].item(),
                kld=loss_dict["kld"].item(),
            )
        )   
        n_batches = i + 1 if 'i' in locals() else 1
        print(
            "[{epoch}, train] loss: {loss:.4f} reconstruction loss: {recons:.4f} kld loss: {kld:.4f}".format(
                epoch=epoch+1,
                loss=(running_loss / n_batches).item(),
                recons=(running_recons / n_batches).item(),
                kld=(running_kld / n_batches).item(),
            )
        )
        error_graph.append((running_recons / n_batches).item())

        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_filename = f"{run_dir}/checkpoint_epoch_{epoch + 1}.pt"
            torch.save(net.state_dict(), checkpoint_filename)
            print(f"  Checkpoint saved: {checkpoint_filename}")

    net.eval()
    example_input = torch.randn(x_batch.size()).to(DEVICE).unsqueeze(1)
    traced_model = torch.jit.trace(net, example_input, check_trace=False)

    error_filepath = f"{run_dir}/error.csv"
    model_filepath = f"{run_dir}/model_jit.pt"
    model_fallback_filepath = f"{run_dir}/model.pt"
    
    torch.save(net.state_dict(), model_fallback_filepath)
    try:
        torch.jit.save(traced_model, model_filepath)
    except Exception as e:
        print(f"Error saving traced model: {e}")
    error_graph = pd.DataFrame(error_graph, columns=['epoch_loss'])
    error_graph.to_csv(error_filepath, index=False)
    print(f"Error graph saved to {error_filepath}")
    
    # plt.plot(error_graph['epoch_loss'])
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss Over Epochs')
    # plt.show()
