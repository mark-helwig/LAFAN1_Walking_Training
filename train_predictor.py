import torch
import torchvision
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import os
from datetime import datetime

DEVICE = "cuda" #if torch.cuda.is_available() # else "cpu" # mps is almost always slower
# print(DEVICE)
if DEVICE == "cuda": torch.backends.cudnn.benchmark = True # enables cuDNN auto-tuner
torch.manual_seed(0)

# from LAFAN1_VAE_Experiment import ROOT_DIR
from vae import LinearVAE, VAE
from latentDynamics import ResidualLatentDynamics, LatentMLPDynamics
import utils
from RobotMovementDataset import RobotMovementDataset

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

def run_training_loop(net, trainloader, encoder, optimizer, epochs, DEVICE, error_graph):
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
    print("Training complete.")

if __name__ == "__main__":

    folder_path = "LAFAN1_Retargeting_Dataset/g1/"
    filenames = get_walking_filenames_in_folder(folder_path)
    
    encoder_filepath = 'models/encoder/model_fallback_1000_epochs_20260202_205355.pt'

    normalize = False
    reconstruct = False

    motions = filenames
    classes = []
    n_classes = len(classes)
    in_channels = 1

    input_frames = 24
    pred_length = 8
    in_size = (input_frames, 36)
    latent_dim = 128
    epochs =  int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    batch_size = 15

    error_graph = []


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
    net = LatentMLPDynamics(latent_dim=latent_dim, hidden_dim=256, device=DEVICE)
    net.to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

    print(net)
    print()

    # run training loop
    run_training_loop(net, trainloader, encoder, optimizer, epochs, DEVICE, error_graph)

    net.eval()
    example_input = torch.randn(batch_size, latent_dim, device=DEVICE)
    traced_model = torch.jit.trace(net, example_input, check_trace=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fallback_filepath = f'models/predictor/pred_length_{pred_length}_fallback_{epochs}_epochs_{timestamp}.pt'
    error_filepath = f'models/predictor/pred_length_{pred_length}_error_{epochs}_epochs_{timestamp}.csv'
    model_filepath = f'models/predictor/pred_length_{pred_length}{epochs}_epochs_{timestamp}.pt'
    
    torch.save(net.state_dict(), fallback_filepath)
    try:
        torch.jit.save(traced_model, model_filepath)
    except Exception as e:
        print(f"Error saving traced model: {e}")
    error_graph = pd.DataFrame(error_graph, columns=['epoch_loss'])
    error_graph.to_csv(error_filepath, index=False)
    print(f"Error graph saved to {error_filepath}")
