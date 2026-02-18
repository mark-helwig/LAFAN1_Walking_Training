import torch
import torchvision
import numpy as np
import pandas as pd
import sys
import os
import argparse
from datetime import datetime

DEVICE = "cuda" #if torch.cuda.is_available() # else "cpu" # mps is almost always slower
# print(DEVICE)
if DEVICE == "cuda": torch.backends.cudnn.benchmark = True # enables cuDNN auto-tuner
torch.manual_seed(0)

# from LAFAN1_VAE_Experiment import ROOT_DIR
from models.vae import LinearVAE
from models.mlp import LatentMLP
import utils
from datasets.RobotMovementDataset import RobotMovementDataset

def predict(encoder, predictor, input_sequence, dataset, device="cpu"):
    """
    encoder: trained VAE encoder
    predictor: trained latent dynamics predictor
    input_sequence: numpy array (time x num_joints)
    dataset: used for denormalization
    """
    encoder.eval()
    predictor.eval()
    encoder.to(device)
    predictor.to(device)

    seq_in = dataset.input_len
    seq_out = dataset.output_len
    num_joints = input_sequence.shape[1]

    generated = []

    for i in range(len(input_sequence) - seq_in):
        # Model expects NCHW: add batch and channel dims -> (1, 1, time, num_joints)
        current_seq = input_sequence[i : i + seq_in]
        x = (
            torch.tensor(current_seq, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .unsqueeze(1)
        )

        with torch.no_grad():
            mu, logvar = encoder.encode(x.flatten(1))
            z = mu
            # Predict next latent state
            z_pred = predictor(z)
            generated_y = encoder.decode(z_pred).view(1, 1, seq_in, num_joints)
            # take only the last time-step (shape: 1 x num_joints) instead of the full seq_in x num_joints
            detached = generated_y.squeeze(0).detach().cpu().numpy()
            last_frame = detached[0, -1, :].reshape(1, -1)
        generated.append(last_frame)

    motion = np.concatenate(generated, axis=0)
    return motion.reshape(-1, motion.shape[-1])

def reconstruct(model, input_sequence, dataset, normalize, device="cpu", debug: bool = False):
    """
    model: trained VAE
    seed_sequence: numpy array (input_len x num_joints)
    num_iterations: how many 5-frame chunks to generate
    dataset: used for denormalization
    """
    model.eval()
    model.to(device)

    seq_in = dataset.input_len
    seq_out = dataset.output_len
    num_joints = input_sequence.shape[1]

    generated = []

    iterations = len(input_sequence) // seq_out
    

    for i in range(len(input_sequence) - seq_in):
        # Model expects NCHW: add batch and channel dims -> (1, 1, time, num_joints)
        current_seq = input_sequence[i : i + seq_in]
        if normalize:
            current_seq = (current_seq - dataset.mean) / dataset.std
        x = (
            torch.tensor(current_seq, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .unsqueeze(1)
        )

        with torch.no_grad():
            mu, logvar = model.encode(x.flatten(1))
            z = mu
            # z = model.reparameterize(mu, logvar)
            generated_y = model.decode(z).view(1, 1, seq_in, num_joints)
            # take only the last time-step (shape: 1 x num_joints) instead of the full seq_in x num_joints
            detached = generated_y.squeeze(0).detach().cpu().numpy()
            last_frame = detached[0, -1, :].reshape(1, -1)
        generated.append(last_frame)

    motion = np.concatenate(generated, axis=0)
    return motion.reshape(-1, motion.shape[-1])

def random(model, input_length, dataset, normalize, device="cpu", debug: bool = False):

    model.eval()
    model.to(device)

    seq_in = dataset.input_len
    seq_out = dataset.output_len

    generated = []

    iterations = input_length // seq_out
    

    for i in range(input_length - seq_in):
        with torch.no_grad():
            # z = model.reparameterize(mu, logvar)
            generated_y = model.sample(1)
            detached = generated_y.squeeze(0).detach().cpu().numpy()
            last_frame = detached[0, -1, :].reshape(1, -1)
        generated.append(last_frame)

    motion = np.concatenate(generated, axis=0)
    return motion.reshape(-1, motion.shape[-1])


def get_walking_filenames_in_folder(folder_path):
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

    parser = argparse.ArgumentParser(description="Demonstrate VAE motion generation or reconstruction.")
    parser.add_argument('--test', choices=['predict', 'random', 'reconstruct'], required=True, help="Type of test to perform: 'predict' for latent space prediction, 'random' for random sampling, 'reconstruct' for reconstruction from input.")
    parser.add_argument('--encoder', required=True, help="Path to the trained encoder model file.")
    parser.add_argument('--predictor', required=False, help="Path to the trained predictor model file. Needed only for 'predict' test.")
    parser.add_argument('--length', type=int, required=False, help="Length of the output sequence.")
    args = parser.parse_args()

    test = args.test
    encoder_filepath = 'models/encoder/' + args.encoder + '.pth'
    predictor_filepath = 'models/predictor/' + args.predictor + '.pth'

    folder_path = "LAFAN1_Retargeting_Dataset/g1/"
    motions = get_walking_filenames_in_folder(folder_path)
    normalize = False
    in_channels = 1
    input_frames = 15
    pred_length = 1
    in_size = (input_frames, 36)
    latent_dim = 128
    batch_size = 15

    dataset = RobotMovementDataset(filenames=motions, input_len=input_frames, output_len=input_frames, device=DEVICE, normalize=normalize, reconstruct=reconstruct)
    output_length = args.length if args.length else len(dataset.raw_data)

    encoder = LinearVAE(in_size=in_size, in_channels=in_channels, latent_dim=latent_dim, context_dim=0, device=DEVICE)
    predictor = LatentMLP(latent_dim=latent_dim, device=DEVICE)
    encoder.load_state_dict(torch.load(encoder_filepath, weights_only=True))
    predictor.load_state_dict(torch.load(predictor_filepath, weights_only=True))
    encoder.to(DEVICE)
    predictor.to(DEVICE)
    
    current_time_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    
    
    match test:
        case 'predict':
            out_filepath = 'LAFAN1_Retargeting_Dataset/g1/predict/demonstrate_' + current_time_str + '.csv'
            samples = predict(encoder, predictor, dataset.raw_data[:output_length], dataset, device=DEVICE)
        case 'random':
            out_filepath = 'LAFAN1_Retargeting_Dataset/g1/random/random_' + current_time_str + '.csv'
            samples = random(encoder, output_length, dataset, normalize, device=DEVICE)
        case 'reconstruct':
            out_filepath = 'LAFAN1_Retargeting_Dataset/g1/reconstruct/demonstrate_' + current_time_str + '.csv'
            samples = reconstruct(encoder, dataset.raw_data[:output_length], dataset, normalize, device=DEVICE)
        case _:
            print(f"Unknown test type: {test}")
            sys.exit(1)
    df_generated = pd.DataFrame(samples)

    print("saving to:", out_filepath)
    df_generated.to_csv(out_filepath, index=False)