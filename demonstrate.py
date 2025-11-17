import torch
import torchvision
import numpy as np
import pandas as pd
import sys
import os

DEVICE = "cuda" #if torch.cuda.is_available() # else "cpu" # mps is almost always slower
# print(DEVICE)
if DEVICE == "cuda": torch.backends.cudnn.benchmark = True # enables cuDNN auto-tuner
torch.manual_seed(0)

# from LAFAN1_VAE_Experiment import ROOT_DIR
from vae import LinearVAE, VAE
import utils
from RobotMovementDataset import RobotMovementDataset

def iterative_generate(model, seed_sequence, num_iterations, dataset, out_len, device="cpu"):
    """
    model: trained VAE
    seed_sequence: numpy array (input_len x num_joints)
    num_iterations: how many 5-frame chunks to generate
    dataset: used for denormalization
    """
    model.eval()
    model.to(device)

    seq_in = dataset.input_len
    seq_out = out_len
    num_joints = seed_sequence.shape[1]

    generated = [seed_sequence.copy()]
    
    current_seq = seed_sequence.copy()

    for i in range(num_iterations):
        x = (
            torch.tensor(current_seq, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .unsqueeze(1)
        )
        with torch.no_grad():
            mu, logvar = model.encode(x.flatten(1))
            z = mu
            generated_y = model.decode(z).view(1, 1, seq_in, num_joints)
        
        y_slice = generated_y[:, :, -seq_out:, :].squeeze(0).squeeze(0)  # take last seq_out frames
        current_seq = current_seq[seq_out:, :]
        y_pred = y_slice.detach().cpu().numpy()
        
        current_seq = np.concatenate([current_seq, y_pred], axis=0)
        generated.append(y_pred)

    motion = np.concatenate(generated, axis=0)
    return motion.reshape(-1, motion.shape[-1])

def iterative_reconstruct(model, input_sequence, dataset, normalize, device="cpu", debug: bool = False):
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

def random_construct(model, input_length, dataset, normalize, device="cpu", debug: bool = False):

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
    

reconstruct = False
random = not reconstruct
folder_path = "LAFAN1_Retargeting_Dataset/g1/"
motions = get_filenames_in_folder(folder_path)
normalize = False
classes = []
n_classes = len(classes)
in_channels = 1
input_frames = 15
pred_length = 1
in_size = (input_frames, 36)
latent_dim = 128
epochs =  int(sys.argv[1]) if len(sys.argv) > 1 else 1000
batch_size = 15

model_filepath = 'C:/Users/Mark/OneDrive - The University of Texas at Austin/Documents/HCRL/LAFAN1_VAE_Experiment/LAFAN1_Retargeting_Dataset/g1/generated/model_' + str(len(motions)) + '_files_' + str(epochs) + '_epochs.pth'

dataset = RobotMovementDataset(filenames=motions, input_len=input_frames, output_len=input_frames, device=DEVICE, normalize=normalize, reconstruct=reconstruct)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

net = LinearVAE(in_size=in_size, in_channels=in_channels, latent_dim=latent_dim, context_dim=n_classes, device=DEVICE)
state_dict = torch.load(model_filepath, weights_only=True)
net.load_state_dict(state_dict)

filepath = 'C:/Users/Mark/OneDrive - The University of Texas at Austin/Documents/HCRL/LAFAN1_VAE_Experiment/LAFAN1_Retargeting_Dataset/g1/generated/demonstrate_' + str(len(motions)) + '_files_' + str(epochs) + '_epochs.csv'
random_filepath = 'C:/Users/Mark/OneDrive - The University of Texas at Austin/Documents/HCRL/LAFAN1_VAE_Experiment/LAFAN1_Retargeting_Dataset/g1/generated/random_' + str(len(motions)) + '_files_' + str(epochs) + '_epochs.csv'
if reconstruct:
    samples_denorm = iterative_reconstruct(net, dataset.raw_data, dataset, normalize, device=DEVICE)
elif random:
    samples_denorm = random_construct(net, 400, dataset, normalize, device=DEVICE)
else:
    rand_index = np.random.randint(0, len(dataset) - dataset.input_len - pred_length - 400)
    samples_denorm = iterative_generate(net, dataset.raw_data[rand_index:rand_index + dataset.input_len], 400, dataset, out_len=pred_length, device=DEVICE)
df_generated = pd.DataFrame(samples_denorm)

print(len(df_generated))
print("saving to:", random_filepath if random else filepath)
df_generated.to_csv(random_filepath if random else filepath, index=False)