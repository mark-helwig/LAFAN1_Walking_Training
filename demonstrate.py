import torch
import torchvision
import numpy as np
import pandas as pd
import sys

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

motion = 'walk1_subject1.csv'
normalize = False
reconstruct = False
classes = []
n_classes = len(classes)
in_channels = 1
input_frames = 15
pred_length = 1
in_size = (input_frames, 36)
latent_dim = 128
epochs =  int(sys.argv[1]) if len(sys.argv) > 1 else 200
batch_size = 15

dataset = RobotMovementDataset(filename=motion, input_len=input_frames, output_len=input_frames, device=DEVICE, normalize=normalize, reconstruct=reconstruct)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

net = LinearVAE(in_size=in_size, in_channels=in_channels, latent_dim=latent_dim, context_dim=n_classes, device=DEVICE)
state_dict = torch.load('generate_model_weights.pth', weights_only=True)
net.load_state_dict(state_dict)

folder = 'C:/Users/Mark/OneDrive - The University of Texas at Austin/Documents/HCRL/LAFAN1_VAE_Experiment/LAFAN1_Retargeting_Dataset/g1/'
start_index = 1191
samples_back_to_home = iterative_generate(net, dataset.df[start_index: start_index + dataset.input_len], 1000, dataset, out_len=pred_length, device=DEVICE)
df_back_to_home_seed = pd.DataFrame(dataset.df[start_index: start_index + dataset.input_len])
df_back_to_home = pd.DataFrame(samples_back_to_home)
df_back_to_home_seed.to_csv(folder + 'walk_backward_seed.csv', index=False)
df_back_to_home.to_csv(folder + 'walk_backward.csv', index=False)
