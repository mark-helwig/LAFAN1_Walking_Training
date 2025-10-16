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
from vae import PhaseVAE
import utils
from RobotMovementDataset import RobotMovementDataset

def generate_sequence_with_phase(model, dataset, seed_seq,
                                 num_cycles=4, steps_per_cycle=45,
                                 device="cpu"):
    

    model.eval()

    seq_in = model.seq_in
    seq_out = model.seq_out
    num_joints = model.input_dim
    total_steps = num_cycles * steps_per_cycle

    # Convert seed to tensor & normalize
    
    # seed_seq = (seed_seq - dataset.mean) / dataset.std
    seed_seq = torch.tensor(seed_seq, dtype=torch.float32)
    seed_seq = seed_seq.unsqueeze(0).to(device)  # (1, seq_in, num_joints)

    # Fixed latent vector for consistent walking style
    z_fixed = torch.randn(1, model.latent_dim, device=device)

    generated_frames = []

    # Start phase at 0
    for step in range(total_steps):
        # Compute current phase angle in radians [0, 2π)
        phase = (2 * np.pi * step / steps_per_cycle) % (2 * np.pi)
        phase_tensor = torch.tensor([phase], dtype=torch.float32, device=device)

        # Predict next seq_out frames
        y_pred, _, _ = model(seed_seq, phase_tensor, z_fixed)
        y_pred = y_pred.squeeze(0)  # (seq_out, num_joints)
        y_pred = y_pred.detach()
        generated_frames.append(y_pred.detach().cpu().numpy())

        # Slide window forward: drop oldest frames, append newest
        seed_seq = torch.cat([seed_seq[:, seq_out:, :], y_pred.unsqueeze(0)], dim=1)

    # Stack all predictions
    motion = np.concatenate(generated_frames, axis=0)  # (total_steps*seq_out, num_joints)
    motion_denorm = dataset.denormalize(motion)
    return motion_denorm


if __name__ == "__main__":
    motion = 'walk1_subject1_shortened.csv'
    normalize = True

    classes = []
    n_classes = len(classes)
    in_channels = 1
    # Input patches are shaped (time, features) = (20, 36)
    in_size = (20, 36)
    latent_dim = 24
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    batch_size = 32
    pred_length = 2
    gait_period = 45

    dataset = RobotMovementDataset(filename=motion, input_len=20, output_len=pred_length, gait_period=gait_period, device=DEVICE, normalize=normalize, phase=True)
    # make dataset
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=2, pin_memory=True, drop_last=True)   
    print(f"Dataset batches: {len(trainloader)}")
    # kld_weight_train = 1/(2*len(trainloader))
    kld_weight_train = .1
    print("kld_weight_train:", kld_weight_train)
    
    # define network
    net = PhaseVAE(input_dim=in_size[1], seq_in=in_size[0], seq_out=pred_length, latent_dim=latent_dim, device=DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

    print(net)
    print()


    for epoch in range(epochs):
        net.train()
        running_loss, running_recons, running_kld, running_smooth = 0, 0, 0, 0
        for i, (x, y, phase) in enumerate(trainloader):
            x, y, phase = x.to(DEVICE), y.to(DEVICE), phase.to(DEVICE)

            y_pred, mu, logvar = net(x, phase)
            loss_dict = net.loss(y_pred, y, mu, logvar,
                                                kld_weight=1e-2, smooth_weight=0.4)

            optimizer.zero_grad()
            loss_dict['loss'].backward()
            optimizer.step()

            running_loss += loss_dict["loss"].detach()
            running_recons += loss_dict["reconstruction"].detach()
            running_kld += loss_dict["kld"].detach()
            running_smooth += loss_dict["smooth"].detach()

        print(
            "[{epoch}, {batch}%] loss: {loss:.4f} reconstruction loss: {recons:.4f} kld loss: {kld:.4f} smooth_loss: {smooth_loss:.4f}".format(
                epoch=epoch+1,
                batch=100,
                loss=loss_dict["loss"].item(),
                recons=loss_dict["reconstruction"].item(),
                kld=loss_dict["kld"].item(),
                smooth_loss=loss_dict["smooth"].item(),
            )
        )   
        n_batches = i + 1 if 'i' in locals() else 1
        print(
            "[{epoch}, train] loss: {loss:.4f} reconstruction loss: {recons:.4f} kld loss: {kld:.4f} smooth loss: {smooth_loss:.4f}".format(
                epoch=epoch+1,
                loss=(running_loss / n_batches).item(),
                recons=(running_recons / n_batches).item(),
                kld=(running_kld / n_batches).item(),
                smooth_loss=(running_smooth / n_batches).item(),
            )
        )

    net.to(DEVICE)
    filepath = 'C:/Users/Mark/OneDrive - The University of Texas at Austin/Documents/HCRL/LAFAN1_VAE_Experiment/LAFAN1_Retargeting_Dataset/g1/generated_walk1_subject1_shortened_' + str(epochs) + '_epochs.csv'
    samples_denorm = generate_sequence_with_phase(net, dataset, dataset.df[0:dataset.input_len], 5, steps_per_cycle=gait_period, device=DEVICE)
    df_generated = pd.DataFrame(samples_denorm)
    df_generated[df_generated.columns[0]] = 0
    df_generated.to_csv(filepath, index=False)  
    print(f"Generated sequence saved to {filepath}")