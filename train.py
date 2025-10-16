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
    

    for i in range(iterations):
        # Model expects NCHW: add batch and channel dims -> (1, 1, time, num_joints)
        current_seq = input_sequence[i*seq_out : i*seq_out + seq_in]
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
            generated_y = model.decode(z).view(1, 1, seq_in, num_joints)
            detached = generated_y.squeeze(0).detach().cpu().numpy()
        generated.append(detached)

    motion = np.concatenate(generated, axis=0)
    return motion.reshape(-1, motion.shape[-1])


if __name__ == "__main__":
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
    # make dataset
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=2, pin_memory=True, drop_last=True)   
    
    kld_weight_train = 1/len(trainloader)
    print(f"Dataset batches: {len(trainloader)}")
    # kld_weight_train = 1/(2*len(trainloader))
    print("kld_weight_train:", kld_weight_train)
    
    # define network
    net = LinearVAE(in_size=in_size, in_channels=in_channels, latent_dim=latent_dim, context_dim=n_classes, device=DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

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

    filepath = 'C:/Users/Mark/OneDrive - The University of Texas at Austin/Documents/HCRL/LAFAN1_VAE_Experiment/LAFAN1_Retargeting_Dataset/g1/generated_walk1_subject1_shortened_' + str(epochs) + '_epochs.csv'
    # net.generate_sequence(200, 'C:/Users/Mark/OneDrive - The University of Texas at Austin/Documents/HCRL/LAFAN1_VAE_Experiment/LAFAN1_Retargeting_Dataset/g1/generated_walk1_subject1_shortened_' + str(epochs) + '_epochs.csv', dataset)
    if reconstruct:
        samples_denorm = iterative_reconstruct(net, dataset.df, dataset, normalize, device=DEVICE)
    else:
        samples_denorm = iterative_generate(net, dataset.df[:dataset.input_len], 400, dataset, out_len=pred_length, device=DEVICE)
    df_generated = pd.DataFrame(samples_denorm)
    # df_generated[df_generated.columns[0]] = 0
    print(len(df_generated))
    df_generated.to_csv(filepath, index=False)  
    if reconstruct:
        torch.save(net.state_dict(), 'reconstruct_model_weights.pth')
    else:
        torch.save(net.state_dict(), 'generate_model_weights.pth')
    print(f"Generated sequence saved to {filepath}")

    
