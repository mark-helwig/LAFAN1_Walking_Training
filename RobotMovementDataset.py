import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class RobotMovementDataset(Dataset):
    def __init__(self, filenames, input_len=10, output_len=5, gait_period=45, device="cpu", normalize=True, phase=False, reconstruct=False):
        """
        df: pandas DataFrame where each row is a time step and each column a joint feature
        input_len: number of past frames used as input
        output_len: number of future frames predicted
        """
        df, self.lengths = self.get_data(filenames)
        self.phase = phase
        self.raw_data = df.astype(np.float32).values  # convert to numpy for slicing
        self.torch_df = torch.tensor(self.raw_data, dtype=torch.float32, device=device)
        self.input_len = input_len
        self.output_len = output_len
        if reconstruct:
            self.output_len = input_len
        self.reconstruct = reconstruct
        # keep device for reference, but tensors will be created on CPU and moved in the training loop
        self.device = device

        if normalize:
            self.mean = self.raw_data.mean(axis=0, keepdims=True)
            self.std = self.raw_data.std(axis=0, keepdims=True) + 1e-8
            self.raw_data = (self.raw_data - self.mean) / self.std
        else:
            self.mean, self.std = np.zeros((1, self.raw_data.shape[1])), np.ones((1, self.raw_data.shape[1]))

        self.samples = len(df) - input_len - output_len + 1
        if self.samples <= 0:
            raise ValueError("Not enough rows in DataFrame for given input/output lengths.")
        
        self.phase_per_frame = np.array([
            2 * np.pi * (i % gait_period) / gait_period for i in range(len(self.raw_data))
        ], dtype=np.float32)

    def get_data(self, filenames):
        data = pd.DataFrame()
        lengths = []
        for i in range(len(filenames)):
            file_data = pd.read_csv('LAFAN1_Retargeting_Dataset/g1/' + filenames[i], header=None)
            data = pd.concat([data, file_data], ignore_index=True)
            lengths.append(len(data))
        # data.to_csv('dataset.csv', index=False)
        return data, lengths
    
    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        adjusted_index = idx
        for i in range(len(self.lengths)):
            if idx < self.lengths[i]:
                if idx + self.input_len + self.output_len > self.lengths[i]:
                    adjusted_index = self.lengths[i] - self.input_len - self.output_len

        x = self.raw_data[adjusted_index : adjusted_index + self.input_len]
        if self.reconstruct:
            y = x
        else:
            y = self.raw_data[adjusted_index + self.output_len : adjusted_index + self.input_len + self.output_len]
        phase = self.phase_per_frame[adjusted_index + self.input_len - 1]
        
        # Convert to PyTorch tensors
        # Return CPU tensors; training loop will handle device placement
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        phase_tensor = torch.tensor(phase, dtype=torch.float32)

        if self.phase:
            return x_tensor, y_tensor, phase_tensor
        return x_tensor, y_tensor

    def denormalize(self, data):
        """Convert normalized data back to original scale."""
        return data * self.std + self.mean