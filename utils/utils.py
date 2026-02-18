from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

def imshow(img: torch.Tensor) -> None:
    img = img / 2 + 0.5 # unnormalize
    plt.imshow(np.transpose(img.numpy(), (1,2,0)))
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def get_encoder_size(in_size: int, hidden_dims: List[int], kernel: int, stride: int, padding: int) -> int:
    s = in_size
    for _ in hidden_dims:
        s = (s-kernel+2*padding)//stride + 1
    return s