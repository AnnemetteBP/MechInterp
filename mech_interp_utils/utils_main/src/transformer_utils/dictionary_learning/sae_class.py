from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SAE(nn.Module):
    def __init__(self,
                 input_dim:Any,
                 dict_size:int=512,
                 sparsity_lambda=1e-3,
                 deterministic_sae:bool=True) -> None:
        
        super().__init__()

        self.encoder = nn.Linear(input_dim, dict_size, bias=False)
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)
        self.sparsity_lambda = sparsity_lambda
        self.deterministic_sae = deterministic_sae
    
    """
    The SAE (Sparse Autoencoder) class learns a compressed, interpretable representation of high-dimensional model activations
    by encoding them into a sparse latent space and reconstructing them.
    It helps identify which latent dimensions (features) most influence a token's representation,
    supporting interpretability and saliency analysis — e.g., for comparing full-precision vs quantized models.
    """

    def __set_deterministic_sae(self:SAE, seed:int=42) -> None:
        """ 
        Forces PyTorch to use only deterministic operations (e.g., disables non-deterministic GPU kernels).
        This is crucial for reproducibility: given the same inputs and model state, to get the same outputs every time.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    def forward(self:SAE, x) -> Tuple[Any, Any]:
        x = x.to(self.encoder.weight.dtype)  # auto match dtype for float16
        code = self.encoder(x)
        reconstruction = self.decoder(code)
        
        return reconstruction, code


    def train_sae(self:SAE, data:Any, epochs:int=5, lr=1e-3, batch_size:int=8) -> List:
        if self.deterministic_sae:
            self.__set_deterministic_sae()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for batch, in loader:
                batch = batch.to(self.encoder.weight.dtype)  # match dtype for forward to work for float16
                recon, code = self.forward(batch)

                # convert both to float32 for stable loss calculation
                loss = F.mse_loss(recon.float(), batch.float()) + self.sparsity_lambda * code.abs().mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        
        return losses


    def encode(self:SAE, x:Any) -> Any:
        x = x.to(self.encoder.weight.dtype)  # Ensure dtype e.g., float16 matches encoder weights
        return self.encoder(x)