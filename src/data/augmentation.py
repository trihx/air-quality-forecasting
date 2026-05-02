"""Time Series Data Augmentation Module.

Implements techniques to artificially augment time series sequences
to prevent deep learning models from overfitting to sensor noise.
"""
import numpy as np
import torch

class TimeSeriesAugmenter:
    """Applies data augmentation to time series sequences."""
    
    def __init__(self, technique="jitter", sigma=0.05):
        """
        Args:
            technique (str): Augmentation technique to apply ('jitter' or 'scale').
            sigma (float): Standard deviation for Gaussian noise in jittering.
        """
        self.technique = technique
        self.sigma = sigma
        
    def __call__(self, X):
        """
        Apply augmentation to a batch of sequences.
        
        Args:
            X (np.ndarray or torch.Tensor): Input sequences of shape (batch, seq_len, features)
            
        Returns:
            Augmented sequences.
        """
        if self.technique == "jitter":
            return self.jitter(X)
        elif self.technique == "scale":
            return self.scale(X)
        return X
        
    def jitter(self, X):
        """
        Add Gaussian noise to the sequences.
        This helps models become robust against sensor micro-fluctuations.
        """
        if isinstance(X, torch.Tensor):
            noise = torch.randn_like(X) * self.sigma
            return X + noise
        else:
            noise = np.random.normal(loc=0, scale=self.sigma, size=X.shape)
            return X + noise
            
    def scale(self, X):
        """
        Scale the sequences by a random factor.
        This helps models generalize across different peak magnitudes.
        """
        if isinstance(X, torch.Tensor):
            # Shape (batch, 1, 1) to broadcast across seq_len and features
            factor = torch.randn(X.size(0), 1, 1, device=X.device) * self.sigma + 1.0
            return X * factor
        else:
            factor = np.random.normal(loc=1.0, scale=self.sigma, size=(X.shape[0], 1, 1))
            return X * factor
