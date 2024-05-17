import torch.nn as nn
from omegaconf import DictConfig

from codebase.model import RotatingLayers
from codebase.utils import model_utils

class ConvEncoder(nn.Module):
    def __init__(self, opt: DictConfig) -> None:
  
        super().__init__()

        self.channel_per_layer = [
            opt.input.channel,
            opt.model.hidden_dim,
            2 * opt.model.hidden_dim,
            2 * opt.model.hidden_dim,
            2 * opt.model.hidden_dim,
        ]

        # Set a fixed dropout probability
        dropout_prob = 0.3  # Default dropout probability

        # Initialize convolutional layers with consistent dropout applied
        self.convolutional = nn.ModuleList([
            nn.Sequential(
                RotatingLayers.RotatingConv2d(
                    opt,
                    self.channel_per_layer[0],
                    self.channel_per_layer[1],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                ),
                nn.Dropout2d(dropout_prob)
            ),
            nn.Sequential(
                RotatingLayers.RotatingConv2d(
                    opt,
                    self.channel_per_layer[1],
                    self.channel_per_layer[1],
                    kernel_size=3,
                    padding=1,
                    stride=1,
                ),
                nn.Dropout2d(dropout_prob)
            ),
            nn.Sequential(
                RotatingLayers.RotatingConv2d(
                    opt,
                    self.channel_per_layer[1],
                    self.channel_per_layer[2],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                ),
                nn.Dropout2d(dropout_prob)
            ),
            nn.Sequential(
                RotatingLayers.RotatingConv2d(
                    opt,
                    self.channel_per_layer[2],
                    self.channel_per_layer[2],
                    kernel_size=3,
                    padding=1,
                    stride=1,
                ),
                nn.Dropout2d(dropout_prob)
            ),
            nn.Sequential(
                RotatingLayers.RotatingConv2d(
                    opt,
                    self.channel_per_layer[2],
                    self.channel_per_layer[3],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                ),
                nn.Dropout2d(dropout_prob)
            ),
            nn.Sequential(
                RotatingLayers.RotatingConv2d(
                    opt,
                    self.channel_per_layer[3],
                    self.channel_per_layer[3],
                    kernel_size=3,
                    padding=1,
                    stride=1,
                ),
                nn.Dropout2d(dropout_prob)
            )
        ])

        # Compute latent dimensions and initialize a linear layer
        self.latent_feature_map_size, self.latent_dim = model_utils.get_latent_dim(
            opt, self.channel_per_layer[-1]
        )
        self.linear = RotatingLayers.RotatingLinear(
            opt, self.latent_dim, opt.model.linear_dim
        )

