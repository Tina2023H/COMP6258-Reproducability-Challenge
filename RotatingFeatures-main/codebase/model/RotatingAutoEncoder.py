from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig

from codebase.model import ConvDecoder, ConvEncoder
from codebase.utils import rotation_utils, model_utils

class RotatingAutoEncoder(nn.Module):
    def __init__(self, opt: DictConfig) -> None:
        super(RotatingAutoEncoder, self).__init__()

        self.opt = opt

        # Initialize the encoder and decoder modules.
        self.encoder = ConvEncoder.ConvEncoder(opt)
        self.decoder = ConvDecoder.ConvDecoder(
            opt, self.encoder.channel_per_layer, self.encoder.latent_dim,
        )

        '''
        Use the below layers for 4Shapes_RGBD

        self.hierarchical_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.ReLU()
            )
        ])
        '''

        # Hierarchical layers to process features at multiple levels of abstraction
        self.hierarchical_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.encoder.channel_per_layer[-1], self.encoder.channel_per_layer[-1] * 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.encoder.channel_per_layer[-1] * 2, self.encoder.channel_per_layer[-1], 3, padding=1),
                nn.ReLU()
            )
        ])

        if self.opt.input.dino_processed:
            self.dino = model_utils.load_dino_model()
            self.preprocess_model = nn.Sequential(
                nn.BatchNorm2d(self.opt.input.channel), nn.ReLU(),
            )

        self.output_weight = nn.Parameter(torch.empty(self.opt.input.channel))
        self.output_bias = nn.Parameter(torch.empty(1, self.opt.input.channel, 1, 1))
        nn.init.constant_(self.output_weight, 1)
        nn.init.constant_(self.output_bias, 0)

    def preprocess_input_images(self, input_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.opt.input.dino_processed:
            with torch.no_grad():
                dino_features = self.dino(input_images)
                dino_features = rearrange(dino_features, "b (h w) c -> b c h w", h=self.opt.input.image_size[0], w=self.opt.input.image_size[1])
            images_to_process = self.preprocess_model(dino_features)
            images_to_reconstruct = dino_features
        else:
            images_to_process = input_images
            images_to_reconstruct = input_images

        images_to_process = rotation_utils.add_rotation_dimensions(self.opt, images_to_process)
        return images_to_process, images_to_reconstruct

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder.convolutional:
            z = layer(z)
        z = rearrange(z, "... c h w -> ... (c h w)")
        z = self.encoder.linear(z)
        return z

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.decoder.linear(z)
        z = rearrange(z, "... (c h w) -> ... c h w", c=self.encoder.channel_per_layer[-1], h=self.encoder.latent_feature_map_size[0], w=self.encoder.latent_feature_map_size[1])
        for layer in self.decoder.convolutional:
            z = layer(z)
        reconstruction = self.apply_output_model(rotation_utils.get_magnitude(z))
        rotation_output, reconstruction = self.center_crop_reconstruction(z, reconstruction)
        return rotation_output, reconstruction

    def apply_output_model(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply scaling and bias to the reconstructed magnitude of rotating features to better match input values.
        """
        return torch.einsum("b c h w, c -> b c h w", z, self.output_weight) + self.output_bias

    def center_crop_reconstruction(self, rotation_output: torch.Tensor, reconstruction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adjust the size of the reconstruction to match the input size by center cropping.
        """
        if self.opt.input.dino_processed:
            rotation_output = rotation_output[:, :, :, 1:-1, 1:-1]
            reconstruction = reconstruction[:, :, 1:-1, 1:-1]
        return rotation_output, reconstruction

    def forward(self, input_images: torch.Tensor, labels: dict, evaluate: bool = False) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through the model integrating preprocessing, encoding, decoding, and optional evaluation.
        """
        images_to_process, images_to_reconstruct = self.preprocess_input_images(input_images)

        z = self.encode(images_to_process)
        for layer in self.hierarchical_layers:
            z = layer(z)

        rotation_output, reconstruction = self.decode(z)

        loss = nn.functional.mse_loss(reconstruction, images_to_reconstruct)
        metrics = rotation_utils.run_evaluation(self.opt, rotation_output, labels) if evaluate else {}
        return loss, metrics

