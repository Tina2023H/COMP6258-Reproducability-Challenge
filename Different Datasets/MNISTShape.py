from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset


class MNISTShapesDataset(Dataset):
    def __init__(self, opt: DictConfig, partition: str) -> None:
        """
        Initialize the MNISTShapes datasets.
        this dataset is from the paper "https://arxiv.org/abs/2204.02075"

        Args:
            opt (DictConfig): Configuration options.
            partition (str): Dataset partition ("train", "val", or "test").
        """
        super(MNISTShapesDataset, self).__init__()

        self.opt = opt
        self.root_dir = Path(opt.cwd, opt.input.load_path)
        self.partition = partition


        file_name = Path(self.root_dir, f"{opt.input.file_name}_{partition}.npz")
 
        dataset = np.load(file_name)
        self.images = dataset["images"]  + 2
        self.images = self.images.astype(np.float32)
        self.pixelwise_instance_labels = dataset["labels"]

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.images.shape[0]

  
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the input image and corresponding gt_labels.
        """
        input_image = self.images[idx]
        labels = {"pixelwise_instance_labels": self.pixelwise_instance_labels[idx]}
        return input_image, labels
