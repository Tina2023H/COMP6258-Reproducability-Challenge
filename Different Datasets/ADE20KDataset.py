import os
from typing import Tuple, Dict

import torch
import torchvision.transforms as transforms
from PIL import Image
from omegaconf import DictConfig
from torch.utils.data import Dataset


class ADE20KDataset(Dataset):
    def __init__(self, opt: DictConfig, partition: str) -> None:
        """
        Initialize the Pascal ADE20K Dataset.
        See for https://groups.csail.mit.edu/vision/datasets/ADE20K/ more information about the dataset.

        Args:
            opt (DictConfig): Configuration options.
            partition (str): Dataset partition ("train", "val", or "test").
        """
        super(ADE20KDataset, self).__init__()

        self.opt = opt
        self.partition = partition


        # Load  dataset.
        self.root_dir = os.path.join(opt.cwd, opt.input.load_path)

        partition_dir = os.path.join(
            self.root_dir, "images", f"{self.partition}.txt"
        )

        partition_dir_anno = os.path.join(
            self.root_dir, "seg", f"{self.partition}.txt"
        )

        with open(partition_dir) as f:
            file_names = [x.strip() for x in f.readlines() if x.strip()]

        with open(partition_dir_anno) as f:
            anno_names = [x.strip() for x in f.readlines() if x.strip()]
        self.images = [
            os.path.join(self.root_dir, "images","repeat_ade20k", f"{x}.jpg") for x in file_names
        ]
        
        self.pixelwise_class_labels = [
            os.path.join(self.root_dir, "images","repeat_ade20k", f"{x}.png")
            for x in anno_names
        ]
        self.pixelwise_instance_labels = [
             os.path.join(self.root_dir, "images","repeat_ade20k", f"{x}.png")
             for x in anno_names
        ]

        # Initialize image processing operations.
        self.to_tensor = transforms.ToTensor()
        self.image_size = 224
        self.random_crop = transforms.Compose(
            [transforms.RandomCrop(self.image_size), transforms.RandomHorizontalFlip()]
        )
        self.center_crop = transforms.CenterCrop(self.image_size)
        self.image_resize = transforms.Resize(
            self.image_size, interpolation=transforms.InterpolationMode.NEAREST
        )

        # For evaluation on validation and test set: use larger label size.
        self.eval_size = 320
        self.label_center_crop = transforms.CenterCrop(self.eval_size)
        self.label_resize = transforms.Resize(
            self.eval_size, interpolation=transforms.InterpolationMode.NEAREST
        )

        # Normalize input images using mean and standard deviation of ImageNet.
        self.normalize = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )
        
        self.num_classes = 20


    def __len__(self) -> int:
        """
        Get the number of images in the dataset.

        Returns:
            int: Number of images.
        """
        return len(self.images)

    @staticmethod
    def _preprocess_pascal_labels(labels: torch.Tensor) -> torch.Tensor:
        """
        Preprocess Pascal VOC labels by converting to integer 255-scale and
        marking object boundaries as "ignore" label.

        Args:
            labels (torch.Tensor): The input labels.

        Returns:
            torch.Tensor: The preprocessed labels.
        """
        labels = labels * 255
        labels[labels == 255] = -1  # "Ignore" label throughout the codebase is -1.
        return labels

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the input image and corresponding gt_labels.
        """

        input_image = Image.open(self.images[idx]).convert("RGB")      
        pixelwise_class_labels = Image.open(self.pixelwise_class_labels[idx])
        
        try:
            pixelwise_instance_labels = Image.open(self.pixelwise_instance_labels[idx])
        except FileNotFoundError as e:
            # Instance labels are not available for all images in the trainaug set.
            if self.partition != "train":
                raise FileNotFoundError(
                    "Instance labels should only be missing for the trainaug partition."
                ) from e
            # Create an empty target.
            pixelwise_instance_labels = Image.new(
                "L", (pixelwise_class_labels.width, pixelwise_class_labels.height), 0
            )

        input_image = self.to_tensor(input_image)
             
        pixelwise_class_labels = self._preprocess_pascal_labels(
            self.to_tensor(pixelwise_class_labels)
        )
        pixelwise_instance_labels = self._preprocess_pascal_labels(
            self.to_tensor(pixelwise_instance_labels)
        )

        if "train" in self.partition:
            # Apply the same random transformations to both inputs and gt_labels.
            miniW = min(input_image.shape[1],pixelwise_class_labels.shape[1], pixelwise_instance_labels.shape[1])
            miniH = min(input_image.shape[2],pixelwise_class_labels.shape[2], pixelwise_instance_labels.shape[2])
            inputResize = transforms.Resize((miniW,miniH))
            input_image = inputResize(input_image)
            pixelwise_class_labels = inputResize(pixelwise_class_labels)
            pixelwise_instance_labels = inputResize(pixelwise_instance_labels)
            
            all_inputs = torch.cat(
                (input_image, pixelwise_class_labels,pixelwise_instance_labels), 0
            )
            
            all_inputs = self.random_crop(self.image_resize(all_inputs))

            input_image, pixelwise_class_labels,pixelwise_instance_labels = (
                all_inputs[:3],
                all_inputs[-2],
                all_inputs[-1],
            )

        else:
            # Apply center crop to validation and test images and labels.
            input_image = self.center_crop(self.image_resize(input_image))
            pixelwise_instance_labels = self.label_center_crop(
                self.label_resize(pixelwise_instance_labels)
            )[0]
            pixelwise_class_labels = self.label_center_crop(
                self.label_resize(pixelwise_class_labels)
            )[0]

        labels = {
            "pixelwise_instance_labels": pixelwise_class_labels

        }
        return input_image, labels
