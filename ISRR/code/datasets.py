import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import ImageTransforms

# Construct particular dataset
class SRDataset(Dataset):
    """
    Dataset Loader
    """

    def __init__(self, data_folder, split, crop_size, scaling_factor, lr_img_type, hr_img_type, test_data_name=None):
        """
        :data_folder: # Path to the folder where the Json data file is located
        :split: 'train' or 'test'
        :crop_size: High-resolution image crop size (the actual training will not use the original image to zoom in, but rather a sub-block of the original image to zoom out)
        :scaling_factor: Enlargement ratio
        :lr_img_type: Low resolution image pre-processing method
        :hr_img_type: High-resolution image pre-processing method
        :test_data_name: In the case of the evaluation phase, the name of the specific dataset to be evaluated is given, e.g. "Set14"
        """

        self.data_folder = data_folder
        self.split = split.lower()
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.test_data_name = test_data_name

        assert self.split in {'train', 'test'}
        if self.split == 'test' and self.test_data_name is None:
            raise ValueError("Please provide the name of dataset!")
        assert lr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}
        assert hr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}

        # If training, all images must have a fixed resolution to ensure that the magnification can be rounded off
        # If testing, there is no need to limit the length and width of the images
        if self.split == 'train':
            assert self.crop_size % self.scaling_factor == 0, "Crop size not divisible by enlargement scale"

        # Read image path
        if self.split == 'train':
            with open(os.path.join(data_folder, 'train_images.json'), 'r') as j:
                self.images = json.load(j)
        else:
            with open(os.path.join(data_folder, self.test_data_name + '_test_images.json'), 'r') as j:
                self.images = json.load(j)

        # Data processing methods
        self.transform = ImageTransforms(split=self.split,
                                         crop_size=self.crop_size,
                                         scaling_factor=self.scaling_factor,
                                         lr_img_type=self.lr_img_type,
                                         hr_img_type=self.hr_img_type)

    def __getitem__(self, i):
        """
        In order to use PyTorch's DataLoader, this method must be provided.
        :parameter i: image retrieval number
        :return: returns the i-th low and high resolution image pair
        """
        # Load images
        img = Image.open(self.images[i], mode='r')
        img = img.convert('RGB')
        if img.width <= 96 or img.height <= 96:
            print(self.images[i], img.width, img.height)
        lr_img, hr_img = self.transform(img)

        return lr_img, hr_img

    def __len__(self):
        """
        In order to use PyTorch's DataLoader, this method must be provided.

        :Returns: the total number of images loaded
        """
        return len(self.images)
