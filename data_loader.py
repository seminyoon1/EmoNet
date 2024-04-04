import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import ipdb
import numpy as np
import imageio
import scipy.misc
import glob

class EmotionNet(Dataset):
    def __init__(self, image_size, metadata_path, transform, mode, fold, shuffling=False):
        self.transform = transform
        self.shuffling = shuffling
        self.image_size = image_size
        self.mode = mode
        self.filenames = []
        self.labels = []
        
        # Mapping class names to integer labels
        self.class_to_index = {'angry': 0, 'happy': 1, 'relaxed': 2, 'sad': 3}
        
        # Traverse subdirectories and collect image files
        for root, _, files in os.walk(metadata_path):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.filenames.append(os.path.join(root, file))
                    # Extract label from subdirectory name
                    label = os.path.basename(os.path.dirname(self.filenames[-1]))
                    self.labels.append(self.class_to_index[label])  # Convert label to integer
        
        # Check if data needs shuffling
        if self.shuffling:
            combined = list(zip(self.filenames, self.labels))
            random.shuffle(combined)
            self.filenames[:], self.labels[:] = zip(*combined)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(self.filenames[index])
        label = self.labels[index]
        image = self.transform(image)
        return image, label
    
def get_loader(metadata_path, image_size, batch_size, mode='train', fold='0', num_workers=4):
    # Normalize images using ImageNet statistics
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1.0, 1.0, 1.0])

    # Define transformations based on the mode
    if mode == 'train' or mode == 'sample':
      transform = transforms.Compose([
          transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0),
          transforms.RandomHorizontalFlip(),
          transforms.Resize((image_size, image_size)),  # Resize images to a fixed size
          transforms.ToTensor(),
          normalize
      ])
    else:
      transform = transforms.Compose([
          transforms.Resize((image_size, image_size)),  # Resize images to a fixed size
          transforms.ToTensor(),
          normalize
      ])

    print("Folder URL:", metadata_path)
    # Initialize EmotionNet dataset
    dataset = EmotionNet(image_size=image_size, metadata_path=metadata_path, transform=transform, mode=mode, fold=fold)
    
    # Initialize DataLoader
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=(mode == 'train' or mode == 'sample'),
                             num_workers=num_workers)
    return data_loader