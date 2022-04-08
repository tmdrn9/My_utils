import torch
from PIL import Image
import albumentations

class ClassficationDataset(torch.utils.data.Dataset):
    def __init__(self, x_dir, y_dir, transform=None):
        super().__init__()
        self.transforms = transform
        self.x_img = x_dir
        self.y = y_dir

    def __len__(self):
        return len(self.x_img)

    def __getitem__(self, idx):
        x_img = self.x_img[idx]
        y = self.y[idx]

        x_img = Image.open(x_img).convert('RGB')

        if self.transforms:
            x_img = self.transforms(x_img)

        return x_img, y

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, x_dir, y_dir, transform=None):
        super().__init__()
        self.transforms = transform
        self.x_img = x_dir
        self.y_img = y_dir

    def __len__(self):
        return len(self.x_img)

    def __getitem__(self, idx):
        x_img = self.x_img[idx]
        y_img = self.y_img[idx]

        x_img = Image.open(x_img).convert('RGB')
        y_img = Image.open(y_img).convert('RGB')

        if self.transforms:
            augmented = self.transforms(image=x_img, target_image=y_img)
            x_img = augmented['image']
            y_img = augmented['target_image']

        return x_img, y_img