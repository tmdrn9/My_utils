import torch
from PIL import Image
import albumentations

class ClassficationDataset(torch.utils.data.Dataset):
    def __init__(self, x_dir, y_dir,train=True,transform=None):
        super().__init__()
        self.train=train
        self.transforms = transform
        self.x_img = x_dir
        self.y = y_dir

    def __len__(self):
        return len(self.x_img)

    def __getitem__(self, idx):
        x_img = self.x_img[idx]
        #albumentation 템플릿쓸때
        x_img = cv2.imread(x_img)
        x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
        ## torchvision.transforms 쓸 때
        # x_img = Image.open(x_img).convert('RGB')
        if self.transforms:
            augmented = self.transforms(image=x_img)
            x_img = augmented['image']
            
        if self.train:
            y = self.y[idx]
            return x_img, y
        
        else:
            return x_img
        
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, x_dir, y_dir, train=True, transform=None):
        super().__init__()
        self.train=train
        self.transforms = transform
        self.x_img = x_dir
        self.y_img = y_dir

    def __len__(self):
        return len(self.x_img)

    def __getitem__(self, idx):
        x_img = self.x_img[idx]
        #albumentation 템플릿쓸때
        x_img = cv2.imread(x_img)
        x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
        
        ## torchvision.transforms 쓸 때
        #x_img = Image.open(x_img).convert('RGB')

        if self.transforms:
            augmented = self.transforms(image=x_img, target_image=y_img)
            x_img = augmented['image']
            y_img = augmented['target_image']
        
        if self.train:
            y_img = self.y_img[idx]        
            #albumentation 템플릿쓸때
            y_img = cv2.imread(y_img)
            y_img = cv2.cvtColor(y_img, cv2.COLOR_BGR2RGB)
            ## torchvision.transforms 쓸 때
            #y_img = Image.open(y_img).convert('RGB')
            return x_img, y_img
        
        else:
            return x_img
