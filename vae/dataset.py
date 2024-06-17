import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
from typing import List, Tuple
from matplotlib import pyplot as plt

class LogoDataset(Dataset):
    def __init__(self, img_files: List[str], resize: Tuple[int, int], transform=None):
        self.img_files = img_files
        self.resize = resize
        self.transform = transform
        self.resize_transform = transforms.Resize(resize)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = read_image(img_path).float()/255
        image = self.resize_transform(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image

    def plot_img(self, idx):
        img_path = self.img_files[idx]
        image = read_image(img_path).float()/255
        image = self.resize_transform(image)
        
        if self.transform:
            image = self.transform(image)

        image = image.reshape(*self.resize, 3)

        plt.figure()
        plt.title(f'Image: {idx}')
        plt.imshow(image)
