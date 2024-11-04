from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import torchvision.transforms as transforms

class OnionDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])
        self.image_files = [f for f in os.listdir(root) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        original_image = image.copy()
        if self.transform:
            image = self.transform(image)
        return image, original_image, self.image_files[idx]

    @staticmethod
    def collate_fn(batch):
        images, original_images, image_files = zip(*batch)
        return torch.stack(images, 0), original_images, image_files 