import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class GhibliDataset(Dataset):
    def __init__(self, root_dir, mode='training', transform=None):
        self.root_dir = os.path.join(root_dir, mode)
        self.mode = mode
        self.transform = transform if transform else transforms.ToTensor()
        self.subfolders = sorted(os.listdir(self.root_dir))
        
        self.pairs = []
        for subfolder in self.subfolders:
            original_path = os.path.join(self.root_dir, subfolder, 'o.png')
            ghibli_path = os.path.join(self.root_dir, subfolder, 'g.png')
            if os.path.exists(original_path) and os.path.exists(ghibli_path):
                self.pairs.append((original_path, ghibli_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        original_path, ghibli_path = self.pairs[idx]
        
        original_image = Image.open(original_path).convert('RGB')
        ghibli_image = Image.open(ghibli_path).convert('RGB')

        original_image = self.transform(original_image)
        ghibli_image = self.transform(ghibli_image)

        return original_image, ghibli_image


# Define the transformations for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to desired input size (change as needed)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images
])

# Creating DataLoaders for training, testing, and validation
def create_dataloaders(root_dir, batch_size=16):
    train_dataset = GhibliDataset(root_dir=root_dir, mode='training', transform=transform)
    val_dataset = GhibliDataset(root_dir=root_dir, mode='validation', transform=transform)
    test_dataset = GhibliDataset(root_dir=root_dir, mode='testing', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


# Example usage
root_dir = '/path/to/dataset'  # Set the path to the dataset root folder

train_loader, val_loader, test_loader = create_dataloaders(root_dir)

# You can now use these loaders in your training loop
for original_images, ghibli_images in train_loader:
    print(original_images.shape, ghibli_images.shape)
    break  # Example to print one batch
