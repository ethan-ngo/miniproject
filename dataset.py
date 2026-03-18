import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image

class LADIDataset(Dataset):
    """
    PyTorch Dataset wrapper for the Hugging Face LADI-v2 dataset.
    Downloads to the WSL root to avoid NAS I/O bottlenecks.
    """
    def __init__(self, split='train', img_size=224):
        super().__init__()
        
        wsl_cache_dir = os.path.expanduser('~/ladi_dataset_cache')
        
        # The damage classes in v2a we care about for the Stage 2 classifier
        self.label_cols = [
            'bridges_any',
            'buildings_any',
            'buildings_affected_or_greater',
            'buildings_minor_or_greater',
            'debris_any',
            'flooding_any',
            'flooding_structures',
            'roads_any',
            'roads_damage',
            'trees_any',
            'trees_damage',
            'water_any'
        ]
        
        csv_map = {
            'train': 'ladi_v2a_labels_train_resized.csv',
            'validation': 'ladi_v2a_labels_val_resized.csv',
            'test': 'ladi_v2a_labels_test_resized.csv'
        }
        
        csv_path = os.path.join(wsl_cache_dir, 'v2', csv_map[split])
        print(f"Loading LADI dataset (split: {split}) from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # Restored for full dataset run
        # self.df = self.df.head(100)
        self.wsl_cache_dir = wsl_cache_dir
        
        # Transforms for the VAE
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(), # converts PIL Image to tensor [0, 1]
        ])
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Get Image
        img_path = os.path.join(self.wsl_cache_dir, row['local_path'])
        
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        tensor_img = self.transform(img)
        
        # 2. Get Multi-Hot Labels
        labels = [float(row[col]) for col in self.label_cols]
        tensor_labels = torch.tensor(labels, dtype=torch.float32)
        
        return tensor_img, tensor_labels

if __name__ == "__main__":
    # Quick test to make sure it loads
    ds = LADIDataset('validation') # Use validation for a quick test
    print(f"Dataset size: {len(ds)}")
    img, label = ds[0]
    print(f"Image shape: {img.shape}, Label shape: {label.shape}")
