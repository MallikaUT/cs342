import numpy as np
import pystk
import torch

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from . import dense_transforms
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15

DATASET_PATH = '/content/drive/MyDrive/Colab Notebooks/dense_data/data'               
#DATASET_PATH = '/content/cs342/final/data_instance'     #render_data instance path

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):
        from PIL import Image
        from glob import glob
        from os import path
        self.data = []
        for f in glob(path.join(dataset_path, '*.csv')):
            i = Image.open(f.replace('.csv', '.png'))
            i.load()
            self.data.append((i, np.loadtxt(f, dtype=np.float32, delimiter=',')))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(*data)
        return data

def collate_tensor_fn(batch):
    # Find the maximum height and width in the batch
    max_height = max(img.shape[1] for img in batch)
    max_width = max(img.shape[2] for img in batch)

    # Pad each image to the maximum height and width
    padded_batch = [torch.nn.functional.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1])) for img in batch]

    # Stack the padded tensors
    stacked_batch = torch.stack(padded_batch, dim=0)

    return stacked_batch

def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, collate_fn=collate_tensor_fn, batch_size=batch_size, shuffle=True, drop_last=True)
