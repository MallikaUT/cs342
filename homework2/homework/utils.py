from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv
import torch
import os

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

class SuperTuxDataset(Dataset):
    """
    WARNING: Do not perform data normalization here. 
    """
    def __init__(self, dataset_path,transform=None):
        """
        Your code here
        Hint: Use your solution (or the master solution) to HW1
        """
        self.dataset_path = dataset_path
        self.data = [] 

        label_map = {
            'background': 0,
            'kart': 1,
            'pickup': 2,
            'nitro': 3,
            'bomb': 4,
            'projectile': 5
         }
 
        with open(os.path.join(dataset_path, 'labels.csv'), 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  
            for row in csv_reader:
                image_path = os.path.join(dataset_path, row[0])
                label = label_map.get(row[1], -1) 
                if label != -1:
                    self.data.append((image_path, label))

        self.transform = transform
        #raise NotImplementedError('SuperTuxDataset.__init__')

    def __len__(self):
        """
        Your code here
        """
        return len(self.data)
        #raise NotImplementedError('SuperTuxDataset.__len__')

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert('RGB')  
        label = int(label)  
              
        transform = transforms.Compose([
            transforms.Resize((64, 64)),  
            transforms.ToTensor(),             
        ])
        image = transform(image)

        return image, label

        #raise NotImplementedError('SuperTuxDataset.__getitem__')


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
