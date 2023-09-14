from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

class SuperTuxDataset(Dataset):
    """
    WARNING: Do not perform data normalization here. 
    """
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use your solution (or the master solution) to HW1
        """
        self.dataset_path = dataset_path
        self.data = [] 
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
        image = Image.open(image_path).convert('RGB')  # Load image and convert to RGB
        label = int(label)  # Ensure label is an integer
        
        # Implement any necessary data transformations (e.g., resizing, normalization) here
        transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Resize the image to a fixed size
            transforms.ToTensor(),  # Convert image to a PyTorch tensor
            # Add more transformations as needed
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
