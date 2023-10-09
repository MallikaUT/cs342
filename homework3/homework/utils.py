import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
import csv
import os
import numpy as np



from . import dense_transforms

#from  . import dense_transforms
#from utils import DenseSuperTuxDataset, load_dense_data, ConfusionMatrix, save_model

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
DENSE_LABEL_NAMES = ['background', 'kart', 'track', 'bomb/projectile', 'pickup/nitro']
# Distribution of classes on dense training set (background and track dominate (96%)
DENSE_CLASS_DISTRIBUTION = [0.52683655, 0.02929112, 0.4352989, 0.0044619, 0.00411153]


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path,transform=None):
        """
        Your code here
        Hint: Use your solution (or the master solution) to HW1 / HW2
        Hint: If you're loading (and storing) PIL images here, make sure to call image.load(),
              to avoid an OS error for too many open files.
        Hint: Do not store torch.Tensor's as data here, but use PIL images, torchvision.transforms expects PIL images
              for most transformations.
        """
        import csv
        from os import path
        self.data = []
        to_tensor = transforms.ToTensor()
        self.transform = transform  # Add the transform argument
        with open(path.join(dataset_path, 'labels.csv'), newline='') as f:
            reader = csv.reader(f)
            for fname, label, _ in reader:
                if label in LABEL_NAMES:
                    image = Image.open(path.join(dataset_path, fname))
                    label_id = LABEL_NAMES.index(label)
                    self.data.append((to_tensor(image), label_id))

    def __len__(self):
        """
        Your code here
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Your code here
        """
        return self.data[idx]

class DenseSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=None, num_classes=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.num_classes = num_classes
        self.samples = []  # List to store (image, label) pairs

        # Populate self.samples with (image, label) pairs
        self._load_samples()

        # Calculate the class distribution
        self.class_distribution = self.compute_class_distribution()

    def _load_samples(self):
        # Get a list of all image files in the dataset directory
        dataset_dir = self.dataset_path
        image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg') or f.endswith('.png')]

        # Create a list of (image, label) pairs
        self.samples = [(os.path.join(dataset_dir, img), os.path.join(dataset_dir, img)) for img in image_files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label_path = self.samples[index]
        image = Image.open(image_path)
        label = Image.open(label_path)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

    def compute_class_distribution(self):
        class_distribution = [0] * self.num_classes

        for _, label_path in self.samples:
            label = Image.open(label_path)
            label = label.convert('L')  # Convert to grayscale image
            label = np.array(label)  # Convert to a NumPy array
            unique_classes, class_counts = np.unique(label, return_counts=True)

            for cls, count in zip(unique_classes, class_counts):
                  if cls < self.num_classes:
                       class_distribution[cls] += count

        return class_distribution




def load_data(dataset_path, num_workers=0, batch_size=128, **kwargs):
    dataset = SuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def load_dense_data(train_dataset_path, valid_dataset_path, batch_size=32, num_workers=0, transform=None):
    train_dataset = DenseSuperTuxDataset(train_dataset_path, transform=transform)
    valid_dataset = DenseSuperTuxDataset(valid_dataset_path, transform=transform)

    # Ensure that batch_size is not greater than the dataset size
    if batch_size > len(train_dataset):
        batch_size = len(train_dataset)
    
    if batch_size > len(valid_dataset):
        batch_size = len(valid_dataset)

    train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, valid_loader

def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()

def _one_hot(x, n):
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()


class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0).detach()

    def __init__(self, size=5):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size)
        self.size = size

    def add(self, preds, labels):
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        """
        self.matrix = self.matrix.to(preds.device)
        self.matrix += self._make(preds, labels).float()

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)

    @property
    def iou(self):
        return self.class_iou.mean()

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)


if __name__ == '__main__':
    train_dataset_path = 'dense_data/train'
    valid_dataset_path = 'dense_data/valid'
    
    # Load training and validation data loaders
    train_loader, valid_loader = load_dense_data(train_dataset_path, valid_dataset_path, batch_size=32, num_workers=0)

    from pylab import show, imshow, subplot, axis

    for i in range(15):
        im, lbl = train_loader.dataset[i]  # Access the dataset from the loader
        subplot(5, 6, 2 * i + 1)
        imshow(F.to_pil_image(im))
        axis('off')
        subplot(5, 6, 2 * i + 2)
        imshow(dense_transforms.label_to_pil_image(lbl))
        axis('off')
    show()

    import numpy as np

    c = np.zeros(5)
    for im, lbl in train_loader.dataset:  # Access the dataset from the loader
        c += np.bincount(lbl.view(-1), minlength=len(DENSE_LABEL_NAMES))
    print(100 * c / np.sum(c))
