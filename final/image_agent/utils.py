from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms 
from . import dense_transforms  # Replace with your actual module
import pandas as pd
import os
from glob import glob


class DetectionSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.data_files = sorted(glob(os.path.join(dataset_path, '*.png')))
        self.transform = transform
        print(dataset_path)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        image_path = self.data_files[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Load corresponding CSV file if available
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        csv_path = os.path.join('/content/dense_data/data', 'data', f'{base_name}.csv')
        if csv_path in self.csv_files:
            csv_data = pd.read_csv(csv_path)  # Adjust the read_csv parameters as needed
        else:
            csv_data = None

        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)

        return image, csv_data if csv_data is not None else 0

def load_detection_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust the size as needed
        transforms.ToTensor(),
    ])

    dataset = DetectionSuperTuxDataset(dataset_path, transform=transform, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

if __name__ == '__main__':
    dataset_path = '/content/dense_data/data'
    dataset = DetectionSuperTuxDataset(dataset_path)
    import torchvision.transforms.functional as F
    from pylab import show, subplots
    import matplotlib.patches as patches
    import numpy as np

    fig, axs = subplots(1, 2)
    for i, ax in enumerate(axs.flat):
        im, ma = dataset[100+i]
        ax.imshow(F.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], fc='none', ec='r', lw=2))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], fc='none', ec='g', lw=2))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], fc='none', ec='b', lw=2))
        ax.axis('off')
    dataset = DetectionSuperTuxDataset(dataset_path,
                                       transform=dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(0),
                                                                           dense_transforms.ToTensor()]))
    fig.tight_layout()
    # fig.savefig('box.png', bbox_inches='tight', pad_inches=0, transparent=True)

    fig, axs = subplots(1, 2)
    for i, ax in enumerate(axs.flat):

        im, *dets = dataset[100+i]
        hm, size = dense_transforms.detections_to_heatmap(dets, im.shape[1:])
        ax.imshow(F.to_pil_image(im), interpolation=None)
        hm = hm.numpy().transpose([1, 2, 0])
        alpha = 0.25*hm.max(axis=2) + 0.75
        r = 1 - np.maximum(hm[:, :, 1], hm[:, :, 2])
        g = 1 - np.maximum(hm[:, :, 0], hm[:, :, 2])
        b = 1 - np.maximum(hm[:, :, 0], hm[:, :, 1])
        ax.imshow(np.stack((r, g, b, alpha), axis=2), interpolation=None)
        ax.axis('off')
    fig.tight_layout()
    # fig.savefig('heat.png', bbox_inches='tight', pad_inches=0, transparent=True)

    show()
