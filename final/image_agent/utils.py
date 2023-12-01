from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torchvision
from . import dense_transforms


class DetectionSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=torchvision.transforms.ToTensor(), min_size=20):
        from glob import glob
        import os
        from os import path
        self.files = []
        self.masks = []
        print(dataset_path)
        for im_f in glob(path.join(dataset_path, 'images', '*')):
            self.files.append(im_f)

        for im_f in glob(path.join(dataset_path, 'masks', '*')):
            self.masks.append(im_f)
        self.transform = transform
        self.min_size = min_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import numpy as np
        b = self.files[idx]
        c = self.masks[idx]
        im = Image.open(b)
        ma = Image.open(c)
        data = im, ma
        if self.transform is not None:
            data = self.transform(data[0]), self.transform(data[1])
        return data


def load_detection_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DetectionSuperTuxDataset('/content/drive/MyDrive/Colab Notebooks/dense_data/data', **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


if __name__ == '__main__':
    dataset = DetectionSuperTuxDataset('/content/drive/MyDrive/Colab Notebooks/dense_data/data')
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
    dataset = DetectionSuperTuxDataset('/content/drive/MyDrive/Colab Notebooks/dense_data/data',
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