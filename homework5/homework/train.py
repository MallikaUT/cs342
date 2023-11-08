from .planner import Planner, save_model
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms
import matplotlib.pyplot as plt  # Add this import for visualization
import torchvision.transforms.functional as TF


def train(args):
    from os import path
    model = Planner()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    # Define your loss function and optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Load your dataset using load_data function
    train_data = load_data('drive_data', transform=dense_transforms, num_workers=4, batch_size=args.batch_size)

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        for i, (image, label) in enumerate(train_data):
            # Forward pass
            output = model(image)

            # Compute the loss
            loss = loss_fn(output, label)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log the visualization
            if train_logger is not None:
                log(train_logger, image, label, output, epoch * len(train_data) + i)

        # Save the model after each epoch
        save_model(model)

    # Optionally, you can also log other training statistics using train_logger


def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)]) / 2
    ax.add_artist(plt.Circle(WH2 * (label[0].cpu().detach().numpy() + 1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2 * (pred[0].cpu().detach().numpy() + 1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('--num_epochs', type=int, default=20)  # Add custom arguments
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)  # Add batch size argument

    args = parser.parse_args()
    train(args)
