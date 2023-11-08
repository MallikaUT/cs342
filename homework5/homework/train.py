from .planner import Planner, save_model
import torch
import torch.utils.tensorboard as tb
from .utils import load_data
import dense_transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import numpy as np

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
    train_data = load_data(args.dataset_path, transform=args.transform, num_workers=args.num_workers, batch_size=args.batch_size)

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        for i, data in enumerate(train_data):
            image, label = data
            image = image.to(args.device)
            label = label.to(args.device)

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
    parser.add_argument('--dataset_path', default='drive_data', type=str)
    parser.add_argument('--transform', default=dense_transforms.ToTensor(), type=eval)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train(args)
