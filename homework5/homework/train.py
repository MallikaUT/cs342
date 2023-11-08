from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
from torch.utils.data import DataLoader
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path
    # Initialize your model
    model = Planner()

    # Create data loaders for training and validation
    # Modify the dataset and batch size according to your data
    train_dataset = load_data('drive_data', transform=dense_transforms.YourTransformClass(), num_workers=args.num_workers)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Create the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.L1Loss()

    # Set up TensorBoard logging
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    # Move the model to the GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Log the loss
        avg_loss = total_loss / len(train_loader)
        train_logger.add_scalar('Loss', avg_loss, epoch)

    # Save the trained model
    save_model(model)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, help='Directory for TensorBoard logs')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    args = parser.parse_args()
    train(args)
