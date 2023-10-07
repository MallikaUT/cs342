import torch
import numpy as np

from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb

def train(args):
    from os import path
    model = FCN(num_classes=21)  # Adjust num_classes based on your task
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Load the train and validation datasets using load_dense_data function
    train_loader, valid_loader = load_dense_data(args.train_data, batch_size=args.batch_size, num_workers=int(args.num_workers))

    # Define loss function (e.g., CrossEntropyLoss) and optimizer (e.g., Adam)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()  # Zero the gradient buffers
            logits = model(images)  # Forward pass

            # Calculate the loss
            loss = criterion(logits, labels)
            total_loss += loss.item()

            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # Logging
            if train_logger is not None:
                log(train_logger, images, labels, logits, epoch * len(train_loader) + batch_idx)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}] - Loss: {avg_loss:.4f}")

        # Validation
        model.eval()  # Set the model to evaluation mode
        confusion_matrix = ConfusionMatrix(num_classes=21)  # Adjust num_classes
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(valid_loader):
                logits = model(images)
                confusion_matrix.add(logits.argmax(1), labels)

        iou = confusion_matrix.iou()
        print(f"Validation IoU: {iou:.4f}")

    # Save the trained model
    save_model(model)

def log(logger, imgs, lbls, logits, global_step):
    # Your existing log function
    pass

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('--train_data', default='data/train')  # Adjust the default path
    parser.add_argument('--batch_size', type=int, default=8)  # Adjust batch size
    parser.add_argument('--num_workers', type=int, default=4)  # Adjust number of workers
    parser.add_argument('--num_epochs', type=int, default=10)  # Adjust the number of epochs
    parser.add_argument('--lr', type=float, default=0.001)  # Adjust learning rate

    args = parser.parse_args()
    train(args)
