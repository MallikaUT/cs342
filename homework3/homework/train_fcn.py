import torch
import numpy as np
import os

from models import FCN, save_model  # Import your FCN model
from utils import load_dense_data, ConfusionMatrix
from dense_transforms import Compose, RandomHorizontalFlip, RandomCrop, Normalize, ToTensor
import torch.utils.tensorboard as tb

def train(args):
    # Create the output directory for model checkpoints
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    model = FCN(num_classes=args.num_classes)  # Adjust num_classes based on your task
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(os.path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(os.path.join(args.log_dir, 'valid'), flush_secs=1)

    # Define transformations for data augmentation and preprocessing
    transform = Compose([
        RandomHorizontalFlip(flip_prob=0.5),
        RandomCrop(size=(args.image_size, args.image_size)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensor()
    ])

    # Load the train and validation datasets using load_dense_data function
    train_loader, valid_loader = load_dense_data(os.path.join(args.train_data, 'train'),  # Corrected path
                                                 batch_size=args.batch_size,
                                                 num_workers=args.num_workers,
                                                 transform=transform)

    # Define loss function (e.g., CrossEntropyLoss) and optimizer (e.g., Adam)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
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
        confusion_matrix = ConfusionMatrix(num_classes=args.num_classes)
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(valid_loader):
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                confusion_matrix.add(logits.argmax(1), labels)

        iou = confusion_matrix.iou()
        print(f"Validation IoU: {iou:.4f}")

        # Save model checkpoints
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}.pt')
            save_model(model, checkpoint_path)

    # Save the final trained model
    final_checkpoint_path = os.path.join(args.checkpoint_dir, 'final_model.pt')
    save_model(model, final_checkpoint_path)

def log(logger, imgs, lbls, logits, global_step):
    # Your existing log function
    pass

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='logs')  # Specify the log directory
    parser.add_argument('--train_data', default='dense_data/data')  # Adjust the default path
    parser.add_argument('--checkpoint_dir', default='checkpoints')  # Specify the checkpoint directory
    parser.add_argument('--batch_size', type=int, default=8)  # Adjust batch size
    parser.add_argument('--num_workers', type=int, default=4)  # Adjust number of workers
    parser.add_argument('--num_epochs', type=int, default=10)  # Adjust the number of epochs
    parser.add_argument('--lr', type=float, default=0.001)  # Adjust learning rate
    parser.add_argument('--image_size', type=int, default=256)  # Specify the image size
    parser.add_argument('--num_classes', type=int, default=21)  # Specify the number of classes
    parser.add_argument('--save_interval', type=int, default=1)  # Specify how often to save checkpoints

    args = parser.parse_args()
    train(args)
