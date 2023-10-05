"""from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    
    #Your code here, modify your HW1 / HW2 code
    

    # Define your dataset and dataloaders
    train_loader = load_data(args.train_data, batch_size=args.batch_size)
    valid_loader = load_data(args.valid_data, batch_size=args.batch_size)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)

        # Log the training loss
        if train_logger:
            train_logger.add_scalar('train/loss', avg_loss, epoch)

        print(f'Epoch [{epoch + 1}/{args.epochs}] - Avg. Loss: {avg_loss:.4f}')

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_data', default='data/train')
    parser.add_argument('--valid_data', default='data/valid')

    args = parser.parse_args()
    train(args)"""


import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
from os import path
import torch.utils.tensorboard as tb


def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize the model
    model = CNNClassifier().to(device)

    # Set up TensorBoard loggers
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    # Define data transformations and data loaders
    train_transforms = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.ToTensor(),
    ])

    train_loader = load_data(args.train_data, batch_size=args.batch_size, transform=train_transforms)
    valid_loader = load_data(args.valid_data, batch_size=args.batch_size)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step, gamma=args.lr_scheduler_gamma)

    # Early stopping variables
    best_valid_accuracy = 0.0
    no_improvement_count = 0

for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0

    for batch_data, batch_labels in train_loader:
        # Unpack the tuple
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)  # Move data and labels to the device

        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print average loss for this epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{args.epochs}] - Avg. Loss: {avg_loss:.4f}")

        # Learning rate scheduling step
        scheduler.step()

        # Validation loop
        model.eval()
        valid_acc_vals = []
        for batch_data, batch_labels in valid_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            valid_outputs = model(batch_data)
            valid_accuracy = accuracy(valid_outputs, batch_labels).detach().cpu().numpy()
            valid_acc_vals.append(valid_accuracy)

        avg_valid_accuracy = sum(valid_acc_vals) / len(valid_acc_vals)

        if valid_logger:
            valid_logger.add_scalar('valid/accuracy', avg_valid_accuracy, epoch)

        print(f'Validation Accuracy: {avg_valid_accuracy:.4f}')

        # Early stopping check
        if avg_valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = avg_valid_accuracy
            no_improvement_count = 0
            # Save the best model
            save_model(model)
        else:
            no_improvement_count += 1

        # Early stopping condition
        if no_improvement_count >= args.early_stopping_patience:
            print("No improvement in validation accuracy. Early stopping.")
            break

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train_data', default='data/train')
    parser.add_argument('--valid_data', default='data/valid')
    parser.add_argument('--weight_decay', type=float, default=0.001)  # L2 weight regularization
    parser.add_argument('--lr_scheduler_step', type=int, default=10)  # Learning rate scheduler step size
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.5)  # Learning rate scheduler gamma
    parser.add_argument('--early_stopping_patience', type=int, default=10)

    args = parser.parse_args()
    train(args)




