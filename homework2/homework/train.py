from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb
import torch.optim as optim


def train(args):
    from os import path
    model = CNNClassifier[args.model]()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train_loader = load_data(args.train_data, batch_size=args.batch_size)
    valid_loader = load_data(args.valid_data, batch_size=args.batch_size)
   

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    """
    Your code here, modify your HW1 code
    
    """

    # Training loop
    num_epochs = 10
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        total_correct = 0
        total_samples = 0

        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_data)

            # Compute loss
            loss = criterion(outputs, batch_labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Compute training accuracy
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

            # Log training loss
            train_logger.add_scalar('train/loss', loss.item(), global_step=global_step)
            global_step += 1

        # Calculate training accuracy and log
        train_accuracy = total_correct / total_samples
        train_logger.add_scalar('train/accuracy', train_accuracy, global_step=epoch)

        # Validation loop
        model.eval()
        total_correct = 0
        total_samples = 0

        for batch_data, batch_labels in valid_loader:
            # Forward pass
            outputs = model(batch_data)

            # Compute validation accuracy
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

        # Calculate validation accuracy and log
        valid_accuracy = total_correct / total_samples
        valid_logger.add_scalar('valid/accuracy', valid_accuracy, global_step=epoch)


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
    train(args)
