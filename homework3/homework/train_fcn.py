import torch
from os import path
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tb
from torchvision import transforms
from .models import FCN, save_model  # Assuming your FCN model and save_model function are in a file named 'models.py'
from .utils import load_dense_data,ConfusionMatrix, DenseSuperTuxDataset
from .dense_transforms import Compose, ToTensor  # You should import your custom transformations here

# Define a function to calculate class weights based on class distribution
def calculate_class_weights(class_distribution):
    total_samples = sum(class_distribution)
    class_weights = [total_samples / (class_distribution[i] + 1e-6) for i in range(len(class_distribution))]
    
    # Normalize the class weights
    sum_weights = sum(class_weights)
    class_weights = [weight / sum_weights for weight in class_weights]
    
    return class_weights

def log(writer, step, loss, iou, accuracy):
    # Log loss, IoU, and accuracy
    writer.add_scalar('Loss', loss, step)
    writer.add_scalar('IoU', iou, step)
    writer.add_scalar('Accuracy', accuracy, step)

def train(args):
    # Initialize your FCN model
    model = FCN(in_channels=args.in_channels, out_channels=args.out_channels)

    # Create data loaders for training and validation sets
    train_data_path = 'dense_data/train'
    valid_data_path = 'dense_data/valid'
    train_loader, valid_loader = load_dense_data(train_data_path, valid_data_path, batch_size=args.batch_size, num_workers=0, transform=transforms)

    train_dataset = DenseSuperTuxDataset(dataset_path=train_data_path, transform=Compose([ToTensor()]),num_classes=args.num_classes)
    #train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset = DenseSuperTuxDataset(dataset_path=valid_data_path, transform=Compose([ToTensor()]),num_classes=args.num_classes)
    #valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize TensorBoard loggers
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Calculate class distribution for training dataset
    train_class_distribution = train_dataset.compute_class_distribution()

    # Define loss function (CrossEntropyLoss) and optimizer (e.g., Adam)
    criterion = torch.nn.CrossEntropyLoss()

    # Calculate class weights based on the class distribution
    class_weights = calculate_class_weights(train_class_distribution)

    # Use class weights in the loss function
    criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        model.train()  # Set the model to training mode
        for batch_idx, (imgs, lbls) in enumerate(train_loader):
            # Forward pass
            logits = model(imgs)

            # Calculate the loss
            loss = criterion(logits, lbls)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute IoU and accuracy using ConfusionMatrix
            confusion_matrix = ConfusionMatrix()
            confusion_matrix.add(logits.argmax(1), lbls)
            iou = confusion_matrix.iou()
            accuracy = confusion_matrix.accuracy()

            # Log performance metrics and visualize results using log function
            global_step = epoch * len(train_loader) + batch_idx
            log(train_logger, global_step, loss.item(), iou, accuracy)

            # Print progress
            print(f"Epoch [{epoch+1}/{args.num_epochs}] | "
                  f"Batch [{batch_idx+1}/{len(train_loader)}] | "
                  f"Loss: {loss.item():.4f} | "
                  f"IoU: {iou:.4f} | "
                  f"Accuracy: {accuracy:.4f}")

        # Validation loop (similar to training loop) to evaluate the model on the validation set
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for valid_batch_idx, (valid_imgs, valid_lbls) in enumerate(valid_loader):
                valid_logits = model(valid_imgs)

                # Compute IoU and accuracy for validation
                valid_confusion_matrix = ConfusionMatrix()
                valid_confusion_matrix.add(valid_logits.argmax(1), valid_lbls)
                valid_iou = valid_confusion_matrix.iou()
                valid_accuracy = valid_confusion_matrix.accuracy()

                # Log validation performance metrics and visualize results using log function
                valid_global_step = epoch * len(valid_loader) + valid_batch_idx
                log(valid_logger, valid_global_step, loss.item(), valid_iou, valid_accuracy)

                # Print validation progress
                print(f"Validation | "
                      f"Epoch [{epoch+1}/{args.num_epochs}] | "
                      f"Batch [{valid_batch_idx+1}/{len(valid_loader)}] | "
                      f"IoU: {valid_iou:.4f} | "
                      f"Accuracy: {valid_accuracy:.4f}")

        # Save the trained model at appropriate intervals
        if (epoch + 1) % args.save_interval == 0:
            save_model(model, epoch)  # Save the model with a unique identifier (e.g., epoch number)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('--num_epochs', type=int, default=100)  # Set the number of training epochs
    parser.add_argument('--batch_size', type=int, default=32)   # Set the batch size
    parser.add_argument('--learning_rate', type=float, default=0.001)  # Set the learning rate
    parser.add_argument('--save_interval', type=int, default=10)  # Save model every 'save_interval' epochs
    # Add other custom arguments here
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--out_channels', type=int, default=6)

    args = parser.parse_args()

    train(args)
