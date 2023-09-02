from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data


def train(args):
    model = model_factory[args.model]()

    """
    Your code here

    """
    criterion = ClassificationLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # Load the training and validation data
    train_loader = load_data(args.train_data, batch_size=args.batch_size)
    valid_loader = load_data(args.valid_data, batch_size=args.batch_size)

    # Training loop
    for epoch in range(args.epochs):
        model.train()  # Set the model in training mode
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()

        # Print training loss for this epoch
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {running_loss / len(train_loader)}")

    # Validation loop
    model.eval()  # Set the model in evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print validation accuracy
    print(f"Validation Accuracy: {100 * correct / total}%")
    raise NotImplementedError('train')

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    parser.add_argument('--lr', type=float, default=0.01)  # Learning rate
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_data', default='data/train')  # Path to training data
    parser.add_argument('--valid_data', default='data/valid')  # Path to validation data


    args = parser.parse_args()
    train(args)
