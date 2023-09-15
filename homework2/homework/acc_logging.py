"""from os import path
import torch
import torch.utils.tensorboard as tb


def test_logging(train_logger, valid_logger):

   
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epoch
    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    
    global_step = 0 
    for epoch in range(10):
        torch.manual_seed(epoch)
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)

            train_logger.add_scalar('loss', dummy_train_loss, global_step=global_step)
            global_step += 1

            dummy_train_accuracy = epoch/10. + torch.randn(10)

        avg_train_accuracy = dummy_train_accuracy.mean().item()
        train_logger.add_scalar('accuracy', avg_train_accuracy, global_step=epoch)
       
       #raise NotImplementedError('Log the training loss')
       # raise NotImplementedError('Log the training accuracy')

        torch.manual_seed(epoch)
        for iteration in range(10):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)

        avg_validation_accuracy = dummy_validation_accuracy.mean().item()
        valid_logger.add_scalar('accuracy', avg_validation_accuracy, global_step=epoch) 

       #raise NotImplementedError('Log the validation accuracy')


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)
"""

from os import path
import torch
import torch.utils.tensorboard as tb

def test_logging(train_logger, valid_logger):
    global_step = 0
    for epoch in range(10):
        torch.manual_seed(epoch)
        train_accuracies = []

        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            train_logger.add_scalar('loss', dummy_train_loss, global_step=global_step)
            global_step += 1

            dummy_train_accuracy = epoch/10. + torch.randn(10)
            train_accuracies.extend(dummy_train_accuracy.tolist())

        avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)
        train_logger.add_scalar('accuracy', avg_train_accuracy, global_step=epoch)

        torch.manual_seed(epoch)
        valid_accuracies = []

        for iteration in range(10):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)
            valid_accuracies.extend(dummy_validation_accuracy.tolist())

        avg_validation_accuracy = sum(valid_accuracies) / len(valid_accuracies)
        valid_logger.add_scalar('accuracy', avg_validation_accuracy, global_step=epoch)

    # Log accuracy for epoch 0 after the loop
    epoch_0_train_accuracies = -0.034079  # Replace with actual values
    epoch_0_train_avg_accuracy = epoch_0_train_accuracies.mean().item()
    train_logger.add_scalar('accuracy_epoch0', epoch_0_train_avg_accuracy, global_step=0)

    epoch_0_valid_accuracies = -0.055433  # Replace with actual values
    epoch_0_valid_avg_accuracy = epoch_0_valid_accuracies.mean().item()
    valid_logger.add_scalar('accuracy_epoch0', epoch_0_valid_avg_accuracy, global_step=0)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)

