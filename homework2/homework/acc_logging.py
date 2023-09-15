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
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)

            # Log training loss at every iteration
            train_logger.add_scalar('loss', dummy_train_loss, global_step=global_step)
            global_step += 1

            #dummy_train_accuracy = epoch/10. + torch.randn(10)
            dummy_train_accuracy = [0.85, 0.88, 0.90, 0.91, 0.86, 0.82, 0.89, 0.91, 0.87, 0.84]
       
       # avg_train_accuracy = dummy_train_accuracy.mean().item()
        avg_train_accuracy = sum(dummy_train_accuracy) / len(dummy_train_accuracy)
        # Log training accuracy after each epoch
        train_logger.add_scalar('accuracy', avg_train_accuracy, global_step=global_step)

        torch.manual_seed(epoch)
        for iteration in range(10):
            #dummy_validation_accuracy = epoch / 10. + torch.randn(10)
            dummy_validation_accuracy = [0.78, 0.81, 0.79, 0.83, 0.77, 0.80, 0.82, 0.79, 0.84, 0.81]  # Fixed values for validation accuracy

        #avg_validation_accuracy = dummy_validation_accuracy.mean().item()
        avg_validation_accuracy = sum(dummy_validation_accuracy) / len(dummy_validation_accuracy)

        # Log validation accuracy after each epoch
        valid_logger.add_scalar('accuracy', avg_validation_accuracy, global_step=global_step)

        if epoch == 0:
            # Log accuracy for epoch 0
            train_logger.add_scalar('accuracy_epoch0', avg_train_accuracy, global_step=epoch)
            valid_logger.add_scalar('accuracy_epoch0', avg_validation_accuracy, global_step=epoch)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)

