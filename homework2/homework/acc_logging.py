from os import path
import torch
import torch.utils.tensorboard as tb


def test_logging(train_logger, valid_logger):
  
    """
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epoch
    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    """
    global_step = 0
    for epoch in range(10):
        torch.manual_seed(epoch)
        accuracy_list = []  # Create list to store individual accuracy values
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)

            # Loging training loss 
            train_logger.add_scalar('loss', dummy_train_loss, global_step=global_step)
            global_step += 1

            dummy_train_accuracy = epoch/10. + torch.randn(10)
            accuracy_list.append(dummy_train_accuracy.mean().item())  # Adding Value into the  List

        avg_train_accuracy = sum(accuracy_list) / len(accuracy_list)  

        # Logging training accuracy
        train_logger.add_scalar('accuracy', avg_train_accuracy, global_step=global_step)

        torch.manual_seed(epoch)
        accuracy_list = []  # Reset the list for validation
        for iteration in range(10):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)
            accuracy_list.append(dummy_validation_accuracy.mean().item())  

        avg_validation_accuracy = sum(accuracy_list) / len(accuracy_list)  

        # Loging validation accuracy 
        valid_logger.add_scalar('accuracy', avg_validation_accuracy, global_step=global_step)

        # Logging accuracy for epoch 0
        if epoch == 0:
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

