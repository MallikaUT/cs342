import torch
import torch.nn.functional as F
import torch.nn as nn


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """
        return -torch.mean(torch.log(F.softmax(input, dim=1)[range(input.size(0)), target]))
        loss = F.cross_entropy(input, target)
        return loss
        raise NotImplementedError('ClassificationLoss.forward')



class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        super().__init__()
        self.fc = torch.nn.Linear(3 * 64 * 64, 6)  # Input size: 3 * 64 * 64, Output size: 6

        #raise NotImplementedError('LinearClassifier.__init__')

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.fc(x)
        raise NotImplementedError('LinearClassifier.forward')


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        super(MLPClassifier, self).__init__()
        #super().__init__()
        self.fc1 = torch.nn.Linear(3 * 64 * 64, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 6)  # Output size: 6
        
        # Define the layers of the MLP
        self.fc1 = nn.Linear(3 * 64 * 64, 128)  # First hidden layer (input size to 128)
        self.relu1 = nn.ReLU()  # ReLU activation for the first hidden layer
        self.fc2 = nn.Linear(128, 6)  # Output layer (128 to 6, one for each class)

        
        
        #raise NotImplementedError('MLPClassifier.__init__')

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        return x
        raise NotImplementedError('MLPClassifier.forward')


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
