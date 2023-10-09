import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        
        # Input normalization
        self.bn1 = nn.BatchNorm2d(3)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Residual blocks (add more if needed)
        self.res1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        )
        
        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16 * 32 * 32, out_features=128)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=6)

    def forward(self, x):
        x = self.bn1(x)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Apply residual block(s)
        x = self.res1(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dropout1(x)  # Dropout layer 1
        
        x = self.fc2(x)
        x = self.dropout2(x)  # Dropout layer 2
        
        return x

class FCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCN, self).__init__()
        
        # Initial convolution layers
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Intermediate convolution layers
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Deconvolution layers (Up-sampling)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        # Skip connection from conv2
        self.skip1 = nn.Conv2d(64, 64, kernel_size=1)
        
        self.upconv2 = nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        
    def forward(self, x):
        # Encoder
        x1 = self.relu2(self.conv2(self.relu1(self.conv1(x))))
        x2 = self.pool1(x1)
        
        x3 = self.relu4(self.conv4(self.relu3(self.conv3(x2))))
        x4 = self.pool2(x3)
        
        # Decoder with skip connections
        x4_up = self.upconv1(x4)
        x4_up = self.skip1(x4_up)
        
        x3_up = x4_up + x3
        x3_up = self.upconv2(x3_up)
        
        return x3_up
        model = FCN(in_channels=3, out_channels=num_classes)

model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
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