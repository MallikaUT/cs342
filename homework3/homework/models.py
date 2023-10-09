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

        # Define your convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Define your up-convolutional layers
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)

        # Define skip connections (residual connections)
        self.skip_conv1 = nn.Conv2d(256, 256, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(128, 128, kernel_size=1)
        self.skip_conv3 = nn.Conv2d(64, 64, kernel_size=1)

    def forward(self, x):
        # Forward pass through convolutional layers
        x1 = torch.relu(self.conv1(x))
        x2 = torch.relu(self.conv2(x1))
        x3 = torch.relu(self.conv3(x2))
        x4 = torch.relu(self.conv4(x3))
        x5 = torch.relu(self.conv5(x4))

        # Apply up-convolutions and skip connections
        x_up1 = torch.relu(self.upconv1(x5))
        x_up1 = torch.cat([x_up1, self.skip_conv1(x3)], dim=1)

        x_up2 = torch.relu(self.upconv2(x_up1))
        x_up2 = torch.cat([x_up2, self.skip_conv2(x2)], dim=1)

        x_up3 = torch.relu(self.upconv3(x_up2))
        x_up3 = torch.cat([x_up3, self.skip_conv3(x1)], dim=1)

        # Final up-convolution to produce segmentation output
        x_out = self.upconv4(x_up3)

        return x_out
        
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