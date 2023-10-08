import torch
import torch.nn as nn

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
    def __init__(self, num_classes):
        super(FCN, self).__init__()

        # Encoder (downsampling path)
        self.encoder_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.encoder_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Max-pooling layers for downsampling
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder (upsampling path)
        self.decoder_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.decoder_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # Transposed convolution layers for upsampling
        self.trans_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.trans_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Final convolution layer for class prediction
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.encoder_conv1(x)
        x2 = self.max_pool(x1)
        x2 = self.encoder_conv2(x2)
        x3 = self.max_pool(x2)
        x3 = self.encoder_conv3(x3)

        # Decoder with skip connections
        x = self.trans_conv1(x3)
        x = torch.cat((x, x2), dim=1)  # Skip connection from encoder
        x = self.decoder_conv1(x)

        x = self.trans_conv2(x)
        x = torch.cat((x, x1), dim=1)  # Skip connection from encoder
        x = self.decoder_conv2(x)

        # Final convolution for class prediction
        x = self.final_conv(x)

        return x

def save_model(model, model_type):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), f'{model_type}.th'))

def load_model(model_type):
    from torch import load
    from os import path
    r = FCN(num_classes=6) if model_type == 'fcn' else CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), f'{model_type}.th'), map_location='cpu'))
    return r
