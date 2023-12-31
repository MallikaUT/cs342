import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

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
        return F.cross_entropy(input, target)


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        """
        Your code here
        Hint: Base this on yours or HW2 master solution if you'd like.
        Hint: Overall model can be similar to HW2, but you likely need some architecture changes (e.g. ResNets)
        """
        self.bn1 = nn.BatchNorm2d(3)
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        )
        
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        
        self.fc1 = nn.Linear(in_features=16 * 32 * 32, out_features=128)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=6)

        #raise NotImplementedError('CNNClassifier.__init__')
    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """
        x = self.bn1(x)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
  
        x = self.res1(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dropout1(x)  
        
        x = self.fc2(x)
        x = self.dropout2(x)  
        
        return x
        #raise NotImplementedError('CNNClassifier.forward')

class FCN(torch.nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        self.down1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.down2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.down3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )
        self.down4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )
        self.up4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )
        self.up3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.up2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.up1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 5, kernel_size=4, stride=2 ,padding=1),
            torch.nn.BatchNorm2d(5),
            torch.nn.ReLU()
        )
        self.d12 = torch.nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1 ,padding=1)
        self.d23 = torch.nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1 ,padding=1)
        self.d34 = torch.nn.ConvTranspose2d(128, 256, kernel_size=3, stride=1 ,padding=1)
        self.du44 = torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1 ,padding=1)
        self.u43 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1 ,padding=1)
        self.u32 = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1 ,padding=1)
        self.u21 = torch.nn.ConvTranspose2d(3, 5, kernel_size=3, stride=1 ,padding=1)
        
        #raise NotImplementedError('FCN.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        b, a, h, w = x.shape
        x_d1 = self.down1(x)         
    
        x_d2 = self.down2(x_d1)   
        x_d2 = x_d2 + self.d12(x_d1)

        x_d3 = self.down3(x_d2)
        x_d3 = x_d3 + self.d23(x_d2)
        x_d4 = self.down4(x_d3)
        x_do4 = x_d4 + self.d34(x_d3)
        x_u4 = self.up4(x_d4)
        x_u4 = x_u4 + self.du44(x_d4)
        x_wskip = torch.cat([x_u4, x_d3], dim=1)
        x_u3 = self.up3(x_wskip)
        x_u3 = x_u3 + self.u43(x_u4)
        x_wskip = torch.cat([x_u3, x_d2], dim=1)

        x_u2 = self.up2(x_wskip)       
        x_u2 = x_u2 + self.u32(x_u3)

        x_wskip = torch.cat([x_u2, x_d1], dim=1)  
        x_u1 = self.up1(x_wskip)
        x_u1 = x_u1 + self.u21(x)
        x_u1 = x_u1[:, :, :h, :w]
        return x_u1
        
        #raise NotImplementedError('FCN.forward')
        
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