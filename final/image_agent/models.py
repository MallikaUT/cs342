import torch
import torch.nn.functional as F
from os import path

def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    max_cls = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks // 2, stride=1)[0, 0]
    possible_det = heatmap - (max_cls > heatmap).float() * 1e5
    print("Shape of possible_det:", possible_det.shape)
    if max_det > possible_det.numel():
        max_det = possible_det.numel()
    score, loc = torch.topk(possible_det.view(-1), max_det)
    return [(float(s), int(l) % heatmap.size(1), int(l) // heatmap.size(1))
            for s, l in zip(score.cpu(), loc.cpu()) if s > min_score]
            
class Detector(torch.nn.Module):

    class BlockConv(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=1, residual: bool = True):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=(kernel_size // 2), stride=1,
                                bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=(kernel_size // 2), stride=stride,
                                bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=(kernel_size // 2), stride=1,
                                bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )
            self.residual = residual
            self.downsample = None
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(
                    torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride, bias=False),
                    torch.nn.BatchNorm2d(n_output)
                )

        def forward(self, x):
            if self.residual:
                identity = x if self.downsample is None else self.downsample(x)
                return self.net(x) + identity
            else:
                return self.net(x)


    class BlockUpConv(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1, residual: bool = True):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=3, padding=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(n_output, n_output, kernel_size=3, padding=1, stride=stride, output_padding=0 if stride == 1 else 1,
                                        bias=False),  # Adjust output_padding
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(n_output, n_output, kernel_size=3, padding=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )
            self.residual = residual
            self.upsample = None
            if stride != 1 or n_input != n_output:
                self.upsample = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=1, stride=stride, output_padding=0 if stride == 1 else 1,
                                            bias=False),  # Adjust output_padding
                    torch.nn.BatchNorm2d(n_output)
                )

        def forward(self, x):
            if self.residual:
                identity = x if self.upsample is None else self.upsample(x)
                return self.net(x) + identity
            else:
                return self.net(x)

    def __init__(self, dim_layers=[32, 64, 128], c_in=3, c_out=2, input_normalization: bool = True,
                 skip_connections: bool = True, residual: bool = False):
        super().__init__()

        self.skip_connections = skip_connections

        c = dim_layers[0]
        self.net_conv = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv2d(c_in, c, kernel_size=7, padding=3, stride=2, bias=False),
            torch.nn.BatchNorm2d(c),
            torch.nn.ReLU()
        )])
        self.net_upconv = torch.nn.ModuleList([
            torch.nn.ConvTranspose2d(c * 2 if skip_connections else c, c_out, kernel_size=7,
                                     padding=3, stride=2, output_padding=1)
        ])
        for k in range(len(dim_layers)):
            l = dim_layers[k]
            self.net_conv.append(self.BlockConv(c, l, stride=2, residual=residual))
            l = l * 2 if skip_connections and k != len(dim_layers) - 1 else l
            self.net_upconv.insert(0, self.BlockUpConv(l, c, stride=2, residual=residual))
            c = dim_layers[k]

        if input_normalization:
            self.norm = torch.nn.BatchNorm2d(c_in)
        else:
            self.norm = None

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)

        h = x.shape[2]
        w = x.shape[3]

        skip_con = []
        for i, layers in enumerate(self.net_conv):
            x = layers(x)
            print(f"After net_conv layer {i}:", x.shape)  # Print sizes after net_conv layers
            skip_con.append(x)
        skip_con.pop(-1)
        skip = False
        for i, layers in enumerate(self.net_upconv):
            if skip and len(skip_con) > 0:
                print(f"Size of x before concatenation in net_upconv layer {i}:", x.shape)
                print(f"Size of tensor from skip_con in net_upconv layer {i}:", skip_con[-1].shape)

                if x.size(2) != skip_con[-1].size(2) or x.size(3) != skip_con[-1].size(3):
                    raise ValueError(f"Sizes of tensors must match except in dimension 2. Got {x.size(2)} and {skip_con[-1].size(2)}")

                x = torch.cat([x, skip_con.pop(-1)], 1)
                print(f"Size of x after concatenation in net_upconv layer {i}:", x.shape)
                x = layers(x)
            else:
                x = layers(x)
                print(f"After net_upconv layer {i}:", x.shape)  # Print sizes after net_upconv layers
                skip = self.skip_connections


        pred = x[:, 0, :h, :w]
        boxes = x[:, 1, :h, :w]

        return pred, boxes


    def detect(self, image, max_pool_ks=7, min_score=0.2, max_det=15):
        heatmap, boxes = self(image[None])  
        heatmap = torch.sigmoid(heatmap.squeeze(0).squeeze(0)) 
        print("Shape of heatmap after sigmoid:", heatmap.shape) 
        sizes = boxes.squeeze(0)
        return [(peak[0], peak[1], peak[2], (sizes[peak[2], peak[1]]).item())
                for peak in extract_peak(heatmap, max_pool_ks, min_score, max_det)]

def save_model(model, name: str = 'detector.pt'):
    torch.save({'model_state_dict': model.state_dict()}, path.join(path.dirname(path.abspath(__file__)), name))

def load_model(model_class, name: str = 'detector.pt', device='cpu'):
    checkpoint = torch.load(path.join(path.dirname(path.abspath(__file__)), name), map_location=device)
    
    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    return model
