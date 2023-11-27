import torch
import torch.nn.functional as F

from . import utils
from .utils import one_hot


#NOTE THIS IS JUST TO TEST THE GRADER, NOV 15 2021


class LanguageModel(object):
    
    def predict_all(self, some_text):
        """
      
        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: torch.Tensor((len(utils.vocab), len(some_text)+1)) of log-probabilities
        """
        
    def predict_next(self, some_text):
        """
        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: a Tensor (len(utils.vocab)) of log-probabilities
        """
        return self.predict_all(some_text)[:, -1]


class Bigram(LanguageModel):
 
    def __init__(self):
        from os import path
        self.first, self.transition = torch.load(path.join(path.dirname(path.abspath(__file__)), 'bigram.th'))


    #bigram predictALL

    def predict_all(self, some_text):
        return torch.cat((self.first[:, None], self.transition.t().matmul(utils.one_hot(some_text))), dim=1)


class AdjacentLanguageModel(LanguageModel):
 
    def predict_all(self, some_text):
        prob = 1e-3*torch.ones(len(utils.vocab), len(some_text)+1)
        if len(some_text):
            one_hot = utils.one_hot(some_text)
            prob[-1, 1:] += 0.5*one_hot[0]
            prob[:-1, 1:] += 0.5*one_hot[1:]
            prob[0, 1:] += 0.5*one_hot[-1]
            prob[1:, 1:] += 0.5*one_hot[:-1]
        return (prob/prob.sum(dim=0, keepdim=True)).log()

#==============================================TCN===========================================



class TCN(torch.nn.Module, LanguageModel):     #MY WARNING:  TCN in example DOES NOT inherit from Language Model
    class CausalConv1dBlock(torch.nn.Module):
        
        
        def __init__(self, in_channels, out_channels, kernel_size, dilation):
          
            self.pad1d = torch.nn.ConstantPad1d((2*dilation,0), 0)
            self.c1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, dilation=dilation)
            #self.b1 = torch.nn.BatchNorm2d(out_channels)
                          
         
        def forward(self, x):
            return F.relu(self.c1(self.pad1d(x))) 
    
    #--------------->TCN INIT

    def __init__(self, layers=[28,16,8], char_set="string"):
        super().__init__()

        total_dilation = 1  # starting dilation at 2, not 1?????
        c = 28
        L = []

        for l in layers:
            
            L.append(torch.nn.ConstantPad1d((total_dilation, 0), 0))
            L.append(torch.nn.Conv1d(c, l, kernel_size=2, dilation=total_dilation))
            L.append(torch.nn.ReLU())
            total_dilation *= 2
            c = l

        self.total_dilation = total_dilation  # Move this line here
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Conv1d(c, 28, 1)
        
    #--------------------------TCN FORWARD()
    
    def forward(self, x):
        print("Input sequence size:", x.size())

        # Check if the input sequence is too short
        if x.size(2) < 3:
            # Handle short sequences, for example, return a default value
            return torch.zeros(x.size(0), x.size(1), 28)  # Modify the shape as needed

        first_char_distribution = torch.nn.Parameter(torch.rand(x.size(0), x.size(1), 1))
        total_dilation = self.total_dilation
        output = self.network(x)
        output = self.classifier(output)
        output = torch.cat((first_char_distribution, output), dim=2)

        return output
        
    def predict_all(self, some_text):
        one_hotx = one_hot(some_text)
        one_hotx = one_hotx.unsqueeze(0)

        # Check if the input sequence is too short
        if one_hotx.size(2) < 3:
            # Handle short sequences, for example, return a default value
            return torch.zeros(len(utils.vocab), len(some_text) + 1)  # Modify the shape as needed

        output = self.forward(one_hotx)

        output = F.log_softmax(output, dim=1)

        # Squeeze the extra dimension
        output = output.squeeze(0)

        return output 
        
        

def save_model(model):
    from os import path
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'tcn.th'))


def load_model():
    from os import path
    r = TCN()
    r.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th'), map_location='cpu'))
    return r