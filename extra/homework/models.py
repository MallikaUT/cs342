import torch
import torch.nn.functional as F

from . import utils
from .utils import one_hot

class LanguageModel(object):
    
    def predict_all(self, some_text):
        """
        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: torch.Tensor((len(utils.vocab), len(some_text)+1)) of log-probabilities
        """
        pass
    
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

    def predict_all(self, some_text):
        return torch.cat((self.first[:, None], self.transition.t().matmul(utils.one_hot(some_text))), dim=1)


class AdjacentLanguageModel(LanguageModel):
 
    def predict_all(self, some_text):
        prob = 1e-3 * torch.ones(len(utils.vocab), len(some_text) + 1)
        if len(some_text):
            one_hot = utils.one_hot(some_text)
            prob[-1, 1:] += 0.5 * one_hot[0]
            prob[:-1, 1:] += 0.5 * one_hot[1:]
            prob[0, 1:] += 0.5 * one_hot[-1]
            prob[1:, 1:] += 0.5 * one_hot[:-1]
        return (prob / prob.sum(dim=0, keepdim=True)).log()

class TCN(torch.nn.Module, LanguageModel):

    class CausalConv1dBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, dilation):
            super(TCN.CausalConv1dBlock, self).__init__()
            self.pad1d = torch.nn.ConstantPad1d((dilation * (kernel_size - 1), 0), 0)
            self.c1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)

        def forward(self, x):
            return F.relu(self.c1(self.pad1d(x)))

    def __init__(self, layers=[28, 16, 8], char_set="string"):
        super(TCN, self).__init__()

        c = 28 
        L = []
        total_dilation = 1
        for l in layers:
            L.append(self.CausalConv1dBlock(c, l, kernel_size=3, dilation=total_dilation))
            L.append(torch.nn.ReLU())
            total_dilation *= 2
            c = l

        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Conv1d(c, 28, 1)

    def forward(self, x):
        first_char_distribution = torch.nn.Parameter(torch.rand(x.shape[0], x.shape[1], 1))
        output = self.network(x)
        output = self.classifier(output)
        output = torch.cat((first_char_distribution, output), dim=2)
        return output

    def predict_all(self, some_text):
        one_hotx = one_hot(some_text)[None]
        output = self.forward(one_hotx)
        output = output[0, :, :]
        log_likelihoods = F.log_softmax(output, dim=1)
        return log_likelihoods

def save_model(model):
    from os import path
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'tcn.th'))

def load_model():
    from os import path
    r = TCN()
    r.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th'), map_location='cpu'))
    return r
