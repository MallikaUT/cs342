import numpy as np
import torch

from . import utils


class LanguageModel(object):
    def predict_all(self, some_text):
        """
        Given some_text, predict the likelihoods of the next character for each substring from 0..i
        The resulting tensor is one element longer than the input, as it contains probabilities for all sub-strings
        including the first empty string (probability of the first character)

        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: torch.Tensor((len(utils.vocab), len(some_text)+1)) of log-probabilities
        """
        raise NotImplementedError("Abstract function LanguageModel.predict_all")

    def predict_next(self, some_text):
        """
        Given some_text, predict the likelihood of the next character

        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: a Tensor (len(utils.vocab)) of log-probabilities
        """
        return self.predict_all(some_text)[:, -1]


class Bigram(LanguageModel):
    """
    Implements a simple Bigram model. You can use this to compare your TCN to.
    The bigram, simply counts the occurrence of consecutive characters in transition, and chooses more frequent
    transitions more often. See https://en.wikipedia.org/wiki/Bigram .
    Use this to debug your `language.py` functions.
    """

    def __init__(self):
        from os import path

        self.first, self.transition = torch.load(
            path.join(path.dirname(path.abspath(__file__)), "bigram.th")
        )

    def predict_all(self, some_text):
        return torch.cat(
            (self.first[:, None], self.transition.t().matmul(utils.one_hot(some_text))),
            dim=1,
        )


class AdjacentLanguageModel(LanguageModel):
    """
    A simple language model that favours adjacent characters.
    The first character is chosen uniformly at random.
    Use this to debug your `language.py` functions.
    """

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
            """
            Causal 1D convolutional block with optional residual connection.

            :param in_channels: Number of input channels.
            :param out_channels: Number of output channels.
            :param kernel_size: Size of the convolutional kernel.
            :param dilation: Dilation rate for the convolutional layer.
            """
            super().__init__()
            # Define the components of the block: padding, convolution, and ReLU activation
            self.network = torch.nn.Sequential(
                torch.nn.ConstantPad1d((2 * dilation, 0), 0),
                torch.nn.Conv1d(
                    in_channels, out_channels, kernel_size, dilation=dilation
                ),
                torch.nn.ReLU(),
            )
            # Optional downsample to match dimensions for the residual connection
            self.downsample = None
            if in_channels != out_channels:
                self.downsample = torch.nn.Conv1d(in_channels, out_channels, 1)

        def forward(self, x):
            """
            Forward pass of the CausalConv1dBlock.

            :param x: Input tensor.
            :return: Output tensor after applying the convolutional block.
            """
            identity = x
            if self.downsample is not None:
                identity = self.downsample(identity)
            # Apply the convolutional block and add the residual connection
            return self.network(x) + identity

    def __init__(
        self, layers=[50, 50, 50, 50, 50, 50, 64, 64, 64], char_set=utils.vocab
    ):
        """
        TCN model with stacked CausalConv1dBlocks and a 1x1 classifier.

        :param layers: List of channel sizes for each CausalConv1dBlock.
        :param char_set: Character set used for language modeling.
        """
        super().__init__()
        # Initialize the probability parameter for the first character
        self.init_prob = torch.nn.Parameter(torch.ones(len(char_set)) / len(char_set))
        self.char_set = char_set
        c = len(char_set)
        L = []
        current_dilation = 1
        # Create the stack of CausalConv1dBlocks based on the specified layers
        for l in layers:
            L.append(self.CausalConv1dBlock(c, l, 3, current_dilation))
            current_dilation *= 2
            c = l
        self.network = torch.nn.Sequential(*L)
        # 1x1 convolutional classifier for generating the output
        self.classifier = torch.nn.Conv1d(c, len(char_set), 1)

    def forward(self, x):
        """
        Forward pass of the TCN model.

        :param x: Input tensor representing one-hot encoded characters.
        :return: Output tensor with log-likelihoods for the next character.
        """
        B, S, L = x.shape
        if L == 0:
            # If input sequence is empty, return the initial probability parameter
            init = (
                self.init_prob.view(len(self.char_set), 1)
                .expand(-1, B)
                .view(-1, len(self.char_set), 1)
            )
            return init
        else:
            # Apply the CausalConv1dBlocks and the 1x1 classifier
            prob = self.classifier(self.network(x))
            init = (
                self.init_prob.view(len(self.char_set), 1)
                .expand(-1, B)
                .view(-1, len(self.char_set), 1)
            )
            output = torch.cat((init, prob), dim=2)
            return output

    def predict_all(self, some_text):
        """
        Predict the log-likelihoods for the next character given a partial sequence.

        :param some_text: Input sequence as a string.
        :return: Log-likelihoods for the next character.
        """
        vocab = self.char_set
        device = next(self.parameters()).device
        if len(some_text) == 0:
            # If input sequence is empty, return log-softmax of the initial probability parameter
            prob = self.init_prob.view(len(vocab), 1)
            return torch.nn.functional.log_softmax(prob, dim=0)
        else:
            # Convert the input string to a one-hot encoded tensor
            x = torch.tensor(
                np.array(list(some_text))[None, :] == np.array(list(vocab))[:, None]
            ).float()
            x = x[None, :, :]
            x = x.to(device)
            # Forward pass through the TCN model
            prob = self.forward(x)
            prob = prob.squeeze()
            # Apply log-softmax to get log-likelihoods
            return torch.nn.functional.log_softmax(prob, dim=0)


def save_model(model):
    from os import path

    return torch.save(
        model.state_dict(), path.join(path.dirname(path.abspath(__file__)), "tcn.th")
    )


def load_model():
    from os import path

    r = TCN()
    r.load_state_dict(
        torch.load(
            path.join(path.dirname(path.abspath(__file__)), "tcn.th"),
            map_location="cpu",
        )
    )
    return r
