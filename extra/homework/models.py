import torch
from torch import nn
from . import utils

class LanguageModel(object):
    def predict_all(self, some_text):
        raise NotImplementedError('Abstract function LanguageModel.predict_all')

    def predict_next(self, some_text):
        return self.predict_all(some_text)[:, -1]

class Bigram(LanguageModel):
    def __init__(self):
        from os import path
        try:
            print("Loading 'bigram.th'...")
            self.first, self.transition = torch.load(path.join(path.dirname(path.abspath(__file__)), 'bigram.th'))
            print("Successfully loaded 'bigram.th'")
        except Exception as e:
            print(f"Error loading 'bigram.th': {e}")
        print("Shape of self.first:", self.first.shape)
        print("Shape of self.transition:", self.transition.shape)

    def predict_all(self, some_text):
        print("some_text:", some_text)
        
        # Adjust the index based on the length of some_text
        last_char_index = min(len(some_text), self.transition.shape[1] - 1)
        
        # Use one_hot for the last character in some_text
        one_hot_last_char = utils.one_hot(some_text[-1]) if last_char_index > 0 else utils.one_hot('')
        
        # Concatenate the first character and transition matrix with the adjusted range
        result = torch.cat((self.first[:, None], self.transition.t().matmul(one_hot_last_char)), dim=1)
        print("Result shape:", result.shape)
        
        # Adjust the result to include the correct number of characters
        return result[:, :last_char_index + 2]  # Include the correct number of characters

class AdjacentLanguageModel(LanguageModel):
    def predict_all(self, some_text):
        prob = 1e-3 * torch.ones(len(utils.vocab), len(some_text) + 1)
        if len(some_text):
            one_hot = utils.one_hot(some_text)

            # Adjust the index to include probabilities for the last character
            last_char_index = min(len(some_text) - 1, prob.shape[1] - 1)

            prob[-1, 1:last_char_index + 2] += 0.5 * one_hot[0]
            prob[:-1, 1:last_char_index + 2] += 0.5 * one_hot[1:]
            prob[0, 1:last_char_index + 2] += 0.5 * one_hot[-1]
            prob[1:, 1:last_char_index + 2] += 0.5 * one_hot[:-1]

        return (prob / prob.sum(dim=0, keepdim=True)).log()

class TCN(nn.Module, LanguageModel):
    class CausalConv1dBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, dilation):
            super(TCN.CausalConv1dBlock, self).__init__()
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation
            )
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x))

    def __init__(self, vocab_size, hidden_channels=50, kernel_size=2, num_layers=5, dilation_base=2):
        super(TCN, self).__init__()
        self.vocab_size = vocab_size  # Add vocab_size attribute
        self.embedding = nn.Embedding(vocab_size, hidden_channels)
        self.conv_blocks = nn.ModuleList([
            self.CausalConv1dBlock(hidden_channels, hidden_channels, kernel_size, dilation_base**i)
            for i in range(num_layers)
        ])
        self.output_layer = nn.Conv1d(hidden_channels, vocab_size, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.conv_blocks:
            x = block(x)
        logits = self.output_layer(x)
        return logits

    def predict_all(self, some_text):
        with torch.no_grad():
            # Convert characters to indices using utils.char_to_index
            indices = [utils.char_to_index(c) for c in some_text]
            indices = torch.tensor(indices).unsqueeze(0)  # Add batch dimension
            indices = indices.long()

            # Forward pass through the TCN
            logits = self.forward(indices)

            # Softmax to get probabilities
            probabilities = self.softmax(logits.squeeze(0))

            # Return log-likelihoods (not logits!)
            return probabilities.log()

# The save_model and load_model functions can remain unchanged
def save_model(model):
    from os import path
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'tcn.th'))

def load_model(vocab_size):
    from os import path
    r = TCN(vocab_size=vocab_size)  # Pass vocab_size to TCN constructor
    r.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th'), map_location='cpu'))
    return r
