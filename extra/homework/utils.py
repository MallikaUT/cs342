import re
import string

import torch
from torch.utils.data import Dataset, DataLoader

vocab = string.ascii_lowercase + ' .'


def one_hot(s: str):
    """
    Converts a string into a one-hot encoding
    :param s: a string with characters in vocab (all other characters will be ignored!)
    :return: a once hot encoding Tensor r (len(vocab), len(s)), with r[j, i] = (s[i] == vocab[j])
    """
    import numpy as np
    if len(s) == 0:
        return torch.zeros((len(vocab), 0))
    return torch.as_tensor(np.array(list(s.lower()))[None, :] == np.array(list(vocab))[:, None]).float()


class SpeechDataset(Dataset):
    """
    Creates a dataset of strings from a text file.
    All strings will be of length max_len and padded with '.' if needed.

    By default this dataset will return a string, this string is not directly readable by pytorch.
    Use transform (e.g. one_hot) to convert the string into a Tensor.
    """

    def __init__(self, dataset_path, transform=None, max_len=250):
        with open(dataset_path) as file:
            Data = file.read()
        
        #code below seems to just make text lower case, remove spacing after period.  Tokenize?
        Data = Data.lower()
        reg = re.compile('[^%s]' % vocab)
        period = re.compile(r'[ .]*\.[ .]*')
        space = re.compile(r' +')
        sentence = re.compile(r'[^.]*\.')
        self.data = space.sub(' ',period.sub('.',reg.sub('', Data)))  #removes spacing after period?
        
        #print (f'the size of data is {len(self.data)}')
        if max_len is None:
            self.range = [(m.start(), m.end()) for m in sentence.finditer(self.data)]
        else:
            self.range = [(m.start(), m.start()+max_len) for m in sentence.finditer(self.data)]
            self.data += self.data[:max_len]

        if transform is not None:
          
            self.data = transform(self.data)

    def __len__(self):
        return len(self.range)

    def __getitem__(self, idx):
        s, e = self.range[idx]

        if isinstance(self.data, str):
            #print (f'Based on above numbers, returning self.data[s:e]: {self.data[s:e]} ')
            return self.data[s:e]        #if it's a string, return s to e substring only

        return self.data[:, s:e]   #how to randomize this?


def load_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = SpeechDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


if __name__ == "__main__":
    train_data = load_data('data/train.txt',  transform=one_hot)
    
    #train_data = load_data('data/valid.txt',  max_len=None)
    #Transform did not work, raised stack exception: load_data('data/valid.txt',  transform=one_hot, max_len=None)  
    for s in train_data:
      print (s.shape)#torch.Size([32, 28, 250])