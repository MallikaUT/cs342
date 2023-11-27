from argparse import ArgumentParser
from .models import LanguageModel, AdjacentLanguageModel, Bigram, load_model
from . import utils
import torch


def log_likelihood(model: LanguageModel, some_text: str):
    """
    Evaluate the log-likelihood of a given string.

    :param model: A LanguageModel
    :param some_text: A string
    :return: float
    """
    some_text = some_text.lower()
    log_probs = model.predict_all(some_text)
    total_log_likelihood = log_probs.sum().item()
    return total_log_likelihood


def sample_random(model: LanguageModel, max_length: int = 100):
    result = ""
    for _ in range(max_length):
        log_probs = model.predict_all(result)
        probabilities = torch.exp(log_probs[:, -1])  # Convert log probabilities to probabilities
        sampled_index = utils.sample_from_distribution(probabilities)
        result += utils.index_to_char(sampled_index)
        if result[-1] == '.':
            break
    return result


class TopNHeap:
    def __init__(self, N):
        self.elements = []
        self.N = N

    def add(self, e):
        from heapq import heappush, heapreplace
        if len(self.elements) < self.N:
            heappush(self.elements, e)
        elif self.elements[0][0] < e[0]:
            heapreplace(self.elements, e)


def beam_search(model: LanguageModel, beam_size: int, n_results: int = 10, max_length: int = 100, average_log_likelihood: bool = False):
    """
    Use beam search to find the highest likelihood generations.

    :param model: A LanguageModel
    :param beam_size: The size of the beam in beam search (number of sentences to keep around)
    :param n_results: The number of results to return
    :param max_length: The maximum sentence length
    :param average_log_likelihood: Pick the best beams according to the average log-likelihood, not the sum
    :return: A list of strings of size n_results
    """
    heap = TopNHeap(n_results)
    beam = [{'text': '', 'log_likelihood': 0.0}]

    while len(heap.elements) < n_results and len(beam) > 0:
        new_beam = []
        for candidate in beam:
            current_text = candidate['text']
            log_probs = model.predict_next(current_text)

            for char_index in range(len(utils.vocab)):
                new_char = utils.index_to_char(char_index)
                new_text = current_text + new_char
                new_log_likelihood = candidate['log_likelihood'] + log_probs[char_index].item()

                if new_char == '.' or len(new_text) >= max_length:
                    heap.add((new_log_likelihood / len(new_text), new_text))
                else:
                    new_beam.append({'text': new_text, 'log_likelihood': new_log_likelihood})

        new_beam.sort(key=lambda x: x['log_likelihood'], reverse=True)
        beam = new_beam[:beam_size]

    result_sentences = [item[1] for item in heap.elements]
    return result_sentences


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', choices=['Adjacent', 'Bigram', 'TCN'], default='Adjacent')
    args = parser.parse_args()

    lm = AdjacentLanguageModel() if args.model == 'Adjacent' else (load_model() if args.model == 'TCN' else Bigram())

    for s in ['abcdefg', 'abcgdef', 'abcbabc', '.abcdef', 'fedcba.']:
        print(s, float(log_likelihood(lm, s)))
    print()

    for i in range(10):
        s = sample_random(lm)
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100):
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100, average_log_likelihood=True):
        print(s, float(log_likelihood(lm, s)) / len(s))
