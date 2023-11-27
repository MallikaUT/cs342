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

    # Sum all log probabilities in the sequence
    total_log_likelihood = torch.sum(log_probs[:, -1]).item()
    return total_log_likelihood

def sample_random(model: LanguageModel, max_length: int = 100, min_likelihood: float = -0.05):
    result = ""
    for _ in range(max_length):
        log_probs = model.predict_all(result)

        # Check if log_probs is empty
        if log_probs.numel() == 0:
            break

        # Handle the case where the result is empty
        if len(result) == 0:
            probabilities = torch.exp(log_probs[:, 0])  # Use the first column for initial probabilities
        else:
            probabilities = torch.exp(log_probs[:, -1])  # Convert log probabilities to probabilities

        sampled_index = utils.sample_from_distribution(probabilities)


        # Debug print
        print(f"sampled_index before conversion: {sampled_index}")
                # Convert sampled_index to character
        sampled_char = utils.index_to_char(sampled_index)

        # Debug prints
        print(f"log_probs shape: {log_probs.shape}, sampled_index: {sampled_index}, min_likelihood: {min_likelihood}")

        # Adjust likelihood threshold
        if log_probs[0, 0] < float(min_likelihood):
            print("Skipping due to low likelihood")
            continue

        result += sampled_char

        if len(result) >= max_length:
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
    heap = TopNHeap(n_results)
    beam = [{'text': '', 'log_likelihood': 0.0}]
    seen_texts = set()

    while len(heap.elements) < n_results and len(beam) > 0:
        new_beam = []
        for candidate in beam:
            current_text = candidate['text']

            # Handle the case where current_text is empty
            if not current_text:
                log_probs = model.predict_all(current_text)
            else:
                log_probs = model.predict_all(current_text)  # Use predict_all instead of predict_next

            for char_index in range(len(utils.vocab)):
                new_char = utils.index_to_char(char_index)
                new_text = current_text + new_char

                # Compute log_probs only once
                log_probs = model.predict_all(new_text)  # Use predict_all instead of predict_next

                # Adjust the index based on the length of new_text
                last_char_index = min(len(new_text) - 1, log_probs.shape[-1] - 1)
                new_log_likelihood = candidate['log_likelihood'] + log_probs[char_index, last_char_index].item()

                print(f"Char: {new_char}, Log Likelihood: {new_log_likelihood}")
                if new_text not in seen_texts:
                    seen_texts.add(new_text)
                    if new_char == '.' or len(new_text) >= max_length:
                        heap.add((new_log_likelihood, new_text))
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

    # Compute vocab_size (you need to adjust this part based on your actual implementation)
    vocab_size = len(utils.vocab)

    if args.model == 'Adjacent':
        lm = AdjacentLanguageModel()
    elif args.model == 'TCN':
        lm = load_model(vocab_size)
    else:
        lm = Bigram()

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
