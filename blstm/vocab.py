import torch


class Vocab:

    def __init__(self, words, max_size=-1, min_freq=0, labels=False):
        self.stoi = {}
        self.itos = {}
        start = 0

        if not labels:
            self.stoi = {'<PAD>': 0, '<UNK>': 1}
            self.itos = {0: '<PAD>', 1: '<UNK>'}
            start = 2
        word_freq = {}
        for w in words:
            if w in word_freq:
                word_freq[w] += 1
            else:
                word_freq[w] = 1
        sort_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for index, word in enumerate(sort_words, start=start):
            if index == max_size or word[1] < min_freq:
                break
            self.stoi[word[0]] = index
            self.itos[index] = word[0]

    def encode(self, text):
        dev = "cuda:0"
        device = torch.device(dev)
        if isinstance(text, str):
            if text in self.stoi:
                return torch.tensor(self.stoi[text]).to(device)
            else:
                return torch.tensor(1).to(device)
        index = []
        for w in text:
            if w in self.stoi:
                index.append(self.stoi[w])
            else:
                index.append(1)
        return torch.tensor(index).to(device)
