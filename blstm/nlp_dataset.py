import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import spacy
from instance import Instance
from vocab import Vocab
from torch.nn.utils.rnn import pad_sequence


class NLPDataset(Dataset):

    def __init__(self, path, textVocab=None, labelVocab=None, ratio=0.7, exclude=None, senter=False, removeStop=True):
        csv_data = pd.read_csv(path)
        data_length = len(csv_data) * ratio
        self.instances = []
        words = []
        labels = []
        if exclude is None:
            nlp = spacy.load("en_core_web_sm")
        else:
            nlp = spacy.load("en_core_web_sm", exclude=exclude)
        if senter:
            nlp.disable_pipe("parser")
            nlp.enable_pipe("senter")
        for index, row in csv_data.iterrows():
            if index < data_length and textVocab is not None:
                continue
            if index > data_length and textVocab is None:
                break
            doc = nlp(row[0].strip())
            if exclude is None:
                instance = Instance([d.lemma_ for d in doc if not d.is_stop or removeStop], row[1].strip())
            else:
                instance = Instance([d.text for d in doc if not d.is_stop or removeStop], row[1].strip())
            words.extend(instance.instance_text)
            labels.append(instance.instance_label)
            self.instances.append(instance)
        if textVocab is None:
            self.textVocab = Vocab(words)
            self.labelVocab = Vocab(labels, labels=True)
        else:
            self.textVocab = textVocab
            self.labelVocab = labelVocab

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        instance = self.instances[item]
        return self.textVocab.encode(instance.instance_text), self.labelVocab.encode(instance.instance_label)


def pad_collate_fn(batch, pad_index=0):
    dev = "cuda:0"
    device = torch.device(dev)
    texts, labels = zip(*batch)  # Assuming the instance is in tuple-like form
    lengths = torch.tensor([len(text) for text in texts]).to(device)
    text_tensor = pad_sequence(texts, padding_value=pad_index)
    return text_tensor.transpose(1, 0), torch.tensor(labels).to(device), lengths


if __name__ == '__main__':
    dev = "cuda:0"
    device = torch.device(dev)
    dataset_ratio = 0.1
    shuffle = False
    batch_size = 1
    disable = ["tagger", "attribute_ruler", "lemmatizer"]
    dataset = NLPDataset('data/text_twitter_raw.csv', ratio=dataset_ratio, exclude=disable, senter=True)
    data_loader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=pad_collate_fn)
    print("Dataset testing")
    texts, labels, lengths = next(iter(data_loader))
    print("First text from dataset")
    print(texts)
    print("First label from dataset")
    print(labels)
    print("First text length from dataset")
    print(lengths)
