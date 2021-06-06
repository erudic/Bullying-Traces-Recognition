import torch
from torch.nn import Embedding
import pandas as pd
import numpy as np
import csv
import chars2vec


def load_glove():
    dev = "cuda:0"
    device = torch.device(dev)
    gloVe = {}
    vectors = open('data/sst_glove_6b_300d.txt', "r")
    for v in vectors:
        vector = v.split(" ")
        text = vector.pop(0)
        gloVe[text] = torch.tensor([float(num) for num in vector]).to(device)
    vectors.close()
    return gloVe


def create_embedding(vocab, gloVe=None, c2v=None):
    dev = "cuda:0"
    device = torch.device(dev)
    tensor_size = 600
    if gloVe is None:
        tensor_size -= 300
    if c2v is None:
        tensor_size -= 300
    if gloVe is None and c2v is None:
        tensor_size = 300
        return Embedding.from_pretrained(torch.randn((len(vocab), tensor_size)).to(device), padding_idx=0, freeze=False)

    embedding_matrix = torch.randn((len(vocab), tensor_size)).to(device)

    for index, word in enumerate(vocab):
        ten = None
        if gloVe is not None:
            glVec = torch.randn(300).to(device)
        if c2v is not None:
            c2Vec = torch.randn(300).to(device)
        if word == '<PAD>':
            ten = torch.zeros((1, tensor_size)).to(device)
        else:
            if gloVe is not None and word in gloVe:
                glVec = gloVe[word]
            if c2v is not None and word in c2v:
                c2Vec = c2v[word]
            if gloVe is None:
                ten = c2Vec
            elif c2v is None:
                ten = glVec
            else:
                torch.cat((glVec, c2Vec), 0)
        if ten is not None:
            embedding_matrix[index] = ten
    return Embedding.from_pretrained(embedding_matrix, padding_idx=0, freeze=False)


def generateC2VRepresentation(vocab):
    li = np.array(list(vocab))
    print(li)
    f = open('data/c2v_300d.csv', 'w', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(['word', 'c2v'])
    c2v_model = chars2vec.load_model('eng_300')
    vec = c2v_model.vectorize_words(li)
    for t, v in zip(li, vec):
        writer.writerow([t, v])
    f.close()
    exit(0)


def load_C2V():
    dev = "cuda:0"
    device = torch.device(dev)
    csv_data = pd.read_csv('data/c2v_300d.csv')
    c2v = {}
    for index, row in csv_data.iterrows():
        row[1] = row[1].replace('[', '')
        row[1] = row[1].replace(']', '')
        k = row[1].strip().split()
        c2v[row[0]] = torch.tensor([float(num) for num in k]).to(device)
    return c2v
