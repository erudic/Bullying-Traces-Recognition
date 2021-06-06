from abc import ABC

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from nlp_dataset import NLPDataset, pad_collate_fn
from utile_functions import load_C2V, load_glove, create_embedding


class LSTMModel(nn.Module, ABC):

    def __init__(self, embedding, lstm=[600, 300], fc=[600, 150, 1], num_layer=2, bidirectional=True, dropout=0.5,
                 criterion=nn.BCEWithLogitsLoss()):
        super().__init__()
        dev = "cuda:0"
        device = torch.device(dev)

        self.embedding_matrix = embedding
        self.criterion = criterion
        l = []
        for i in range(0, len(lstm) - 1):
            lst = nn.LSTM(lstm[i], lstm[i + 1], num_layers=num_layer, bidirectional=bidirectional, dropout=dropout).to(
                device)
            l.append(lst)

        self.lstms = nn.ModuleList(l)
        f = []
        for i in range(0, len(fc) - 1):
            f.append(nn.Linear(in_features=fc[i], out_features=fc[i + 1], bias=True).to(device))
        self.fcs = nn.ModuleList(f)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        dev = "cuda:0"
        device = torch.device(dev)
        texts, labels, lengths = X
        matrix_x = self.embedding_matrix(texts)
        matrix_x = matrix_x.transpose(1, 0)

        hidden = None
        out = matrix_x
        for lstm in self.lstms:
            out, hidden = lstm(out, hidden)
        h = out[-1]
        for fc in self.fcs:
            h = fc(h)
            h = torch.relu(h)
        return h

    def get_loss(self, x, y):
        return self.criterion(x, y)

    def train_model(self, train_data_loader, optimizer):
        self.train()
        for i, data in enumerate(train_data_loader):
            texts, labels, lengths = data
            loss = self.get_loss(self.forward(data).squeeze(-1), labels.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.25)
            optimizer.step()
            optimizer.zero_grad()

    def evaluate(self, data_loader, epoch):
        self.eval()
        acc = 0
        f1 = 0
        recall = 0
        prec = 0
        counter = 0
        loss = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                texts, labels, lengths = data
                logits = self.forward(data).squeeze(-1)
                loss += self.get_loss(logits, labels.float())
                y_pred = torch.round(torch.sigmoid(logits))
                labels_np = labels.cpu().detach().numpy()
                y_np = y_pred.cpu().detach().numpy()
                acc += accuracy_score(labels_np, y_np)
                f1 += f1_score(labels_np, y_np)
                recall += recall_score(labels_np, y_np)
                prec += precision_score(labels_np, y_np)
                counter += 1
            print(f'Epoch: {epoch}, eval loss= {loss / counter}')
            print(f'Accuracy= {acc / counter}')
            print(f'f1= {f1 / counter}')
            print(f'Precision= {prec / counter}')
            print(f'Recall= {recall / counter}')
            print("-----------------------------")

    def test_model(self, test_loader):
        for i, data in enumerate(test_loader):
            texts, labels, lengths = data
            logits = self.forward(data).squeeze(-1)
            loss = self.get_loss(logits, labels.float())
            y_pred = torch.round(torch.sigmoid(logits))
            print(y_pred)
            labels_np = labels.cpu().detach().numpy()
            y_np = y_pred.cpu().detach().numpy()
            acc = accuracy_score(labels_np, y_np)
            f1 = f1_score(labels_np, y_np)
            recall = recall_score(labels_np, y_np)
            prec = precision_score(labels_np, y_np)
            print(f'Accuracy= {acc}')
            print(f'f1= {f1}')
            print(f'Precision= {prec}')
            print(f'Recall= {recall}')
            print(y_pred)


if __name__ == '__main__':
    dev = "cuda:0"
    device = torch.device(dev)

    train_dataset = NLPDataset('data/text_twitter_raw.csv', ratio=0.01, exclude=None, senter=True, removeStop=False)
    '''
    valid_dataset = NLPDataset('data/text_twitter_raw.csv', ratio=0.7, textVocab=train_dataset.textVocab,
                               labelVocab=train_dataset.labelVocab)
    '''
    gloVe = load_glove()
    c2v = load_C2V()

    em = create_embedding(train_dataset.textVocab.stoi, gloVe, c2v)
    model = LSTMModel(em).to(device)
    print(model.parameters)
    learning_rate = 0.001
    batch_size = 64
    epoch = 20
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  shuffle=False, collate_fn=pad_collate_fn)
    '''
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                  shuffle=False, collate_fn=pad_collate_fn)
    '''
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # train
    for i in range(0, epoch):
        model.train_model(train_dataloader, optim)
        model.evaluate(train_dataloader, i)
