from utile_functions import load_C2V, load_glove, create_embedding
from nlp_dataset import NLPDataset


if __name__ == '__main__':
    dataset = NLPDataset('data/text_twitter_raw.csv', ratio=0.05, exclude=None, senter=True, removeStop=True)
    gloVe = load_glove()
    c2v = load_C2V()
    em = create_embedding(dataset.textVocab.stoi, gloVe, c2v)
    print(em.weight.shape)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
