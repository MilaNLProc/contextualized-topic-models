import numpy as np
from sentence_transformers import SentenceTransformer


def get_bag_of_words(data, min_length):

    vect = [np.bincount(x[x != np.array(None)].astype('int'), minlength=min_length)
            for x in data if np.sum(x[x != np.array(None)]) != 0]
    return np.array(vect)


def bert_embeddings_from_file(text_file, sbert_model_to_load):
    model = SentenceTransformer(sbert_model_to_load)

    with open(text_file, encoding="latin") as filino:
        train_text = list(map(lambda x: x, filino.readlines()))

    return np.array(model.encode(train_text))


def bert_embeddings_from_list(texts, sbert_model_to_load):
    model = SentenceTransformer(sbert_model_to_load)
    return np.array(model.encode(texts))


class TextHandler:

    def __init__(self, file_name):
        self.file_name = file_name
        self.vocab_dict = {}
        self.vocab = []
        self.index_dd = None
        self.idx2token = None
        self.training_bow = None

    def load_text_file(self):
        """
        Loads a text file
        :param text_file:
        :return:
        """
        with open(self.file_name, "r") as filino:
            data = filino.readlines()

        return data

    def prepare(self):
        data = self.load_text_file()

        concatenate_text = ""
        for line in data:
            line = line.strip()
            concatenate_text += line + " "
        concatenate_text = concatenate_text.strip()

        self.vocab = list(set(concatenate_text.split()))

        for index, vocab in list(zip(range(0, len(self.vocab)), self.vocab)):
            self.vocab_dict[vocab] = index

        self.index_dd = np.array(list(map(lambda y: np.array(list(map(lambda x:
                                                                      self.vocab_dict[x], y.split()))), data)))
        self.idx2token = {v: k for (k, v) in self.vocab_dict.items()}
        self.bow = get_bag_of_words(self.index_dd, len(self.vocab))




