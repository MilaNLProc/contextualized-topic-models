import numpy as np
from sentence_transformers import SentenceTransformer
import scipy.sparse


def get_bag_of_words(data, min_length):

    vect = [np.bincount(x[x != np.array(None)].astype('int'), minlength=min_length)
            for x in data if np.sum(x[x != np.array(None)]) != 0]


    vect = scipy.sparse.csr_matrix(vect)
    return vect


def bert_embeddings_from_file(text_file, sbert_model_to_load):
    model = SentenceTransformer(sbert_model_to_load)

    with open(text_file) as filino:
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
        self.bow = None

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
        indptr = [0]
        indices = []
        data = []
        vocabulary = {}
        with open(self.file_name, "r") as filino:
            docs = filino.readlines()

        for d in docs:
            for term in d.split():
                index = vocabulary.setdefault(term, len(vocabulary))
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))

        self.vocab_dict = vocabulary
        self.vocab = list(vocabulary.keys())
        self.idx2token = {v: k for (k, v) in self.vocab_dict.items()}
        self.bow = scipy.sparse.csr_matrix((data, indices, indptr), dtype=int)
