import numpy as np
from sentence_transformers import SentenceTransformer
import scipy.sparse


def get_bag_of_words(data, min_length):
    """
    Creates the bag of words
    """
    vect = [np.bincount(x[x != np.array(None)].astype('int'), minlength=min_length)
            for x in data if np.sum(x[x != np.array(None)]) != 0]

    vect = scipy.sparse.csr_matrix(vect)
    return vect


def bert_embeddings_from_file(text_file, sbert_model_to_load, batch_size=200):
    """
    Creates SBERT Embeddings from an input file
    """
    model = SentenceTransformer(sbert_model_to_load)
    with open(text_file, encoding="utf-8") as filino:
        train_text = list(map(lambda x: x, filino.readlines()))

    return np.array(model.encode(train_text, show_progress_bar=True, batch_size=batch_size))


def bert_embeddings_from_list(texts, sbert_model_to_load, batch_size=200):
    """
    Creates SBERT Embeddings from a list
    """
    model = SentenceTransformer(sbert_model_to_load)
    return np.array(model.encode(texts, show_progress_bar=True, batch_size=batch_size))


class TextHandler:
    """
    Class used to handle the text preparation and the BagOfWord
    """
    def __init__(self, file_name=None, sentences=None):
        self.file_name = file_name
        self.sentences = sentences
        self.vocab_dict = {}
        self.vocab = []
        self.index_dd = None
        self.idx2token = None
        self.bow = None

    def prepare(self):
        indptr = [0]
        indices = []
        data = []
        vocabulary = {}

        if self.sentences == None and self.file_name == None:
            raise Exception("Sentences and file_names cannot both be none")

        if self.sentences != None:
            docs = self.sentences
        elif self.file_name != None:
            with open(self.file_name, encoding="utf-8") as filino:
                docs = filino.readlines()
        else:
            raise Exception("One parameter between sentences and file_name should be selected")

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
