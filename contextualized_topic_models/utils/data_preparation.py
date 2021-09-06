import numpy as np
from sentence_transformers import SentenceTransformer
import scipy.sparse
import warnings
from contextualized_topic_models.datasets.dataset import CTMDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder

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


class TopicModelDataPreparation:

    def __init__(self, contextualized_model=None, show_warning=True):
        self.contextualized_model = contextualized_model
        self.vocab = []
        self.id2token = {}
        self.vectorizer = None
        self.label_encoder = None
        self.show_warning = show_warning

    def load(self, contextualized_embeddings, bow_embeddings, id2token, labels=None):
        return CTMDataset(contextualized_embeddings, bow_embeddings, id2token, labels)

    def fit(self, text_for_contextual, text_for_bow, labels=None):
        """
        This method fits the vectorizer and gets the embeddings from the contextual model

        :param text_for_contextual: list of unpreprocessed documents to generate the contextualized embeddings
        :param text_for_bow: list of preprocessed documents for creating the bag-of-words
        :param labels: list of labels associated with each document (optional).

        """

        if self.contextualized_model is None:
            raise Exception("You should define a contextualized model if you want to create the embeddings")

        # TODO: this count vectorizer removes tokens that have len = 1, might be unexpected for the users
        self.vectorizer = CountVectorizer()

        train_bow_embeddings = self.vectorizer.fit_transform(text_for_bow)
        train_contextualized_embeddings = bert_embeddings_from_list(text_for_contextual, self.contextualized_model)
        self.vocab = self.vectorizer.get_feature_names()
        self.id2token = {k: v for k, v in zip(range(0, len(self.vocab)), self.vocab)}

        if labels:
            self.label_encoder = OneHotEncoder()
            encoded_labels = self.label_encoder.fit_transform(np.array([labels]).reshape(-1, 1))
        else:
            encoded_labels = None

        return CTMDataset(train_contextualized_embeddings, train_bow_embeddings, self.id2token, encoded_labels)

    def transform(self, text_for_contextual, text_for_bow=None, labels=None):
        """
        This methods create the input for the prediction. Essentially, it creates the embeddings with the contextualized
        model of choice and with trained vectorizer.

        If text_for_bow is missing, it should be because we are using ZeroShotTM
        """

        if self.contextualized_model is None:
            raise Exception("You should define a contextualized model if you want to create the embeddings")

        if text_for_bow is not None:
            test_bow_embeddings = self.vectorizer.transform(text_for_bow)
        else:
            # dummy matrix
            if self.show_warning:
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn("The method did not have in input the text_for_bow parameter. This IS EXPECTED if you "
                          "are using ZeroShotTM in a cross-lingual setting")

            test_bow_embeddings = scipy.sparse.csr_matrix(np.zeros((len(text_for_contextual), 1)))
        test_contextualized_embeddings = bert_embeddings_from_list(text_for_contextual, self.contextualized_model)

        if labels:
            encoded_labels = self.label_encoder.transform(np.array([labels]).reshape(-1, 1))
        else:
            encoded_labels = None

        return CTMDataset(test_contextualized_embeddings, test_bow_embeddings, self.id2token, encoded_labels)
