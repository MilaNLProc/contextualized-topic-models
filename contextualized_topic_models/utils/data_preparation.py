import numpy as np
from sentence_transformers import SentenceTransformer
import scipy.sparse
import warnings
from contextualized_topic_models.datasets.dataset import CTMDataset
from sklearn.feature_extraction.text import CountVectorizer

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

    def __init__(self, contextualized_model=None):
        self.contextualized_model = contextualized_model
        self.vocab = []
        self.id2token = {}
        self.vectorizer = None

    def load(self, contextualized_embeddings, bow_embeddings, id2token):
        return CTMDataset(bow_embeddings, contextualized_embeddings, id2token)

    def create_training_set(self, text_for_contextual, text_for_bow):

        if self.contextualized_model is None:
            raise Exception("You should define a contextualized model if you want to create the embeddings")

        # TODO: this count vectorizer removes tokens that have len = 1, might be unexpected for the users
        self.vectorizer = CountVectorizer()

        train_bow_embeddings = self.vectorizer.fit_transform(text_for_bow)
        train_contextualized_embeddings = bert_embeddings_from_list(text_for_contextual, self.contextualized_model)
        self.vocab = self.vectorizer.get_feature_names()
        self.id2token = {k: v for k, v in zip(range(0, len(self.vocab)), self.vocab)}

        return CTMDataset(train_bow_embeddings, train_contextualized_embeddings, self.id2token)

    def create_test_set(self, text_for_contextual, text_for_bow=None):

        if self.contextualized_model is None:
            raise Exception("You should define a contextualized model if you want to create the embeddings")

        if text_for_bow is not None:
            test_bow_embeddings = self.vectorizer.transform(text_for_bow)
        else:
            # dummy matrix
            test_bow_embeddings = scipy.sparse.csr_matrix(np.zeros((len(text_for_contextual), 1)))
        test_contextualized_embeddings = bert_embeddings_from_list(text_for_contextual, self.contextualized_model)

        return CTMDataset(test_bow_embeddings, test_contextualized_embeddings, self.id2token)

class QuickText:
    """
    Integrated class to handle all the text preprocessing needed
    """
    def __init__(self, bert_model, text_for_bow, text_for_bert=None):
        """
        :param bert_model: string, bert model to use
        :param text_for_bow: list, list of sentences with the preprocessed text
        :param text_for_bert: list, list of sentences with the unpreprocessed text
        """
        self.vocab_dict = {}
        self.vocab = []
        self.index_dd = None
        self.idx2token = None
        self.bow = None
        self.bert_model = bert_model
        self.text_handler = ""
        self.data_bert = None
        self.text_for_bow = text_for_bow
        self.text_for_bert = text_for_bert
        self.loaded_from_config = False


    def prepare_bow(self):
        indptr = [0]
        indices = []
        data = []
        vocabulary = {}

        if self.text_for_bow is not None:
            docs = self.text_for_bow
        else:
            docs = self.text_for_bert

        for d in docs:
            for term in d.split():
                index = vocabulary.setdefault(term, len(vocabulary))
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))

        self.vocab_dict = vocabulary
        self.vocab = list(vocabulary.keys())

        warnings.simplefilter('always', DeprecationWarning)
        if len(self.vocab) > 2000:
            warnings.warn("The vocab you are using has more than 2000 words, reconstructing high-dimensional vectors requires"
                          "significantly more training epochs and training samples. "
                          "Consider reducing the number of vocabulary items. "
                          "See https://github.com/MilaNLProc/contextualized-topic-models#preprocessing "
                          "and https://github.com/MilaNLProc/contextualized-topic-models#tldr", Warning)

        self.idx2token = {v: k for (k, v) in self.vocab_dict.items()}
        self.bow = scipy.sparse.csr_matrix((data, indices, indptr), dtype=int)

    def load_configuration(self, bow_embeddings, contextualized_embeddings, vocab, id2token):
        """
        This method defines a way to instantiate the model with pre-trained data.
        """

        assert len(contextualized_embeddings) == bow_embeddings.shape[0]
        assert len(vocab) == len(id2token)
        self.data_bert = contextualized_embeddings
        self.bow = bow_embeddings
        self.vocab = vocab
        self.idx2token = id2token
        self.loaded_from_config = True

    def load_pre_trained_contextualized(self, contextualized_embeddings):
        """
        In case the contextualized embeddings have been already trained, it is possible to load them with this method
        """
        self.data_bert = contextualized_embeddings

    def load_dataset(self):
        if self.loaded_from_config:
            training_dataset = CTMDataset(self.bow, self.data_bert, self.idx2token)
        else:
            self.prepare_bow()

            if self.data_bert is None:
                if self.text_for_bert is not None:
                    self.data_bert = bert_embeddings_from_list(self.text_for_bert, self.bert_model)
                else:
                    self.data_bert = bert_embeddings_from_list(self.text_for_bow, self.bert_model)

            training_dataset = CTMDataset(self.bow, self.data_bert, self.idx2token)

        return training_dataset

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

        warnings.simplefilter('always', DeprecationWarning)
        if len(self.vocab) > 2000:
            warnings.warn("TextHandler class is deprecated and will be removed in version 2.0. Use QuickText.", Warning)

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

        warnings.simplefilter('always', DeprecationWarning)
        if len(self.vocab) > 2000:
            warnings.warn("The vocab you are using has more than 2000 words, reconstructing high-dimensional vectors requires"
                          "significantly more training epochs and training samples. "
                          "Consider reducing the number of vocabulary items. "
                          "See https://github.com/MilaNLProc/contextualized-topic-models#preprocessing "
                          "and https://github.com/MilaNLProc/contextualized-topic-models#tldr", Warning)

        self.idx2token = {v: k for (k, v) in self.vocab_dict.items()}
        self.bow = scipy.sparse.csr_matrix((data, indices, indptr), dtype=int)
