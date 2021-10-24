from sklearn.feature_extraction.text import CountVectorizer
import string
from nltk.corpus import stopwords as stop_words
from gensim.utils import deaccent
import warnings

class WhiteSpacePreprocessing():
    """
    Provides a very simple preprocessing script that filters infrequent tokens from text
    """

    def __init__(self, documents, stopwords_language="english", vocabulary_size=2000):
        """

        :param documents: list of strings
        :param stopwords_language: string of the language of the stopwords (see nltk stopwords)
        :param vocabulary_size: the number of most frequent words to include in the documents. Infrequent words will be discarded from the list of preprocessed documents
        """
        self.documents = documents
        self.stopwords = set(stop_words.words(stopwords_language))
        self.vocabulary_size = vocabulary_size

        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn("WhiteSpacePreprocessing is deprecated and will be removed in future versions."
                      "Use WhiteSpacePreprocessingStopwords.")

    def preprocess(self):
        """
        Note that if after filtering some documents do not contain words we remove them. That is why we return also the
        list of unpreprocessed documents.

        :return: preprocessed documents, unpreprocessed documents and the vocabulary list
        """
        preprocessed_docs_tmp = self.documents
        preprocessed_docs_tmp = [deaccent(doc.lower()) for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [doc.translate(
            str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 0 and w not in self.stopwords])
                                 for doc in preprocessed_docs_tmp]

        vectorizer = CountVectorizer(max_features=self.vocabulary_size)
        vectorizer.fit_transform(preprocessed_docs_tmp)
        temp_vocabulary = set(vectorizer.get_feature_names())

        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if w in temp_vocabulary])
                                 for doc in preprocessed_docs_tmp]

        preprocessed_docs, unpreprocessed_docs = [], []
        for i, doc in enumerate(preprocessed_docs_tmp):
            if len(doc) > 0:
                preprocessed_docs.append(doc)
                unpreprocessed_docs.append(self.documents[i])

        vocabulary = list(set([item for doc in preprocessed_docs for item in doc.split()]))

        return preprocessed_docs, unpreprocessed_docs, vocabulary


class WhiteSpacePreprocessingStopwords():
    """
    Provides a very simple preprocessing script that filters infrequent tokens from text
    """

    def __init__(self, documents, stopwords_list=None, vocabulary_size=2000):
        """

        :param documents: list of strings
        :param stopwords_list: list of the stopwords to remove
        :param vocabulary_size: the number of most frequent words to include in the documents. Infrequent words will be discarded from the list of preprocessed documents
        """
        self.documents = documents
        self.stopwords = set(stopwords_list)
        self.vocabulary_size = vocabulary_size

    def preprocess(self):
        """
        Note that if after filtering some documents do not contain words we remove them. That is why we return also the
        list of unpreprocessed documents.

        :return: preprocessed documents, unpreprocessed documents and the vocabulary list
        """
        preprocessed_docs_tmp = self.documents
        preprocessed_docs_tmp = [deaccent(doc.lower()) for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [doc.translate(
            str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 0 and w not in self.stopwords])
                                 for doc in preprocessed_docs_tmp]

        vectorizer = CountVectorizer(max_features=self.vocabulary_size)
        vectorizer.fit_transform(preprocessed_docs_tmp)
        temp_vocabulary = set(vectorizer.get_feature_names())

        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if w in temp_vocabulary])
                                 for doc in preprocessed_docs_tmp]

        preprocessed_docs, unpreprocessed_docs = [], []
        for i, doc in enumerate(preprocessed_docs_tmp):
            if len(doc) > 0:
                preprocessed_docs.append(doc)
                unpreprocessed_docs.append(self.documents[i])

        vocabulary = list(set([item for doc in preprocessed_docs for item in doc.split()]))

        return preprocessed_docs, unpreprocessed_docs, vocabulary


