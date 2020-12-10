from sklearn.feature_extraction.text import CountVectorizer
import string
from nltk.corpus import stopwords as stop_words
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

    def preprocess(self):
        """
        Note that if after filtering some documents do not contain words we remove them. That is why we return also the
        list of unpreprocessed documents.

        :return: preprocessed documents, unpreprocessed documents and the vocabulary list
        """
        preprocessed_docs_tmp = self.documents
        preprocessed_docs_tmp = [doc.lower() for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [doc.translate(
            str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 0 and w not in self.stopwords])
                             for doc in preprocessed_docs_tmp]

        vectorizer = CountVectorizer(max_features=self.vocabulary_size, token_pattern=r'\b[a-zA-Z]{2,}\b')
        vectorizer.fit_transform(preprocessed_docs_tmp)
        vocabulary = set(vectorizer.get_feature_names())
        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if w in vocabulary])
                                 for doc in preprocessed_docs_tmp]

        preprocessed_docs, unpreprocessed_docs = [], []
        for i, doc in enumerate(preprocessed_docs_tmp):
            if len(doc) > 0:
                preprocessed_docs.append(doc)
                unpreprocessed_docs.append(self.documents[i])

        return preprocessed_docs, unpreprocessed_docs, list(vocabulary)


class SimplePreprocessing(WhiteSpacePreprocessing):
    def __init__(self, documents, stopwords_language="english"):
        super().__init__(documents, stopwords_language)
        warnings.simplefilter('always', DeprecationWarning)

        if self.__class__.__name__ == "CTM":

            warnings.warn("SimplePrepocessing is deprecated and will be removed in version 2.0, "
                          "use WhiteSpacePreprocessing", DeprecationWarning)


