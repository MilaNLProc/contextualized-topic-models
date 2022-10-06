from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import KeyedVectors
import gensim.downloader as api
from scipy.spatial.distance import cosine
import abc

from contextualized_topic_models.evaluation.rbo import rbo
import numpy as np
import itertools


class Measure:
    def __init__(self):
        pass

    def score(self):
        pass


class TopicDiversity(Measure):
    def __init__(self, topics):
        super().__init__()
        self.topics = topics

    def score(self, topk=25):
        """
        :param topk: topk words on which the topic diversity will be computed
        :return:
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            unique_words = set()
            for t in self.topics:
                unique_words = unique_words.union(set(t[:topk]))
            td = len(unique_words) / (topk * len(self.topics))
            return td


class Coherence(abc.ABC):
    """
    :param topics: a list of lists of the top-k words
    :param texts: (list of lists of strings) represents the corpus on which
     the empirical frequencies of words are computed
    """
    def __init__(self, topics, texts):
        self.topics = topics
        self.texts = texts
        self.dictionary = Dictionary(self.texts)

    @abc.abstractmethod
    def score(self):
        pass


class CoherenceNPMI(Coherence):
    def __init__(self, topics, texts):
        super().__init__(topics, texts)

    def score(self, topk=10, per_topic=False):
        """
        :param topk: how many most likely words to consider in the evaluation
        :param per_topic: if True, returns the coherence value for each topic
         (default: False)
        :return: NPMI coherence
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            npmi = CoherenceModel(
                topics=self.topics, texts=self.texts,
                dictionary=self.dictionary,
                coherence='c_npmi', topn=topk)
            if per_topic:
                return npmi.get_coherence_per_topic()
            else:
                return npmi.get_coherence()


class CoherenceUMASS(Coherence):
    def __init__(self, topics, texts):
        super().__init__(topics, texts)

    def score(self, topk=10, per_topic=False):
        """
        :param topk: how many most likely words to consider in the evaluation
        :param per_topic: if True, returns the coherence value for each topic
         (default: False)
        :return: UMass coherence
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            umass = CoherenceModel(
                topics=self.topics, texts=self.texts,
                dictionary=self.dictionary,
                coherence='u_mass', topn=topk)
            if per_topic:
                return umass.get_coherence_per_topic()
            else:
                return umass.get_coherence()


class CoherenceUCI(Coherence):
    def __init__(self, topics, texts):
        super().__init__(topics, texts)

    def score(self, topk=10, per_topic=False):
        """
        :param topk: how many most likely words to consider in the evaluation
        :param per_topic: if True, returns the coherence value for each topic
         (default: False)
        :return: UCI coherence
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            uci = CoherenceModel(
                topics=self.topics, texts=self.texts,
                dictionary=self.dictionary,
                coherence='c_uci', topn=topk)
            if per_topic:
                return uci.get_coherence_per_topic()
            else:
                return uci.get_coherence()


class CoherenceCV(Coherence):
    def __init__(self, topics, texts):
        super().__init__(topics, texts)

    def score(self, topk=10, per_topic=False):
        """
        :param topk: how many most likely words to consider in the evaluation
        :param per_topic: if True, returns the coherence value for each topic
        (default: False)
        :return: C_V coherence
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            cv = CoherenceModel(
                topics=self.topics, texts=self.texts,
                dictionary=self.dictionary,
                coherence='c_v', topn=topk)
            if per_topic:
                return cv.get_coherence_per_topic()
            else:
                return cv.get_coherence()


class CoherenceWordEmbeddings(Measure):
    def __init__(self, topics, word2vec_path=None, binary=False):
        """
        :param topics: a list of lists of the top-n most likely words
        :param word2vec_path: if word2vec_file is specified, it retrieves the
         word embeddings file (in word2vec format) to compute similarities
         between words, otherwise 'word2vec-google-news-300' is downloaded
        :param binary: if the word2vec file is binary
        """
        super().__init__()
        self.topics = topics
        self.binary = binary
        if word2vec_path is None:
            self.wv = api.load('word2vec-google-news-300')
        else:
            self.wv = KeyedVectors.load_word2vec_format(
                word2vec_path, binary=binary)

    def score(self, topk=10):
        """
        :param topk: how many most likely words to consider in the evaluation
        :return: topic coherence computed on the word embeddings similarities
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            arrays = []
            for index, topic in enumerate(self.topics):
                if len(topic) > 0:
                    local_simi = []
                    for word1, word2 in itertools.combinations(
                            topic[:topk], 2):
                        if (word1 in self.wv.index_to_key
                                and word2 in self.wv.index_to_key):
                            local_simi.append(self.wv.similarity(word1, word2))
                    arrays.append(np.mean(local_simi))
            return np.mean(arrays)


class InvertedRBO(Measure):
    def __init__(self, topics):
        """
        :param topics: a list of lists of words
        """
        super().__init__()
        self.topics = topics

    def score(self, topk=10, weight=0.9):
        """
        :param weight: p (float), default 1.0: Weight of each agreement at
         depth d: p**(d-1). When set to 1.0, there is no weight, the rbo
         returns to average overlap.
        :return: rank_biased_overlap over the topics
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            collect = []
            for list1, list2 in itertools.combinations(self.topics, 2):
                rbo_val = rbo.rbo(list1[:topk], list2[:topk], p=weight)[2]
                collect.append(rbo_val)
            return 1 - np.mean(collect)


class Matches(Measure):
    def __init__(
        self, doc_distribution_original_language,
            doc_distribution_unseen_language):
        """
         :param doc_distribution_original_language: numpy array of the topical
         distribution of the documents in the original language
         (dim: num docs x num topics)
         :param doc_distribution_unseen_language: numpy array of the topical
          distribution of the documents in an unseen language
          (dim: num docs x num topics)
         """
        super().__init__()
        self.orig_lang_docs = doc_distribution_original_language
        self.unseen_lang_docs = doc_distribution_unseen_language
        if len(self.orig_lang_docs) != len(self.unseen_lang_docs):
            raise Exception(
                'Distributions of the comparable documents must'
                ' have the same length')

    def score(self):
        """
        :return: proportion of matches between the predicted topic in the
         original language and the predicted topic in the unseen language of
         the document distributions
        """
        matches = 0
        for d1, d2 in zip(self.orig_lang_docs, self.unseen_lang_docs):
            if np.argmax(d1) == np.argmax(d2):
                matches = matches + 1
        return matches/len(self.unseen_lang_docs)


class KLDivergence(Measure):
    def __init__(
        self, doc_distribution_original_language,
            doc_distribution_unseen_language):
        """
         :param doc_distribution_original_language: numpy array of the topical
         distribution of the documents in the original language
         (dim: num docs x num topics)
         :param doc_distribution_unseen_language: numpy array of the topical
          distribution of the documents in an unseen language
          (dim: num docs x num topics)
         """
        super().__init__()
        self.orig_lang_docs = doc_distribution_original_language
        self.unseen_lang_docs = doc_distribution_unseen_language
        if len(self.orig_lang_docs) != len(self.unseen_lang_docs):
            raise Exception(
                'Distributions of the comparable documents must'
                ' have the same length')

    def score(self):
        """
        :return: average kullback leibler divergence between the distributions
        """
        kl_mean = 0
        for d1, d2 in zip(self.orig_lang_docs, self.unseen_lang_docs):
            kl_mean = kl_mean + kl_div(d1, d2)
        return kl_mean/len(self.unseen_lang_docs)


def kl_div(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


class CentroidDistance(Measure):
    def __init__(
        self, doc_distribution_original_language,
        doc_distribution_unseen_language, topics, word2vec_path=None,
            binary=True, topk=10):
        """
         :param doc_distribution_original_language: numpy array of the topical
         distribution of the documents in the original language
         (dim: num docs x num topics)
         :param doc_distribution_unseen_language: numpy array of the topical
         distribution of the documents in an unseen language
         (dim: num docs x num topics)
         :param topics: a list of lists of the top-n most likely words
         :param word2vec_path: if word2vec_file is specified, it retrieves the
         word embeddings file (in word2vec format) to compute similarities
         between words, otherwise
         'word2vec-google-news-300' is downloaded
         :param binary: if the word2vec file is binary
         :param topk: max number of topical words
         """
        super().__init__()
        self.topics = [t[:topk] for t in topics]
        self.orig_lang_docs = doc_distribution_original_language
        self.unseen_lang_docs = doc_distribution_unseen_language
        if len(self.orig_lang_docs) != len(self.unseen_lang_docs):
            raise Exception(
                'Distributions of the comparable documents must'
                ' have the same length')

        if word2vec_path is None:
            self.wv = api.load('word2vec-google-news-300')
        else:
            self.wv = KeyedVectors.load_word2vec_format(
                word2vec_path, binary=binary)

    def score(self):
        """
        :return: average centroid distance between the words of the most
         likely topic of the document distributions
        """
        cd = 0
        for d1, d2 in zip(self.orig_lang_docs, self.unseen_lang_docs):
            top_words_orig = self.topics[np.argmax(d1)]
            top_words_unseen = self.topics[np.argmax(d2)]

            centroid_lang = self.get_centroid(top_words_orig)
            centroid_en = self.get_centroid(top_words_unseen)

            cd += (1 - cosine(centroid_lang, centroid_en))
        return cd/len(self.unseen_lang_docs)

    def get_centroid(self, word_list):
        vector_list = []
        for word in word_list:
            if word in self.wv.index_to_key:
                vector_list.append(self.wv.get_vector(word))
        vec = sum(vector_list)
        return vec / np.linalg.norm(vec)
