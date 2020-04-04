from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import KeyedVectors
import gensim.downloader as api

from .rbo import rbo
import numpy as np
import itertools

def compute_topic_diversity(topics, topk=25):
    '''
    :param topics:  a list of lists of the top-k words
    :param topk: how many most likely words to consider in the evaluation (default: 25)
    :return: topic diversity
    '''
    unique_words = set()
    for t in topics:
        unique_words = unique_words.union(set(t[:topk]))
    td = len(unique_words) / (topk * len(topics))
    return td


def compute_npmi(topics, texts, topk=10):
    '''
    :param topics: a list of lists of the top-k words
    :param texts: (list of lists of strings) represents the corpus on which the empirical frequencies of words are
    computed
    :param topk: how many most likely words to consider in the evaluation
    :return:
    '''
    dictionary = Dictionary(texts)
    npmi = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary,
                          coherence='c_npmi', topn=topk)
    return npmi.get_coherence()

def compute_word_embeddings_coherence(topics, topk=10, word2vec_file=None, binary=False):
    '''
    :param topics: a list of lists of the top-n most likely words
    :param topk: how many most likely words to consider in the evaluation
    :param word2vec_file: if word2vec_file is specified, it retrieves the word embeddings file (in word2vec format) to
     compute similarities between words, otherwise 'word2vec-google-news-300' is downloaded
    :param binary: if the word2vec file is binary
    :return: topic coherence computed on the word embeddings similarities
    '''
    if word2vec_file is None:
        wv = api.load('word2vec-google-news-300')
    else:
        wv = KeyedVectors.load_word2vec_format(word2vec_file, binary=binary)
    arrays = []
    for index, topic in enumerate(topics):
        if len(topic) > 0:
            local_simi = []
            for word1, word2 in itertools.combinations(topic[0:topk], 2):
                if word1 in wv.vocab and word2 in wv.vocab:
                    local_simi.append(wv.similarity(word1, word2))
            arrays.append(np.mean(local_simi))
    return np.mean(arrays)


def compute_rank_biased_overlap(topic_list, topk = 10, weight=0.9):
    '''
    :param weight: p (float), default 1.0: Weight of each agreement at depth d:
    p**(d-1). When set to 1.0, there is no weight, the rbo returns to average overlap.
    :param topic_list: a list of lists of words
    :return: rank_biased_overlap over the topics
    '''
    collect = []
    for list1, list2 in itertools.combinations(topic_list, 2):
        rbo_val = rbo(list1[:topk], list2[:topk], p=weight)[2]
        collect.append(rbo_val)
    return np.mean(collect)
