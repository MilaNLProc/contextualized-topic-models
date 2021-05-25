#!/usr/bin/env python

"""Tests for `contextualized_topic_models` package."""

from contextualized_topic_models.models.ctm import CTM, ZeroShotTM, CombinedTM
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file, bert_embeddings_from_list
import numpy as np
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing, SimplePreprocessing
import os
import pytest
import nltk


nltk.download("stopwords")

@pytest.fixture
def root_dir():
    return os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def data_dir(root_dir):
    return root_dir + "/../contextualized_topic_models/data/"


def test_validation_set(data_dir):

    with open(data_dir + '/gnews/GoogleNews.txt') as filino:
        data = filino.readlines()

    tp = TopicModelDataPreparation("distiluse-base-multilingual-cased")

    training_dataset = tp.fit(data[:100], data[:100])
    validation_dataset = tp.transform(data[100:105], data[100:105])

    ctm = CombinedTM(reduce_on_plateau=True, solver='sgd', bow_size=len(tp.vocab), contextual_size=512, num_epochs=100, n_components=5)
    ctm.fit(training_dataset, validation_dataset=validation_dataset, patience=5, save_dir=data_dir+'test_checkpoint')

    assert os.path.exists(data_dir+"test_checkpoint")


def test_training_all_classes_ctm(data_dir):

    with open(data_dir + 'sample_text_document') as filino:
        data = filino.readlines()

    tp = TopicModelDataPreparation("distiluse-base-multilingual-cased")

    training_dataset = tp.fit(data, data)
    ctm = ZeroShotTM(bow_size=len(tp.vocab), contextual_size=512, num_epochs=1, n_components=5)
    ctm.fit(training_dataset)  # run the model

    testing_dataset = tp.transform(data)
    predictions = ctm.get_doc_topic_distribution(testing_dataset, n_samples=2)

    assert len(predictions) == len(testing_dataset)

    topics = ctm.get_topic_lists(2)
    assert len(topics) == 5

    training_dataset = tp.fit(data, data)
    ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=512, num_epochs=1, n_components=5)
    ctm.fit(training_dataset)  # run the model

    topics = ctm.get_topic_lists(2)
    assert len(topics) == 5

    testing_dataset = tp.transform(data, data)
    predictions = ctm.get_doc_topic_distribution(testing_dataset, n_samples=2)

    assert len(predictions) == len(testing_dataset)


def test_preprocessing(data_dir):
    docs = [line.strip() for line in open(data_dir + "gnews/GoogleNews.txt", 'r').readlines()]
    sp = WhiteSpacePreprocessing(docs, "english")
    prep_corpus, unprepr_corpus, vocab = sp.preprocess()

    assert len(prep_corpus) == len(unprepr_corpus)  # prep docs must have the same size as the unprep docs
    assert len(prep_corpus) <= len(docs)  # preprocessed docs must be less than or equal the original docs

    assert len(vocab) <= sp.vocabulary_size  # check vocabulary size

    sp = SimplePreprocessing(docs)
    prep_corpus, unprepr_corpus, vocab = sp.preprocess()

    assert len(prep_corpus) == len(unprepr_corpus)  # prep docs must have the same size as the unprep docs
    assert len(prep_corpus) <= len(docs)  # preprocessed docs must be less than or equal the original docs

    assert len(vocab) <= sp.vocabulary_size  # check vocabulary size





