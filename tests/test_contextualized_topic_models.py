#!/usr/bin/env python

"""Tests for `contextualized_topic_models` package."""

from contextualized_topic_models.models.ctm import CTM, ZeroShotTM, CombinedTM
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file, bert_embeddings_from_list
import numpy as np
from contextualized_topic_models.utils.data_preparation import TextHandler
from contextualized_topic_models.utils.data_preparation import QuickText, TopicModelDataPreparation
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


def test_embeddings_from_scratch(data_dir):

    handler = TextHandler(data_dir + "sample_text_document")
    handler.prepare()  # create vocabulary and training data

    assert np.array_equal(handler.bow.todense(), np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2]]))

def test_training_all_classes_ctm(data_dir):
    handler = TextHandler(data_dir + "sample_text_document")
    handler.prepare()  # create vocabulary and training data

    train_bert = bert_embeddings_from_file(data_dir + 'sample_text_document',
                                           "distiluse-base-multilingual-cased")

    training_dataset = CTMDataset(handler.bow, train_bert, handler.idx2token)

    ctm = CTM(input_size=len(handler.vocab), bert_input_size=512, num_epochs=1, inference_type="combined",
              n_components=5)

    ctm.fit(training_dataset)  # run the model
    topics = ctm.get_topic_lists(2)
    assert len(topics) == 5

    thetas = ctm.get_doc_topic_distribution(training_dataset)
    assert len(thetas) == len(train_bert)

    ctm = ZeroShotTM(input_size=len(handler.vocab), bert_input_size=512, num_epochs=1,
                     n_components=5)
    ctm.fit(training_dataset)  # run the model
    topics = ctm.get_topic_lists(2)
    assert len(topics) == 5

    thetas = ctm.get_doc_topic_distribution(training_dataset)
    assert len(thetas) == len(train_bert)

    ctm = CombinedTM(input_size=len(handler.vocab), bert_input_size=512, num_epochs=1,
                     n_components=5)
    ctm.fit(training_dataset)  # run the model
    topics = ctm.get_topic_lists(2)
    assert len(topics) == 5

    thetas = ctm.get_doc_topic_distribution(training_dataset)
    assert len(thetas) == len(train_bert)

    with open(data_dir + 'sample_text_document') as filino:
        data = filino.readlines()

    handler = TextHandler(sentences=data)
    handler.prepare()  # create vocabulary and training data

    train_bert = bert_embeddings_from_list(data, "distiluse-base-multilingual-cased")
    training_dataset = CTMDataset(handler.bow, train_bert, handler.idx2token)

    ctm = CTM(input_size=len(handler.vocab), bert_input_size=512, num_epochs=1, inference_type="combined",
              n_components=5)

    ctm.fit(training_dataset)  # run the model
    topics = ctm.get_topic_lists(2)

    assert len(topics) == 5
    thetas = ctm.get_doc_topic_distribution(training_dataset)

    assert len(thetas) == len(train_bert)

    qt = QuickText("distiluse-base-multilingual-cased", text_for_bow=data, text_for_bert=data)

    dataset = qt.load_dataset()

    ctm = ZeroShotTM(input_size=len(qt.vocab), bert_input_size=512, num_epochs=1, n_components=5)
    ctm.fit(dataset)  # run the model
    topics = ctm.get_topic_lists(2)
    assert len(topics) == 5

    qt_from_conf = QuickText("distiluse-base-multilingual-cased", None, None)
    qt_from_conf.load_configuration(qt.bow, qt.data_bert, qt.vocab, qt.idx2token)
    dataset = qt_from_conf.load_dataset()

    ctm = ZeroShotTM(input_size=len(qt.vocab), bert_input_size=512, num_epochs=1, n_components=5)
    ctm.fit(dataset)  # run the model
    topics = ctm.get_topic_lists(2)
    assert len(topics) == 5

    tp = TopicModelDataPreparation("distiluse-base-multilingual-cased")

    training_dataset = tp.create_training_set(data, data)
    ctm = ZeroShotTM(input_size=len(tp.vocab), bert_input_size=512, num_epochs=1, n_components=5)
    ctm.fit(training_dataset)  # run the model

    topics = ctm.get_topic_lists(2)
    assert len(topics) == 5

    testing_dataset = tp.create_test_set(data)
    predictions = ctm.get_doc_topic_distribution(testing_dataset, n_samples=2)

    assert len(predictions) == len(testing_dataset)

    testing_dataset = tp.create_test_set(data, data)
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


