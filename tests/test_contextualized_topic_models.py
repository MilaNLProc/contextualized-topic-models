#!/usr/bin/env python

"""Tests for `contextualized_topic_models` package."""

from contextualized_topic_models.models.ctm import CTM
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file, bert_embeddings_from_list
import numpy as np
import pickle
from contextualized_topic_models.utils.data_preparation import TextHandler
from contextualized_topic_models.datasets.dataset import CTMDataset

import os
import pytest

@pytest.fixture
def root_dir():
    return os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def data_dir(root_dir):
    return root_dir + "/../contextualized_topic_models/data/"

def test_shape_checks(data_dir):
    handler = TextHandler(data_dir +"gnews/GoogleNews.txt")
    handler.prepare()  # create vocabulary and training data

    # load BERT data
    with open(data_dir +"gnews/bert_embeddings_gnews", "rb") as filino:
        training_bert = pickle.load(filino)

    assert len(training_bert) == len(handler.bow.todense())

def test_training_with_saved_data(data_dir):
    handler = TextHandler(data_dir + "gnews/GoogleNews.txt")
    handler.prepare()  # create vocabulary and training data

    # load BERT data
    with open(data_dir +"gnews/bert_embeddings_gnews", "rb") as filino:
        training_bert = pickle.load(filino)

    training_dataset = CTMDataset(handler.bow, training_bert, handler.idx2token)

    ctm = CTM(input_size=len(handler.vocab), bert_input_size=768, num_epochs=1, inference_type="combined",
              n_components=5)

    ctm.fit(training_dataset)  # run the model

    print(ctm.get_topics(2))

    ctm.get_thetas(training_dataset)


def test_embeddings_from_scratch(data_dir):

    handler = TextHandler(data_dir + "sample_text_document")
    handler.prepare()  # create vocabulary and training data

    assert np.array_equal(handler.bow.todense(), np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2]]))

    train_bert = bert_embeddings_from_file(data_dir + 'sample_text_document', "distiluse-base-multilingual-cased")

    assert len(train_bert) == len(handler.bow.todense())

def test_training(data_dir):
    handler = TextHandler(data_dir + "sample_text_document")
    handler.prepare()  # create vocabulary and training data

    train_bert = bert_embeddings_from_file(data_dir + 'sample_text_document', "distiluse-base-multilingual-cased")

    training_dataset = CTMDataset(handler.bow, train_bert, handler.idx2token)

    ctm = CTM(input_size=len(handler.vocab), bert_input_size=512, num_epochs=1, inference_type="combined",
              n_components=5)

    ctm.fit(training_dataset)  # run the model

    topics = (ctm.get_topic_lists(2))

    assert len(topics) == 5

    thetas = ctm.get_thetas(training_dataset)

    assert len(thetas) == len(train_bert)

