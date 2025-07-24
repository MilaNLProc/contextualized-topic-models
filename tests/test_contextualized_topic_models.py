#!/usr/bin/env python

"""Tests for `contextualized_topic_models` package."""

from contextualized_topic_models.models.ctm import CTM, ZeroShotTM, CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
import os
from contextualized_topic_models.models.kitty_classifier import Kitty
import pytest
import nltk
import numpy as np

nltk.download("stopwords")

@pytest.fixture
def root_dir():
    return os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def data_dir(root_dir):
    return root_dir + "/../contextualized_topic_models/data/"


def test_labels_set(data_dir):

    with open(data_dir + '/gnews/GoogleNews.txt') as filino:
        data = filino.readlines()

    with open(data_dir + '/gnews/GoogleNews_LABEL.txt') as filino:
        labels = list(map(lambda x:x.replace("\n", ""), filino.readlines()))

    tp = TopicModelDataPreparation("distiluse-base-multilingual-cased")

    training_dataset = tp.fit(data[:100], data[:100], labels=labels[:100])

    ctm = CombinedTM(reduce_on_plateau=True, solver='sgd', bow_size=len(tp.vocab),
                     contextual_size=512, num_epochs=1, n_components=5, batch_size=2, label_size=len(set(labels[:100])))
    ctm.fit(training_dataset)


def test_kitty(data_dir):

    kt = Kitty()

    with open(data_dir + '/sample_text_document') as filino:
        training = filino.readlines()

    kt.train(training, topics=5, epochs=1, stopwords_list=["stop", "words"], embedding_model="paraphrase-distilroberta-base-v2")

    kt.assigned_classes = {0: "nature", 3: "shop/offices", 4: "sport"}

    tn = kt.transform(['beautiful sea in the ocean'], labels=['nature', 'shop/offices'])

    kt.predict(['beautiful sea in the ocean'], 5)

    kt.predict_topic(['beautiful sea in the ocean'], 5)

    assert len(tn) == 1


def test_preprocessing():

    testing_data = [" this is some documents \t", "  test  "]

    sp = WhiteSpacePreprocessing(testing_data, stopwords_language="english")
    preprocessed_documents, unpreprocessed_corpus, vocab = sp.preprocess()

    assert len(preprocessed_documents) == 2
    assert len(unpreprocessed_corpus) == 2
    assert len(vocab) >= 2


def test_validation_set(data_dir):

    with open(data_dir + '/gnews/GoogleNews.txt') as filino:
        data = filino.readlines()

    tp = TopicModelDataPreparation("distiluse-base-multilingual-cased")

    training_dataset = tp.fit(data[:100], data[:100])
    validation_dataset = tp.transform(data[100:105], data[100:105])

    ctm = ZeroShotTM(bow_size=len(tp.vocab), contextual_size=512, num_epochs=1, n_components=5, batch_size=2)
    ctm.fit(training_dataset, validation_dataset)


def test_training_all_classes_ctm(data_dir):

    with open(data_dir + 'sample_text_document') as filino:
        data = filino.readlines()

    tp = TopicModelDataPreparation("distiluse-base-multilingual-cased")

    training_dataset = tp.fit(data, data)
    ctm = ZeroShotTM(bow_size=len(tp.vocab), contextual_size=512, num_epochs=1, n_components=5, batch_size=2)
    ctm.fit(training_dataset)

    assert len(ctm.get_topics()) == 5

    ctm.get_topic_lists(25)

    thetas = ctm.get_doc_topic_distribution(training_dataset, n_samples=5)

    assert len(thetas) == len(data)

    predicted_topics = ctm.get_doc_topic_distribution(training_dataset, n_samples=5)

    assert len(predicted_topics) == len(data)

    ctm = CTM(bow_size=len(tp.vocab), contextual_size=512, num_epochs=1, n_components=5, batch_size=2)
    ctm.fit(training_dataset)

    assert len(ctm.get_topics()) == 5

    ctm.get_topic_lists(25)

    thetas = ctm.get_doc_topic_distribution(training_dataset, n_samples=5)

    assert len(thetas) == len(data)

    predicted_topics = ctm.get_doc_topic_distribution(training_dataset, n_samples=5)

    assert len(predicted_topics) == len(data)


def test_training_ctm_combined_labels(data_dir):

    with open(data_dir + '/gnews/GoogleNews.txt') as filino:
        data = filino.readlines()
    with open(data_dir + '/gnews/GoogleNews_LABEL.txt') as filino:
        labels = filino.readlines()

    tp = TopicModelDataPreparation("paraphrase-distilroberta-base-v2")

    training_dataset = tp.fit(data[:100], data[:100], labels=labels[:100])

    ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=768, num_epochs=1, n_components=5, batch_size=2,
                     label_size=len(set(labels[:100])))
    ctm.fit(training_dataset)

    assert len(ctm.get_topics()) == 5

    ctm.get_topic_lists(25)

    thetas = ctm.get_doc_topic_distribution(training_dataset, n_samples=5)

    assert len(thetas) == len(data[:100])






