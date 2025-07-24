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

    topic = kt.predict(["test sentence"])

    assert topic[0] in kt.assigned_classes.values()

    kt.pretty_print_word_classes()


def test_custom_embeddings(data_dir):

    with open(data_dir + "/custom_embeddings/sample_text.txt") as filino:
        training = filino.read().splitlines()

    embeddings = np.load(data_dir + "/custom_embeddings/sample_embeddings.npy")

    turkish_stopwords = nltk.corpus.stopwords.words('turkish')

    kt = Kitty()
    kt.train(training, custom_embeddings=embeddings, topics=5, epochs=1,
             stopwords_list=turkish_stopwords, hidden_sizes=(200, 200))


def test_validation_set(data_dir):

    with open(data_dir + '/gnews/GoogleNews.txt') as filino:
        data = filino.readlines()

    tp = TopicModelDataPreparation("distiluse-base-multilingual-cased")

    training_dataset = tp.fit(data[:100], data[:100])
    validation_dataset = tp.transform(data[100:105], data[100:105])

    ctm = CombinedTM(reduce_on_plateau=True, solver='sgd',  batch_size=2, bow_size=len(tp.vocab), contextual_size=512, num_epochs=1, n_components=5)
    ctm.fit(training_dataset, validation_dataset=validation_dataset, patience=5, save_dir=data_dir+'test_checkpoint')

    assert os.path.exists(data_dir+"test_checkpoint")


def test_training_all_classes_ctm(data_dir):

    with open(data_dir + 'sample_text_document') as filino:
        data = filino.readlines()

    tp = TopicModelDataPreparation("distiluse-base-multilingual-cased")

    training_dataset = tp.fit(data, data)
    ctm = ZeroShotTM(bow_size=len(tp.vocab), contextual_size=512, num_epochs=1, n_components=5, batch_size=2)
    ctm.fit(training_dataset)  # run the model

    testing_dataset = tp.transform(data)
    predictions = ctm.get_doc_topic_distribution(testing_dataset, n_samples=2)

    assert len(predictions) == len(testing_dataset)

    topics = ctm.get_topic_lists(2)
    assert len(topics) == 5

    training_dataset = tp.fit(data, data)
    ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=512, num_epochs=1, n_components=5, batch_size=2)
    ctm.fit(training_dataset)  # run the model

    topics = ctm.get_topic_lists(2)
    assert len(topics) == 5

    ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=512, num_epochs=1, n_components=5,loss_weights={"beta": 10}, batch_size=2)
    ctm.fit(training_dataset)  # run the model
    assert ctm.weights == {"beta": 10}

    topics = ctm.get_topic_lists(2)
    assert len(topics) == 5

    testing_dataset = tp.transform(data, data)
    predictions = ctm.get_doc_topic_distribution(testing_dataset, n_samples=2)

    assert len(predictions) == len(testing_dataset)


def test_preprocessing(data_dir):
    docs = [line.strip() for line in open(data_dir + "gnews/GoogleNews.txt", 'r').readlines()]
    sp = WhiteSpacePreprocessing(docs, "english")
    prep_corpus, unprepr_corpus, vocab, retained_indices = sp.preprocess()

    assert len(prep_corpus) == len(unprepr_corpus)  # prep docs must have the same size as the unprep docs
    assert len(prep_corpus) <= len(docs)  # preprocessed docs must be less than or equal the original docs

    assert len(vocab) <= sp.vocabulary_size  # check vocabulary size






