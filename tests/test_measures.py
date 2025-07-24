#!/usr/bin/env python

"""Tests for measures"""

import pytest
from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.evaluation.measures import CoherenceCV, CoherenceUMass, CoherenceNPMI, \
    InvertedRBO, TopicDiversity, TopicDiversityTF, Sil
import os

@pytest.fixture
def root_dir():
    return os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def data_dir(root_dir):
    return root_dir + "/../contextualized_topic_models/data/"

def test_diversities(data_dir):

    with open(data_dir + '/sample_text_document') as filino:
        data = filino.readlines()

    tp = TopicModelDataPreparation("distiluse-base-multilingual-cased")

    training_dataset = tp.fit(data, data)
    ctm = ZeroShotTM(bow_size=len(tp.vocab), contextual_size=512, num_epochs=1, n_components=5, batch_size=2)
    ctm.fit(training_dataset)

    td_1 = TopicDiversity(topk=25)
    topic_diversity_1 = td_1.score(ctm.get_topic_lists(5))

    assert topic_diversity_1 >= 0

    td_2 = TopicDiversityTF(topk=25)
    topic_diversity_2 = td_2.score(ctm.get_topic_lists(5))

    assert topic_diversity_2 >= 0

def test_coherences(data_dir):

    with open(data_dir + '/sample_text_document') as filino:
        training = filino.readlines()

    tp = TopicModelDataPreparation("distiluse-base-multilingual-cased")

    training_dataset = tp.fit(training, training)

    ctm = ZeroShotTM(bow_size=len(tp.vocab), contextual_size=512, num_epochs=1, n_components=5, batch_size=2)
    ctm.fit(training_dataset)

    topic_words = ctm.get_topic_lists(5)

    coherence_cv = CoherenceCV(texts=training, topk=3)
    cv = coherence_cv.score(topic_words)

    assert cv > -100

    coherence_npmi = CoherenceNPMI(texts=training, topk=3)
    npmi = coherence_npmi.score(topic_words)

    assert npmi > -100

    coherence_umass = CoherenceUMass(texts=training, topk=3)
    umass = coherence_umass.score(topic_words)

    assert umass > -100
