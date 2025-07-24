import pytest
import os

from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.evaluation.measures import (
    CoherenceNPMI, CoherenceWordEmbeddings, CoherenceCV,
    InvertedRBO, TopicDiversity)
from contextualized_topic_models.utils.data_preparation import (
    TopicModelDataPreparation)


@pytest.fixture
def root_dir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def data_dir(root_dir):
    return root_dir + "/../contextualized_topic_models/data/"


@pytest.fixture
def train_model(data_dir):
    with open(data_dir + 'gnews/GoogleNews.txt', 'r') as filino:
        data = filino.readlines()

    tp = TopicModelDataPreparation("distiluse-base-multilingual-cased")

    training_dataset = tp.fit(data, data)
    ctm = ZeroShotTM(
        bow_size=len(tp.vocab), contextual_size=512,
        num_epochs=2, n_components=5)
    ctm.fit(training_dataset)
    return ctm


def test_diversities(train_model):

    topics = train_model.get_topic_lists(25)

    irbo = InvertedRBO(topics=topics)
    score = irbo.score()
    assert 0 <= score <= 1

    td = TopicDiversity(topics=topics)
    score = td.score()
    assert 0 <= score <= 1


def test_coherences(data_dir, train_model):
    with open(data_dir + 'gnews/GoogleNews.txt', "r") as fr:
        texts = [doc.split() for doc in fr.read().splitlines()]

    topics = train_model.get_topic_lists(10)

    npmi = CoherenceNPMI(texts=texts, topics=topics)
    score = npmi.score()
    assert -1 <= score <= 1

    cv = CoherenceCV(texts=texts, topics=topics)
    score = cv.score()
    assert -1 <= score <= 1

    cwe = CoherenceWordEmbeddings(topics=topics)
    score = cwe.score()
    assert -1 <= score <= 1
