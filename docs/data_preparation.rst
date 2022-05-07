================
Data Preparation
================

One of the most fundamental pieces in CTM is the data preparation. This allows us to generate embeddings
and to use them to train the variational autoencoder.

Introduction
============

This entire process is regulated by the TopicModelDataPreparation class.

.. code-block:: python

    from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation

    # we first need to get an embedding model. This is based on distilroberta and trained on paraphrase
    # data.
    qt = TopicModelDataPreparation("paraphrase-distilroberta-base-v2")



Next, we need some data. This is just an example of the data we can pass to CTM:

.. code-block:: python

    text_for_contextual = [
        "hello, this is unpreprocessed text you can give to the model",
        "have fun with our topic model",
    ]

    text_for_bow = [
        "hello unpreprocessed give model",
        "fun topic model",
    ]

You see that we have two lists: the first one contains the original documents while the second one contains
the bag of words representation that is going to be used to generate our topic words. The original documents
are the ones that are passed to the embedding model to create the contextualized representations.

To generate the embeddings you can simply run

.. code-block:: python

        training_dataset = qt.fit(text_for_contextual=list_of_unpreprocessed_documents,
        text_for_bow=list_of_preprocessed_documents)

Using Custom Embeddings
=======================

Note that you can also use your own Custom Embeddings if you want. You just need to change
the way you fit the TopicModelDataPreparation object. Setting custom_embeddings to an array will skip the
use of the contextual model to generate the embeddings and the embedding you will pass will be used.

.. code-block:: python

        def fit(self, text_for_contextual, text_for_bow, labels=None, custom_embeddings=None):

SBERT
=====

Underneath we are using `SBERT <https://www.sbert.net>`_, you should now that some SBERT models truncate your document
to a max length. You should check `this <https://www.sbert.net/examples/applications/computing-embeddings/README.html#input-sequence-length>`_
if you want to know more.

Nonetheless, changing the max length in CTM is easy:

.. code-block:: python

    from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation

    # we first need to get an embedding model. This is based on distilroberta and trained on paraphrase
    # data.
    qt = TopicModelDataPreparation("paraphrase-distilroberta-base-v2", max_seq_length=200)
