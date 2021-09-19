=================================
CombinedTM: Coherent Topic Models
=================================

Combined TM combines the BoW with SBERT, a process that seems to increase
the coherence of the predicted topics (https://aclanthology.org/2021.acl-short.96/).

Usage
=====

Here is how you can use the CombinedTM. This is a standard topic model that also uses contextualized embeddings. The good thing about CombinedTM is that it makes your topic much more coherent (see the paper https://arxiv.org/abs/2004.03974).
n_components=50 specifies the number of topics.

.. code-block:: python

    from contextualized_topic_models.models.ctm import CombinedTM
    from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
    from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file

    qt = TopicModelDataPreparation("paraphrase-distilroberta-base-v2")

    training_dataset = qt.fit(text_for_contextual=list_of_unpreprocessed_documents, text_for_bow=list_of_preprocessed_documents)

    ctm = CombinedTM(bow_size=len(qt.vocab), contextual_size=768, n_components=50) # 50 topics

    ctm.fit(training_dataset) # run the model

    ctm.get_topics()

Once the model is trained, it is very easy to get the topics!

.. code-block:: python

    ctm.get_topics()

Creating the Test Set
=====================

The **transform** method will take care of most things for you, for example the generation
of a corresponding BoW by considering only the words that the model has seen in training.

If you use **CombinedTM** you need to include the test text for the BOW:

.. code-block:: python

    testing_dataset = qt.transform(text_for_contextual=testing_text_for_contextual, text_for_bow=testing_text_for_bow)

    # n_sample how many times to sample the distribution (see the doc)
    ctm.get_doc_topic_distribution(testing_dataset, n_samples=20) # returns a (n_documents, n_topics) matrix with the topic distribution of each document

Warning
~~~~~~~

Note that the way we use the transform method here is different from what we do for `ZeroShotTM <https://contextualized-topic-models.readthedocs.io/en/latest/zeroshot.html>`_!
This is very important!


Tutorial
~~~~~~~~

.. |colab1_2| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1fXJjr_rwqvpp1IdNQ4dxqN4Dp88cxO97?usp=sharing
    :alt: Open In Colab

You can find a tutorial here: |colab1_2| it will show you how you can use CombinedTM.
