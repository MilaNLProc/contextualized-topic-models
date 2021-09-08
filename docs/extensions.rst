==============================
Extensions: SuperCTM and β-CTM
==============================

SuperCTM
========

Inspiration for SuperCTM has been taken directly from the work by `Card et al., 2018 <https://aclanthology.org/P18-1189/>`_ (you can read this as
"we essentially implemented their approach in our architecture"). SuperCTM should give better representations of the documents - this is somewhat expected, since we are using the labels to give more information to the model - and in theory should also make the model able to find topics more coherent with respect to the labels.
The model is super easy to use and requires minor modifications to the already implemented pipeline:


.. code-block:: python

    from contextualized_topic_models.models.ctm import ZeroShotTM
    from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation

    text_for_contextual = [
        "hello, this is unpreprocessed text you can give to the model",
        "have fun with our topic model",
    ]

    text_for_bow = [
        "hello unpreprocessed give model",
        "fun topic model",
    ]

    labels = [0, 1] # we need to have  a label for each document

    qt = TopicModelDataPreparation("paraphrase-multilingual-mpnet-base-v2")

    # training dataset should contain the labels
    training_dataset = qt.fit(text_for_contextual=text_for_contextual, text_for_bow=text_for_bow, labels=labels)

    # model should know the label size in advance
    ctm = CombinedTM(bow_size=len(qt.vocab), contextual_size=768, n_components=50, label_size=len(set(labels)))

    ctm.fit(training_dataset) # run the model

    ctm.get_topics(2)


β-CTM
=====

We also implemented the intuition found in the work by `Higgins et al., 2018 <https://openreview.net/forum?id=Sy2fzU9gl>`_, where a weight is applied
to the KL loss function. The idea is that giving more weight to the KL part of the loss function helps in creating disentangled representations
by forcing independence in the components. Again, the model should be straightforward to use:

.. code-block:: python

     ctm = CombinedTM(bow_size=len(qt.vocab), contextual_size=768, n_components=50, loss_weights={"beta" : 3})




