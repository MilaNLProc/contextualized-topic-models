============================================================================
ZeroShotTM: Topic Modeling with Missing Words and Cross-Lingual Capabilities
============================================================================


Our ZeroShotTM can be used for zero-shot topic modeling. This is because the entire document is encoded
into an embedding using a contextualized model. Thus, we are not limited by the usual problems you might
encounter with bag of words: at test time, words that are missing from the training set will be
encoded using the contextualized model, thus providing a reliable topic model even in sparse context!

More interestingly, this model can be used for cross-lingual topic modeling (See next sections)!
You can also read  paper (https://www.aclweb.org/anthology/2021.eacl-main.143)

Usage
=====

.. code-block:: python

    from contextualized_topic_models.models.ctm import ZeroShotTM
    from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
    from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file

    text_for_contextual = [
        "hello, this is unpreprocessed text you can give to the model",
        "have fun with our topic model",
    ]

    text_for_bow = [
        "hello unpreprocessed give model",
        "fun topic model",
    ]

    qt = TopicModelDataPreparation("paraphrase-multilingual-mpnet-base-v2")

    training_dataset = qt.fit(text_for_contextual=text_for_contextual, text_for_bow=text_for_bow)

    ctm = ZeroShotTM(bow_size=len(qt.vocab), contextual_size=768, n_components=50)

    ctm.fit(training_dataset) # run the model

    ctm.get_topics(2)


As you can see, the high-level API to handle the text is pretty easy to use;
**text_for_bert** should be used to pass to the model a list of documents that are not preprocessed.
Instead, to **text_for_bow** you should pass the preprocessed text used to build the BoW.

Once the model is trained, it is very easy to get the topics!

.. code-block:: python

    ctm.get_topics()

Creating the Test Set
=====================

The **transform** method will take care of most things for you, for example the generation
of a corresponding BoW by considering only the words that the model has seen in training.
However, this comes with some bumps when dealing with the ZeroShotTM, as we will se in the next section.

If you use **ZeroShotTM** you do not need to use the `testing_text_for_bow` because if you are using
a different set of test documents, this will create a BoW of a different size. Thus, the best
way to do this is to pass just the text that is going to be given in input to the contexual model:

.. code-block:: python

    testing_dataset = qt.transform(text_for_contextual=testing_text_for_contextual)

    # n_sample how many times to sample the distribution (see the doc)
    ctm.get_doc_topic_distribution(testing_dataset, n_samples=20)

Warning
~~~~~~~

Note that the way we use the transform method here is different from what we do for `CombinedTM <https://contextualized-topic-models.readthedocs.io/en/latest/combined.html>`_!
This is very important!

Cross-Lingual Topic Modeling
============================

Once you have trained the ZeroShotTM model with multilingual embeddings,
you can use this simple pipeline to predict the topics for documents in a different language (as long as this language
is covered by **paraphrase-multilingual-mpnet-base-v2**).

.. code-block:: python

    # here we have a Spanish document
    testing_text_for_contextual = [
        "hola, bienvenido",
    ]

    # since we are doing multilingual topic modeling, we do not need the BoW in
    # ZeroShotTM when doing cross-lingual experiments (it does not make sense, since we trained with an english Bow
    # to use the spanish BoW)
    testing_dataset = qt.transform(testing_text_for_contextual)

    # n_sample how many times to sample the distribution (see the doc)
    ctm.get_doc_topic_distribution(testing_dataset, n_samples=20) # returns a (n_documents, n_topics) matrix with the topic distribution of each document

**Advanced Notes:** We do not need to pass the Spanish bag of word: the bag of words of the two languages will not be comparable! We are passing it to the model for compatibility reasons, but you cannot get
the output of the model (i.e., the predicted BoW of the trained language) and compare it with the testing language one.
