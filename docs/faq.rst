==========================
Frequently Asked Questions
==========================

I am getting very poor results. What can I do?
***********************************************
There are many elements that can influence the final results in a topic model.
A good preprocessing is fundamental for obtaining meaningful topics.
`On this link <https://github.com/MilaNLProc/contextualized-topic-models#tldr>`_
you can find some suggestions on how to preprocess your data,
but be careful because each dataset may have their pecularities.
If you still get poor results, don't hesitate to `contact us <https://github.com/MilaNLProc/contextualized-topic-models#development-team>`_! We would be happy to help you :)


Am I forced to use the SBERT Embeddings?
****************************************

Not at all! you can actually the embedding you like most. Keep in mind to check that the matrix you create
as the same number of documents (first dimension) as the Bag of Word representation and you are good to go!
Check the code `here <https://github.com/MilaNLProc/contextualized-topic-models/blob/master/contextualized_topic_models/utils/data_preparation.py>`_
to see how we create the representations.


Is the BoW needed even for the ZeroShotTM?
******************************************

Yes, it is. The BoW is necessary in the reconstruction phase, without that we would lose the symbolic information
that allows us to get the topics.


ZeroShotTM or CombinedTM? Which one should I use?
*************************************************

ZeroShotTm and CombinedTM can be basically used for the same tasks. ZeroShotTM has two main pros:

1) it can handle unseen words in the test phase. This makes it very useful to be used in our
Kitty module, for example.

2) If your objective is to do `cross-lingual topic modeling`_
(i.e. train a topic model on a dataset in one language and predict the topics for data in other languages),
then ZeroShotTM is the model for you.

If you just aim at extracting topics from a corpus, you can use either the CombinedTM or the ZeroShotTM.
We have designed the CombinedTM for the purpose of obtaining more coherent topics,
so we suggest you use this for more general topic extraction.
Yet, as you can read in `this paper <https://www.aclweb.org/anthology/2021.eacl-main.143/>`_,
the ZeroShotTM model still gets results that are very similar to the ones of the CombinedTM.

Can I load my own embeddings?
*****************************

Sure, here is a snippet that can help you. You need to create the embeddings (for bow and contextualized) and you also need
to have the vocab and an id2token dictionary (maps integers ids to words).

.. code-block:: python

    qt = TopicModelDataPreparation()

    training_dataset = qt.load(contextualized_embeddings, bow_embeddings, id2token)
    ctm = CombinedTM(bow_size=len(vocab), contextual_size=768, n_components=50)
    ctm.fit(training_dataset) # run the model
    ctm.get_topics()

You can give a look at the code we use in the TopicModelDataPreparation object to get an idea on how to create everything from scratch.
For example:

.. code-block:: python

        vectorizer = CountVectorizer() #from sklearn

        train_bow_embeddings = vectorizer.fit_transform(text_for_bow)
        train_contextualized_embeddings = bert_embeddings_from_list(text_for_contextual, "chosen_contextualized_model")
        vocab = vectorizer.get_feature_names_out()
        id2token = {k: v for k, v in zip(range(0, len(vocab)), vocab)}


How do I choose the correct number of topics?
*********************************************

The "n_component" parameter represents the number of topics for CTM. There is not a "right" answer about the choice of the number of topics. Usually, researchers try a different number of topics (10, 30, 50, etc, depending on the prior knowledge on the dataset) and select the number of topics that guarantees the highest average `topic coherence`_. We also suggest you take into consideration the `topic diversity`_.

.. _topic coherence: https://github.com/MilaNLProc/contextualized-topic-models/blob/cb495ca29f73a6d01fbe4ff7bc5b746b2716a593/contextualized_topic_models/evaluation/measures.py#L56
.. _topic diversity: https://github.com/MilaNLProc/contextualized-topic-models/blob/cb495ca29f73a6d01fbe4ff7bc5b746b2716a593/contextualized_topic_models/evaluation/measures.py#L159
.. _cross-lingual topic modeling: https://github.com/MilaNLProc/contextualized-topic-models#cross-lingual-topic-modeling
