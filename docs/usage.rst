=====
Usage
=====

Frequently Asked Questions
--------------------------

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
If your objective is to do `cross-lingual topic modeling`_ (i.e. train a topic model on a dataset in one language and predict the topics for data in other languages), then ZeroShotTM is the model for you. If you just aim at extracting topics from a corpus, you can use either the CombinedTM or the ZeroShotTM. We have designed the Combined topic model for the purpose of obtaining more coherent topics, so we suggest you use this. Yet, as you can read in `this paper <https://www.aclweb.org/anthology/2021.eacl-main.143/>`_, the ZeroShotTM model still gets results that are very similar to the ones of the combinedTM.   



How do I choose the correct number of topics?
***********************************************

The "n_component" parameter represents the number of topics for CTM. There is not a "right" answer about the choice of the number of topics. Usually, researchers try a different number of topics (10, 30, 50, etc, depending on the prior knowledge on the dataset) and select the number of topics that guarantees the highest average `topic coherence`_. We also suggest you take into consideration the `topic diversity`_. 

.. _topic coherence: https://github.com/MilaNLProc/contextualized-topic-models/blob/cb495ca29f73a6d01fbe4ff7bc5b746b2716a593/contextualized_topic_models/evaluation/measures.py#L56
.. _topic diversity: https://github.com/MilaNLProc/contextualized-topic-models/blob/cb495ca29f73a6d01fbe4ff7bc5b746b2716a593/contextualized_topic_models/evaluation/measures.py#L159
.. _cross-lingual topic modeling: https://github.com/MilaNLProc/contextualized-topic-models#cross-lingual-topic-modeling
