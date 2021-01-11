=====
Usage
=====

Frequently Asked Questions
--------------------------

Am I forced to use the SBERT Embeddings?
****************************************

Not at all! you can actually the embedding you like most. Keep in mind to check that the matrix you create
as the same number of documents (first dimension) as the Bag of Word representation and you are good to go!
Check the code `here <https://github.com/MilaNLProc/contextualized-topic-models/blob/master/contextualized_topic_models/utils/data_preparation.py#L25>`_
to see how we create the representations.


Is the BoW needed even for the ZeroShotTM?
******************************************

Yes, it is. The BoW is necessary in the reconstruction phase, without that we would lose the symbolic information
that allows us to get the topics.


