==========
Evaluation
==========

We have also included some of the metrics normally used in the evaluation of topic models, for example you can compute the coherence of your
topics using NPMI using our simple and high-level API.

Metrics Covered
===============

The metrics we cover are the one we also describe in our papers.

+ Coherence (e.g., NPMI, Word Embeddings)
+ Topic Diversity (e.g., Inversed RBO)
+ Matches
+ Centroid Distance

You can take a look at the `evaluation measures file <https://github.com/MilaNLProc/contextualized-topic-models/blob/master/contextualized_topic_models/evaluation/measures.py>`_
to get a better idea on how to add them to your code.

Example
=======

If you want to compute the NPMI coherence you can simply do as follows:

.. code-block:: python

    from contextualized_topic_models.evaluation.measures import CoherenceNPMI

    with open('preprocessed_documents.txt', "r") as fr:
        texts = [doc.split() for doc in fr.read().splitlines()] # load text for NPMI

    npmi = CoherenceNPMI(texts=texts, topics=ctm.get_topic_lists(10))
    npmi.score()
