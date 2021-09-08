==========
CombinedTM
==========

Combined TM combines the BoW with SBERT, a process that seems to increase
the coherence of the predicted topics (https://arxiv.org/pdf/2004.03974.pdf).

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



