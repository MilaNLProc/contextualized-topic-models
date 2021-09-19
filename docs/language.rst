=================================
Mono and Multi-Lingual Embeddings
=================================

Multilingual
~~~~~~~~~~~~

Some of the examples below use a multilingual embedding model :code:`paraphrase-multilingual-mpnet-base-v2`. This means that the representations you are going to use are mutlilinguals. However you might need a broader coverage of languages. In that case, you can check `SBERT`_ to find a model you can use.

English
~~~~~~~

If you are doing topic modeling in English, **you SHOULD use an English sentence-bert model**, for example `paraphrase-distilroberta-base-v2`. In that case,
it's really easy to update the code to support monolingual English topic modeling. If you need other models you can check `SBERT`_ for other models.

.. code-block:: python

    qt = TopicModelDataPreparation("paraphrase-distilroberta-base-v2")

Language-Specific
~~~~~~~~~~~~~~~~~

In general, our package should be able to support all the models described in the `sentence transformer package <https://github.com/UKPLab/sentence-transformers>`_ and in HuggingFace. You need to take a look at `HuggingFace models <https://huggingface.co/models>`_ and find which is the one for your language. For example, for Italian, you can use `UmBERTo`_. How to use this in the model, you ask? well, just use the name of the model you want instead of the english/multilingual one:

.. code-block:: python

    qt = TopicModelDataPreparation("Musixmatch/umberto-commoncrawl-cased-v1")


.. _SBERT: https://www.sbert.net/docs/pretrained_models.html
.. _UmBERTo: https://huggingface.co/Musixmatch/umberto-commoncrawl-cased-v1
