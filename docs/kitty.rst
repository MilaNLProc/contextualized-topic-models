.. highlight:: shell

========================================================================
Kitty: Human-in-the-loop Classification with Contextualized Topic Models
========================================================================

Kitty is a utility to generate a simple topic classifier from a topic model. It first runs
a CTM instance on the data for you and you can then select and label a set of topics of interest. Once
this is done, you can apply this selection to a wider range of documents.

Please cite the following papers if you use Kitty:

* Bianchi, F., Terragni, S., & Hovy, D. (2021). `Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence`. ACL. https://aclanthology.org/2021.acl-short.96/
* Bianchi, F., Terragni, S., Hovy, D., Nozza, D., & Fersini, E. (2021). `Cross-lingual Contextualized Topic Models with Zero-shot Learning`. EACL. https://www.aclweb.org/anthology/2021.eacl-main.143/

To simply put it, Kitty makes use of `ZeroShotTM <https://contextualized-topic-models.readthedocs.io/en/latest/zeroshot.html>`_ to extract topic from your data. Kitty runs some very simple
preprocessing on your data to remove words that might not be too useful. We also have a google colab tutorial.

.. |kitty_colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1ZO6y-laPMnIT6boMwNXK4WNiyAUWUK4L?usp=sharing
    :alt: Open In Colab

|kitty_colab|


Usage
=====

In this example we use an english embedding model, however you might need langauge-specific models to do this, check out the `related section of the documentation <https://contextualized-topic-models.readthedocs.io/en/latest/language.html>`_


.. code-block:: python

    from contextualized_topic_models.models.kitty_classifier import Kitty

    # read the training data
    training_set = list(map(lambda x : x.strip(), open("train_data").readlines()))

    kt = Kitty()
    kt.train(training, topics=5, epochs=1, stopwords_list=["stop", "words"], embedding_model="paraphrase-distilroberta-base-v2")

    print(kt.pretty_print_word_classes())

This could probably output topics like these ones:

.. code-block:: shell

    0	family, plant, types, type, moth
    1	district, mi, area, village, west
    2	released, series, television, album, film
    3	school, station, historic, public, states
    4	born, football, team, played, season

Now, you can then use a simple dictionary to assign the topics to some labels. For
example, topic 0 seems to be describing nature related things.

.. code-block:: python

    kt.assigned_classes = {0 : "nature", 1 : "location",
                           2 : "entertainment", 3 : "shop/offices", 4: "sport"}

    kt.predict(["the village of Puza is a very nice village in Italy"])

    >> location

    kt.predict(["Pussetto is a soccer player that currently plays for Udiense Calcio"])

    >> sport


If you are using a jupyter notebook, you can use the widget to fill in the labels.

.. code-block:: python

    kt.widget_annotation()

Cross-Lingual Support
=====================

A nice feature of Kitty is that it can be used to filter documents in different
languages. Assume you have access to a large corpus of Italian documents and
a smaller corpus of English documents. You can run Kitty on the English documents,
map the labels and apply Kitty on the Italian documents. It is enough to change the
embedding model.

.. code-block:: python

    from contextualized_topic_models.models.kitty_classifier import Kitty

    # read the training data
    training = list(map(lambda x : x.strip(), open("train_data").readlines()))

    # define kitty with a multilingual embedding model
    kt = Kitty(embedding_model="paraphrase-multilingual-mpnet-base-v2",  contextual_size=768)

    kt.train(training, 5, stopwords_list=["stopwords"]) # train a topic model with 5 topics

    print(kt.pretty_print_word_classes())

You can then apply the mapping as we did before and predict in different languages:

.. code-block:: python

    kt.predict(["Pussetto è un calciatore che attualmente gioca per l'Udinese Calcio"])

    >> sport

You should refer to `SBERT Pretrained Models <https://www.sbert.net/docs/pretrained_models.html>`_ to know
if the languages you want to use are supported by SBERT.



Using Custom Embeddings with Kitty
===================================

Do you have custom embeddings and want to use them for faster results? Just give them to Kitty!

.. code-block:: python

    from contextualized_topic_models.models.kitty_classifier import Kitty
    import numpy as np

    # read the training data
    training_data = list(map(lambda x : x.strip(), open("train_data").readlines()))
    customer_embeddings = np.load('customer_embeddings.npy')

    kt = Kitty()
    kt.train(training_data, custom_embeddings=customer_embeddings, stopwords_list=["stopwords"])

    print(kt.pretty_print_word_classes())


Note: Custom embeddings must be numpy.arrays.


What Makes Kitty Different from Other Topic Models?
====================================================

Nothing! It just offers a user-friendly utility that makes use of the ZeroShotTM model in the backend.


