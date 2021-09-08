.. highlight:: shell

========================================================================
Kitty: Human-in-the-loop Classification with Contextualized Topic Models
========================================================================

Kitty is a utility to generate a simple topic classifier from a topic model. It first runs
a CTM instance on the data for you and you can then select and label a set of topics of interest. Once
this is done, you can apply this selection to a wider range of documents.

Usage
=====

.. code-block:: python

    from contextualized_topic_models.models.kitty_classifier import Kitty
    
    # read the training data
    training_set = list(map(lambda x : x.strip(), open("train_data").readlines()))

    kt = Kitty(language="english")
    kt.train(training_set, 5) # train a topic model with 5 topics

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

    kt.assigned_classes = {0 : "nature", 1 : "location", 2 : "entertainment", 3 : "shop/offices", 4: "sport"}

    kt.predict(["the village of Puza is a very nice village in Italy"])

    >> location

    kt.predict(["Pussetto is a soccer player that currently plays for Udiense Calcio"])

    >> sport

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

    kt.train(training, 5) # train a topic model with 5 topics

    print(kt.pretty_print_word_classes())

You can then apply the mapping as we did before and predict in different languages:

.. code-block:: python

    kt.predict(["Pussetto è un calciatore che attualmente gioca per l'Udinese Calcio"])

    >> sport

You should refer to `SBERT Pretrained Models <https://www.sbert.net/docs/pretrained_models.html>`_ to know
if the languages you want to use are supported by SBERT.

What Makes Kitty Different Other Topic Models?
==============================================

Nothing! It just offers a user-friendly utility that makes use of the ZeroShotTM model in the backend.


