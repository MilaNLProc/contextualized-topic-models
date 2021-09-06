.. highlight:: shell

=====
Kitty
=====

Kitty is a utility to generate a simple classifiers from a topic model. It first run
a CTM instance on the data for you and you can then select a set of topics of interest. Once
this is done, you can apply this selection to a wider range of documents.

Usage
=====

.. code-block:: python

    from contextualized_topic_models.models.kitty_classifier import Kitty

    training = list(map(lambda x : x.strip(), open("train_data").readlines()))

    kt = Kitty()
    kt.train(training, 5)

    print(kt.pretty_print_word_classes())

This could probably output something like:

.. code-block:: shell

    0	family, plant, types, type, moth
    1	district, mi, area, village, west
    2	released, series, television, album, film
    3	school, station, historic, public, states
    4	born, football, team, played, season

You can then use a simple dictionary to assign the topics to some labels

.. code-block:: python

    kt.assigned_classes = {0 : "nature", 1 : "location", 2 : "entertainment", 3 : "shop/offices", 4: "sport"}

    kt.predict(["the village of Puza is a very nice village in Italy"])

    >> location

    kt.predict(["Pussetto is a soccer player that currently plays for Udiense Calcio"])

    >> sport

What Makes Kitty Different Other Topic Models?
==============================================

Nothing! it's just offer an user friendly utility that makes
use of the ZeroShotTM model in the backend.


