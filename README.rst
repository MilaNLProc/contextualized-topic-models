===========================
Contextualized Topic Models
===========================


.. image:: https://img.shields.io/pypi/v/contextualized_topic_models.svg
        :target: https://pypi.python.org/pypi/contextualized_topic_models

.. image:: https://travis-ci.com/MilaNLProc/contextualized-topic-models.svg
        :target: https://travis-ci.com/MilaNLProc/contextualized-topic-models

.. image:: https://readthedocs.org/projects/contextualized-topic-models/badge/?version=latest
        :target: https://contextualized-topic-models.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


Contextualized Topic Models


* Free software: MIT license
* Documentation: https://contextualized-topic-models.readthedocs.io.

Super big shout-out to `Stephen Carrow`_ for creating the awesome https://github.com/estebandito22/PyTorchAVITM package
from which we constructed the foundations of this package. We are happy to redistribute again this software under the MIT License.


Features
--------

* TODO


Quick Guide
-----------

Install the package using pip

.. code-block:: bash

    pip install -U contextualized_topic_models


The contextual neural topic model can be easily instantiated using few parameters (although there is a wide range of parameters you can use to change the behaviour of the neural topic model. When you generate
embeddings with BERT remember that there is a maximum length and for documents that are too long some words will be ignored.

.. code-block:: python

    from contextualized_topic_models.models.cotm import COTM
    from contextualized_topic_models.utils.data_preparation import VocabAndTextFromFile
    from contextualized_topic_models.utils.data_preparation import get_bag_of_words
    from contextualized_topic_models.utils.data_preparation import embed_documents

    vocab_obj = TextHandler("text_file_one_doc_per_line.txt")

    vocab, training_ids, idx2token = vocab_obj.get_training() # create vocabulary and training data

    # generate BERT data
    training_bert = bert_embeddings_from_file("text_file_one_doc_per_line.txt", "distiluse-base-multilingual-cased")

    training_bow = get_bag_of_words(training_ids, len(vocab)) # create bag of words

    training_dataset = COTMDataset(training_bow, training_bert, idx2token)

    cotm = COTM(input_size=len(vocab), bert_input_size=512, inference_type="contextual", n_components=50) # run the model
    cotm.fit(training_dataset)


See the example notebook in the `contextualized_topic_models/examples` folder. If you want you can also compute evaluate your topics using different measures,
for example coherence with the NPMI.

.. code-block:: python

    from contextualized_topic_models.evaluation.measures import CoherenceNPMI

    with open('text_file_one_doc_per_line.txt',"r") as fr:
        texts = [doc.split() for doc in fr.read().splitlines()] # load text for NPMI

    npmi = CoherenceNPMI(texts=texts, topics=cotm.get_topic_lists(10))
    npmi.score()

Team
----

* Federico Bianchi <f.bianchi@unibocconi.it> Bocconi University
* Silvia Terragni <s.terragni4@campus.unimib.it> University of Milan-Bicocca
* Dirk Hovy <dirk.hovy@unibocconi.it> Bocconi University

Credits
-------


This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.
To ease the use of the library we have also incuded the `rbo`_ package, all the rights reserved to the author of that package.



.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _`Stephen Carrow` : https://github.com/estebandito22
.. _`rbo` : https://github.com/dlukes/rbo
