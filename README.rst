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

* Combines BERT and Neural Variational Topic Models
* Two different methodologies: combined, where we combine BoW and BERT embeddings and contextual, that uses only BERT embeddings
* Includes methods to create embedded representations and BoW
* Includes evaluation metrics


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
    from contextualized_topic_models.utils.data_preparation import embed_documents

    handler = TextHandler("documents.txt")
    handler.prepare() # create vocabulary and training data

    # generate BERT data
    training_bert = bert_embeddings_from_file("documents.txt", "distiluse-base-multilingual-cased")

    training_dataset = COTMDataset(handler.bow, training_bert, handler.idx2token)

    cotm = COTM(input_size=len(handler.vocab), bert_input_size=512, inference_type="contextual", n_components=50)

    cotm.fit(training_dataset) # run the model

See the example notebook in the `contextualized_topic_models/examples` folder. If you want you can also compute evaluate your topics using different measures,
for example coherence with the NPMI.

.. code-block:: python

    from contextualized_topic_models.evaluation.measures import CoherenceNPMI

    with open('documents.txt',"r") as fr:
        texts = [doc.split() for doc in fr.read().splitlines()] # load text for NPMI

    npmi = CoherenceNPMI(texts=texts, topics=cotm.get_topic_lists(10))
    npmi.score()


Predict topics for novel documents

.. code-block:: python


    test_handler = TextHandler("spanish_documents.txt")
    test_handler.prepare() # create vocabulary and training data

    # generate BERT data
    testing_bert = bert_embeddings_from_file("spanish_documents.txt", "distiluse-base-multilingual-cased")

    testing_dataset = COTMDataset(test_handler.bow, testing_bert, test_handler.idx2token)
    cotm.get_thetas(testing_dataset)

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
