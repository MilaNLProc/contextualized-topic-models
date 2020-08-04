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

.. image:: https://img.shields.io/github/contributors/MilaNLProc/contextualized-topic-models
        :target: https://github.com/MilaNLProc/contextualized-topic-modelsgraphs/contributors/
        :alt: Contributors

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
        :target: https://lbesson.mit-license.org/
        :alt: License

.. image:: https://pepy.tech/badge/contextualized-topic-models/month
        :target: https://pepy.tech/project/contextualized-topic-models/month
        :alt: Downloads


Contextualized Topic Models (CTM) are a family of topic models that use pre-trained representations of language (e.g., BERT) to
support topic modeling. See the papers for details:

* `Cross-lingual Contextualized Topic Models with Zero-shot Learning` https://arxiv.org/pdf/2004.07737v1.pdf
* `Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence` https://arxiv.org/pdf/2004.03974.pdf

Combined Topic Model
--------------------

.. image:: https://raw.githubusercontent.com/MilaNLProc/contextualized-topic-models/master/img/lm_topic_model.png
   :target: https://raw.githubusercontent.com/MilaNLProc/contextualized-topic-models/master/img/lm_topic_model.png
   :align: center
   :width: 400px

Fully Contextual Topic Model
----------------------------

.. image:: https://raw.githubusercontent.com/MilaNLProc/contextualized-topic-models/master/img/lm_topic_model_multilingual.png
   :target: https://raw.githubusercontent.com/MilaNLProc/contextualized-topic-models/master/img/lm_topic_model_multilingual.png
   :align: center
   :width: 400px

Software details:

* Free software: MIT license
* Documentation: https://contextualized-topic-models.readthedocs.io.
* Super big shout-out to `Stephen Carrow`_ for creating the awesome https://github.com/estebandito22/PyTorchAVITM package from which we constructed the foundations of this package. We are happy to redistribute again this software under the MIT License.



Features
--------

* Combines BERT and Neural Variational Topic Models
* Two different methodologies: combined, where we combine BoW and BERT embeddings and contextual, that uses only BERT embeddings
* Includes methods to create embedded representations and BoW
* Includes evaluation metrics


Overview
--------

Install the package using pip

.. code-block:: bash

    pip install -U contextualized_topic_models


The contextual neural topic model can be easily instantiated using few parameters (although there is a wide range of
parameters you can use to change the behaviour of the neural topic model). When you generate
embeddings with BERT remember that there is a maximum length and for documents that are too long some words will be ignored.

An important aspect to take into account is which network you want to use: the one that combines BERT and the BoW or the one that just uses BERT.
It's easy to swap from one to the other:

Combined Topic Model:

.. code-block:: python

    CTM(input_size=len(handler.vocab), bert_input_size=512, inference_type="combined", n_components=50)

Fully Contextual Topic Model:

.. code-block:: python

    CTM(input_size=len(handler.vocab), bert_input_size=512, inference_type="contextual", n_components=50)




Contextual Topic Modeling
-------------------------

Here is how you can use the combined topic model. The high level API is pretty easy to use:

.. code-block:: python

    from contextualized_topic_models.models.ctm import CTM
    from contextualized_topic_models.utils.data_preparation import TextHandler
    from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file
    from contextualized_topic_models.datasets.dataset import CTMDataset

    handler = TextHandler("documents.txt")
    handler.prepare() # create vocabulary and training data

    # generate BERT data
    training_bert = bert_embeddings_from_file("documents.txt", "distiluse-base-multilingual-cased")

    training_dataset = CTMDataset(handler.bow, training_bert, handler.idx2token)

    ctm = CTM(input_size=len(handler.vocab), bert_input_size=512, inference_type="combined", n_components=50)

    ctm.fit(training_dataset) # run the model

See the example notebook in the `contextualized_topic_models/examples` folder.
We have also included some of the metrics normally used in the evaluation of topic models, for example you can compute the coherence of your
topics using NPMI using our simple and high-level API.

.. code-block:: python

    from contextualized_topic_models.evaluation.measures import CoherenceNPMI

    with open('documents.txt',"r") as fr:
        texts = [doc.split() for doc in fr.read().splitlines()] # load text for NPMI

    npmi = CoherenceNPMI(texts=texts, topics=ctm.get_topic_lists(10))
    npmi.score()


Cross-lingual Topic Modeling
----------------------------

The fully contextual topic model can be used for cross-lingual topic modeling! See the paper (https://arxiv.org/pdf/2004.07737v1.pdf)


.. code-block:: python

    from contextualized_topic_models.models.ctm import CTM
    from contextualized_topic_models.utils.data_preparation import TextHandler
    from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file
    from contextualized_topic_models.datasets.dataset import CTMDataset

    handler = TextHandler("english_documents.txt")
    handler.prepare() # create vocabulary and training data

    training_bert = bert_embeddings_from_file("documents.txt", "distiluse-base-multilingual-cased")

    training_dataset = CTMDataset(handler.bow, training_bert, handler.idx2token)

    ctm = CTM(input_size=len(handler.vocab), bert_input_size=512, inference_type="contextual", n_components=50)

    ctm.fit(training_dataset) # run the model


Predict topics for novel documents

.. code-block:: python


    test_handler = TextHandler("spanish_documents.txt")
    test_handler.prepare() # create vocabulary and training data

    # generate BERT data
    testing_bert = bert_embeddings_from_file("spanish_documents.txt", "distiluse-base-multilingual-cased")

    testing_dataset = CTMDataset(test_handler.bow, testing_bert, test_handler.idx2token)
    ctm.get_thetas(testing_dataset)



Mono vs Cross-lingual
---------------------
All the examples we saw used a multilingual embedding model :code:`distiluse-base-multilingual-cased`.
However, if you are doing topic modeling in English, you can use the English sentence-bert model. In that case,
it's really easy to update the code to support mono-lingual english topic modeling.

.. code-block:: python

    training_bert = bert_embeddings_from_file("documents.txt", "bert-base-nli-mean-tokens")
    ctm = CTM(input_size=len(handler.vocab), bert_input_size=768, inference_type="combined", n_components=50)

In general, our package should be able to support all the models described in the `sentence transformer package <https://github.com/UKPLab/sentence-transformers>`_.

Development Team
----------------

* Federico Bianchi <f.bianchi@unibocconi.it> Bocconi University
* Silvia Terragni <s.terragni4@campus.unimib.it> University of Milan-Bicocca
* Dirk Hovy <dirk.hovy@unibocconi.it> Bocconi University

References
----------

Combined Topic Model

::

    @article{bianchi2020pretraining,
        title={Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence},
        author={Federico Bianchi and Silvia Terragni and Dirk Hovy},
        year={2020},
       journal={arXiv preprint arXiv:2004.03974},
    }


Fully Contextual Topic Model

::

    @article{bianchi2020crosslingual,
        title={Cross-lingual Contextualized Topic Models with Zero-shot Learning},
        author={Federico Bianchi and Silvia Terragni and Dirk Hovy and Debora Nozza and Elisabetta Fersini},
        year={2020},
       journal={arXiv preprint arXiv:2004.07737},
    }



Credits
-------


This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.
To ease the use of the library we have also incuded the `rbo`_ package, all the rights reserved to the author of that package.

Note
----

Remember that this is a research tool :)

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _`Stephen Carrow` : https://github.com/estebandito22
.. _`rbo` : https://github.com/dlukes/rbo
