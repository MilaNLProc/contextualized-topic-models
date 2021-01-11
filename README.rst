===========================
Contextualized Topic Models
===========================

.. image:: https://img.shields.io/pypi/v/contextualized_topic_models.svg
        :target: https://pypi.python.org/pypi/contextualized_topic_models

.. image:: https://github.com/MilaNLProc/contextualized-topic-models/workflows/Python%20package/badge.svg
        :target: https://github.com/MilaNLProc/contextualized-topic-models/actions

.. image:: https://readthedocs.org/projects/contextualized-topic-models/badge/?version=latest
        :target: https://contextualized-topic-models.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/github/contributors/MilaNLProc/contextualized-topic-models
        :target: https://github.com/MilaNLProc/contextualized-topic-models/graphs/contributors/
        :alt: Contributors

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
        :target: https://lbesson.mit-license.org/
        :alt: License

.. image:: https://pepy.tech/badge/contextualized-topic-models
        :target: https://pepy.tech/project/contextualized-topic-models
        :alt: Downloads

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1GCKpfu6ZfyVTk9_FovxnyH48OkNIYOIb?usp=sharing
    :alt: Open In Colab

Contextualized Topic Models (CTM) are a family of topic models that use pre-trained representations of language (e.g., BERT) to
support topic modeling. See the papers for details:

* Bianchi, F., Terragni, S., Hovy, D., Nozza, D., & Fersini, E. (2021). `Cross-lingual Contextualized Topic Models with Zero-shot Learning`. EACL. https://arxiv.org/pdf/2004.07737v1.pdf
* Bianchi, F., Terragni, S., & Hovy, D. (2020). `Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence` https://arxiv.org/pdf/2004.03974.pdf


.. image:: https://raw.githubusercontent.com/MilaNLProc/contextualized-topic-models/master/img/logo.png
   :align: center
   :width: 200px

README
------

Make **sure** you read the doc a bit.
The cross-lingual topic modeling requires to use a ZeroShot model and it is trained only on **ONE** language;
with the power of multilingual BERT it can then be used to predict the topics of documents in unseen languages.
For more details, you can read the two papers mentioned above.


Jump start Tutorial
-------------------

.. |colab118| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1wVWHe8xHasnUbzNs40MwlkJsUhvN98se?usp=sharing
    :alt: Open In Colab

.. |colab218| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1-KZ7bwS7eM24Q4dbIBEv2C4gC-6xWOmB?usp=sharing
    :alt: Open In Colab

.. |colab1new| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1GCKpfu6ZfyVTk9_FovxnyH48OkNIYOIb?usp=sharing
    :alt: Open In Colab

.. |colabzt18| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1-KZ7bwS7eM24Q4dbIBEv2C4gC-6xWOmB?usp=sharing
    :alt: Open In Colab

.. |colabzt| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/13YhYgJN9EjSQw5bsZYzMaaiNKQpt_SQn?usp=sharing
    :alt: Open In Colab


+----------------------------------------------------------------+--------------------+
| Name                                                           | Link               |
+================================================================+====================+
| Zero-Shot Cross-lingual Topic Modeling (stable **v1.8.0**)     | |colabzt18|        |
+----------------------------------------------------------------+--------------------+
| CombinedTM for Wikipedia Documents (stable **v1.8.0**)         | |colab118|         |
+----------------------------------------------------------------+--------------------+
| CombinedTM with Preprocessing (stable **v1.8.0**)              | |colab218|         |
+----------------------------------------------------------------+--------------------+
| Zero-Shot Cross-lingual Topic Modeling (**v1.7.0**)            | |colabzt|          |
+----------------------------------------------------------------+--------------------+
| CombinedTM for Wikipedia Documents (**v1.7.0**)                | |colab1new|        |
+----------------------------------------------------------------+--------------------+

TL;DR
-----

+ In CTMs we have two models. CombinedTM and ZeroShotTM, which have different use cases.
+ CTMs work better when the size of the bag of words **has been restricted to a number of terms** that does not go over **2000 elements** (this is because we have a neural model that reconstructs the input bag of word). This is **NOT** a strict limit, however, consider preprocessing your dataset. We have a preprocessing_ pipeline that can help you in dealing with this.
+ Check the contextual model you are using, the **multilingual model one used on English data might not give results that are as good** as the pure English trained one.
+ **Preprocessing is key**. If you give a contextual model like BERT preprocessed text, it might be difficult to get out a good representation. What we usually do is use the preprocessed text for the bag of word creating and use the NOT preprocessed text for BERT embeddings. Our preprocessing_ class can take care of this for you.


Software Details
~~~~~~~~~~~~~~~~

* Free software: MIT license
* Documentation: https://contextualized-topic-models.readthedocs.io.
* Super big shout-out to `Stephen Carrow`_ for creating the awesome https://github.com/estebandito22/PyTorchAVITM package from which we constructed the foundations of this package. We are happy to redistribute this software again under the MIT License.


Features
~~~~~~~~

* Combines Contextual Language Models (e.g., BERT) and Neural Variational Topic Models
* Two different methodologies: Combined, where we combine BoW and contextual embeddings and ZeroShot, that uses only contextual embeddings
* Includes methods to create embedded representations and BoW
* Includes evaluation metrics
* Includes wordclouds


Overview
--------

**Important**: If you want to use CUDA you need to install the correct version of
the CUDA systems that matches your distribution, see pytorch_.

Install the package using pip

.. code-block:: bash

    pip install -U contextualized_topic_models

Contextual neural topic models can be easily instantiated using few parameters (although there is a wide range of
parameters you can use to change the behaviour of the neural topic model). When you generate
embeddings with BERT remember that there is a maximum length and for documents that are too long some words will be ignored.

An important aspect to take into account is which network you want to use: the one that combines BERT and the BoW or the one that just uses BERT.
It's easy to swap from one to the other:

ZeroShotTM:

.. code-block:: python

    ZeroShotTM(input_size=len(qt.vocab), bert_input_size=embedding_dimension, n_components=number_of_topics)

CombinedTM:

.. code-block:: python

    CombinedTM(input_size=len(qt.vocab), bert_input_size=embedding_dimension,  n_components=number_of_topics)


But remember that you can do zero-shot cross-lingual topic modeling only with the :code:`ZeroShotTM` model. See cross-lingual-topic-modeling_

Mono vs Multilingual Embeddings: Which Embeddings Should I Use?
----------------------------------------------------------------

All the examples below use a multilingual embedding model :code:`distiluse-base-multilingual-cased`.
If you are doing topic modeling in English, **you SHOULD use the English sentence-bert model**, `bert-base-nli-mean-tokens`. In that case,
it's really easy to update the code to support monolingual English topic modeling.

.. code-block:: python

    qt = TopicModelDataPreparation("bert-base-nli-mean-tokens")

In general, our package should be able to support all the models described in the `sentence transformer package <https://github.com/UKPLab/sentence-transformers>`_ and in HuggingFace.

Zero-Shot Cross-Lingual Topic Modeling
--------------------------------------

Our ZeroShotTM can be used for zero-shot topic modeling. It can handle words that are not used during the training phase.
More interestingly, this model can be used for cross-lingual topic modeling! See the paper (https://arxiv.org/pdf/2004.07737v1.pdf)

.. code-block:: python

    from contextualized_topic_models.models.ctm import ZeroShotTM
    from contextualized_topic_models.utils.data_preparation import QuickText
    from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file
    from contextualized_topic_models.datasets.dataset import CTMDataset

    text_for_contextual = [
        "hello, this is unpreprocessed text you can give to the model",
        "have fun with our topic model",
    ]

    text_for_bow = [
        "hello unpreprocessed give model",
        "fun topic model",
    ]

    qt = TopicModelDataPreparation("distiluse-base-multilingual-cased")

    training_dataset = qt.create_training_set(text_for_contextual, text_for_bow)

    ctm = ZeroShotTM(input_size=len(qt.vocab), bert_input_size=512, n_components=50)

    ctm.fit(training_dataset) # run the model

    ctm.get_topics()


As you can see, the high-level API to handle the text is pretty easy to use;
**text_for_bert** should be used to pass to the model a list of documents that are not preprocessed.
Instead, to **text_for_bow** you should pass the preprocessed text used to build the BoW.

**Advanced Notes:** in this way, SBERT can use all the information in the text to generate the representations.

Predict Topics for Unseen Documents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have trained the cross-lingual topic model,
you can use this simple pipeline to predict the topics for documents in a different language (as long as this language
is covered by **distiluse-base-multilingual-cased**).

.. code-block:: python

    # here we have a Spanish document
    testing_text_for_contextual = [
        "hola, bienvenido",
    ]

    testing_dataset = qt.create_test_set(testing_text_for_contextual)

    # n_sample how many times to sample the distribution (see the doc)
    ctm.get_doc_topic_distribution(testing_dataset, n_samples=20) # returns a (n_documents, n_topics) matrix with the topic distribution of each document

**Advanced Notes:** We do not need to pass the Spanish bag of word: the bag of words of the two languages will not be comparable! We are passing it to the model for compatibility reasons, but you cannot get
the output of the model (i.e., the predicted BoW of the trained language) and compare it with the testing language one.

Showing The Topic Word Cloud
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also create a word cloud of the topic!

.. code-block:: python

    ctm.get_wordcloud(topic_id=47, n_words=15)

.. image:: https://raw.githubusercontent.com/MilaNLProc/contextualized-topic-models/master/img/displaying_topic.png
   :align: center
   :width: 400px


Combined Topic Modeling
-----------------------

Here is how you can use the CombinedTM. This is a standard topic model that also uses BERT.

.. code-block:: python

    from contextualized_topic_models.models.ctm import CombinedTM
    from contextualized_topic_models.utils.data_preparation import QuickText
    from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file
    from contextualized_topic_models.datasets.dataset import CTMDataset

    qt = TopicModelDataPreparation("bert-base-nli-mean-tokens")

    training_dataset = qt.create_training_set(list_of_unpreprocessed_documents, list_of_preprocessed_documents)

    ctm = CombinedTM(input_size=len(qt.vocab), bert_input_size=768, n_components=50)

    ctm.fit(training_dataset) # run the model

    ctm.get_topics()


**Advanced Notes:** Combined TM combines the BoW with SBERT, a process that seems to increase
the coherence of the predicted topics (https://arxiv.org/pdf/2004.03974.pdf).

More Advanced Stuff
-------------------

Training and Testing with CombinedTM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    training_dataset = qt.create_test_set(testing_text_for_contextual, testing_text_for_bow)

    # n_sample how many times to sample the distribution (see the doc)
    ctm.get_doc_topic_distribution(testing_dataset, n_samples=20)


Can I load my own embeddings?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sure, here is a snippet that can help you. You need to create the embeddings (for bow and contextualized) and you also need
to have the vocab and an id2token dictionary (maps integers ids to words).

.. code-block:: python

    qt = TopicModelDataPreparation()

    training_dataset = qt.load(contextualized_embeddings, bow_embeddings, id2token)
    ctm = CombinedTM(input_size=len(vocab), bert_input_size=768, n_components=50)
    ctm.fit(training_dataset) # run the model
    ctm.get_topics()

You can give a look at the code we use in the TopicModelDataPreparation object to get an idea on how to create everything from scratch.
For example:

.. code-block:: python

        self.vectorizer = CountVectorizer() #from sklearn

        train_bow_embeddings = self.vectorizer.fit_transform(text_for_bow)
        train_contextualized_embeddings = bert_embeddings_from_list(text_for_contextual, self.contextualized_model)
        self.vocab = self.vectorizer.get_feature_names()
        self.id2token = {k: v for k, v in zip(range(0, len(self.vocab)), self.vocab)}

Evaluation
~~~~~~~~~~

We have also included some of the metrics normally used in the evaluation of topic models, for example you can compute the coherence of your
topics using NPMI using our simple and high-level API.

.. code-block:: python

    from contextualized_topic_models.evaluation.measures import CoherenceNPMI

    with open('preprocessed_documents.txt', "r") as fr:
        texts = [doc.split() for doc in fr.read().splitlines()] # load text for NPMI

    npmi = CoherenceNPMI(texts=texts, topics=ctm.get_topic_lists(10))
    npmi.score()


Preprocessing
~~~~~~~~~~~~~

Do you need a quick script to run the preprocessing pipeline? We got you covered! Load your documents
and then use our SimplePreprocessing class. It will automatically filter infrequent words and remove documents
that are empty after training. The preprocess method will return the preprocessed and the unpreprocessed documents.
We generally use the unpreprocessed for BERT and the preprocessed for the Bag Of Word.

.. code-block:: python

    from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing

    documents = [line.strip() for line in open("unpreprocessed_documents.txt").readlines()]
    sp = WhiteSpacePreprocessing(documents)
    preprocessed_documents, unpreprocessed_documents, vocab = sp.preprocess()


Development Team
----------------

* `Federico Bianchi`_ <f.bianchi@unibocconi.it> Bocconi University
* `Silvia Terragni`_ <s.terragni4@campus.unimib.it> University of Milan-Bicocca
* `Dirk Hovy`_ <dirk.hovy@unibocconi.it> Bocconi University

References
----------

If you use this in a research work please cite these papers:

ZeroShotTM

::

    @inproceedings{bianchi2020crosslingual,
        title={Cross-lingual Contextualized Topic Models with Zero-shot Learning},
        author={Federico Bianchi and Silvia Terragni and Dirk Hovy and Debora Nozza and Elisabetta Fersini},
        booktitle={EACL},
        year={2021}
    }

CombinedTM

::

    @article{bianchi2020pretraining,
        title={Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence},
        author={Federico Bianchi and Silvia Terragni and Dirk Hovy},
        year={2020},
       journal={arXiv preprint arXiv:2004.03974},
    }

ZeroShot Topic Model
--------------------

.. image:: https://raw.githubusercontent.com/MilaNLProc/contextualized-topic-models/master/img/lm_topic_model_multilingual.png
   :target: https://raw.githubusercontent.com/MilaNLProc/contextualized-topic-models/master/img/lm_topic_model_multilingual.png
   :align: center
   :width: 400px

Combined Topic Model
--------------------

.. image:: https://raw.githubusercontent.com/MilaNLProc/contextualized-topic-models/master/img/lm_topic_model.png
   :target: https://raw.githubusercontent.com/MilaNLProc/contextualized-topic-models/master/img/lm_topic_model.png
   :align: center
   :width: 400px


Credits
-------


This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.
To ease the use of the library we have also included the `rbo`_ package, all the rights reserved to the author of that package.

Note
----

Remember that this is a research tool :)

.. _pytorch: https://pytorch.org/get-started/locally/
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _preprocessing: https://github.com/MilaNLProc/contextualized-topic-models#preprocessing
.. _cross-lingual-topic-modeling: https://github.com/MilaNLProc/contextualized-topic-models#cross-lingual-topic-modeling
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _`Stephen Carrow` : https://github.com/estebandito22
.. _`rbo` : https://github.com/dlukes/rbo
.. _Federico Bianchi: https://federicobianchi.io
.. _Silvia Terragni: https://silviatti.github.io/
.. _Dirk Hovy: https://dirkhovy.com/
