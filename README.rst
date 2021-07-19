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
    :target: https://colab.research.google.com/drive/1fXJjr_rwqvpp1IdNQ4dxqN4Dp88cxO97?usp=sharing
    :alt: Open In Colab

.. image:: https://raw.githubusercontent.com/aleen42/badges/master/src/medium.svg
    :target: https://fbvinid.medium.com/contextualized-topic-modeling-with-python-eacl2021-eacf6dfa576
    :alt: Medium Blog Post

Contextualized Topic Models (CTM) are a family of topic models that use pre-trained representations of language (e.g., BERT) to
support topic modeling. See the papers for details:

* Bianchi, F., Terragni, S., & Hovy, D. (2021). `Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence`. ACL. https://arxiv.org/pdf/2004.03974.pdf
* Bianchi, F., Terragni, S., Hovy, D., Nozza, D., & Fersini, E. (2021). `Cross-lingual Contextualized Topic Models with Zero-shot Learning`. EACL. https://www.aclweb.org/anthology/2021.eacl-main.143/


.. image:: https://raw.githubusercontent.com/MilaNLProc/contextualized-topic-models/master/img/logo.png
   :align: center
   :width: 200px

.. contents:: Table of Contents
   :depth: 1

Topic Modeling with Contextualized Embeddings
---------------------------------------------

Our new topic modeling family supports many different languages (i.e., the one supported by HuggingFace models) and comes in two versions: **CombinedTM** combines contextual embeddings with the good old bag of words to make more coherent topics; **ZeroShotTM** is the perfect topic model for task in which you might have missing words in the test data and also, if trained with muliglingual embeddings, inherits the property of being a multilingual topic model!


Published Papers
~~~~~~~~~~~~~~~~

CombinedTM has been accepted at ACL2021 and ZeroShotTM  has been accepted at EACL2021!

If you want to replicate our results, you can use our code.
You will find the W1 dataset in the colab and here: https://github.com/vinid/data, if you need the W2 dataset, send us an email (it is a bit bigger than W1 and we could not upload it on github).

.. image:: https://raw.githubusercontent.com/MilaNLProc/contextualized-topic-models/master/img/ctm_both.jpeg
   :align: center
   :width: 600px


**Note:** Thanks to Camille DeJarnett (Stanford/CMU),  Xinyi Wang (CMU), and Graham Neubig (CMU) we found out that with the current version of the package and all the dependencies (e.g., the sentence transformers embedding model, CUDA version, PyTorch version), results with the model *distiluse-base-multilingual-cased* are lower than what appears in the paper. We suggest to use *paraphrase-multilingual-mpnet-base-v2* which is a newer multilingual model that has results that are higher than those in the paper.

See for example the results on the matches metric for Italian in the following table.

+---------------------------------------+---------------------------------------+
| Model Name                            |              Matches                  |
+=======================================+=======================================+
| paraphrase-multilingual-mpnet-base-v2 |               **0.67**                |
+---------------------------------------+---------------------------------------+
| distiluse-base-multilingual-cased     |               0.57                    |
+---------------------------------------+---------------------------------------+
| paper                                 |               0.62                    |
+---------------------------------------+---------------------------------------+

Thus, if you use ZeroShotTM for a multilingual task, we suggest the use of *paraphrase-multilingual-mpnet-base-v2*.


Tutorials
---------

You can look at our `medium`_ blog post or start from one of our Colab Tutorials:


.. |colab1_2| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1fXJjr_rwqvpp1IdNQ4dxqN4Dp88cxO97?usp=sharing
    :alt: Open In Colab

.. |colab2_2| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1bfWUYEypULFk_4Tfff-Pb_n7-tSjEe9v?usp=sharing
    :alt: Open In Colab

.. |colab3_3| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1upTRu4zSm1VMbl633n9qkIDA526l22E_?usp=sharing
    :alt: Open In Colab

+--------------------------------------------------------------------------------+------------------+
| Name                                                                           | Link             |
+================================================================================+==================+
| Combined TM on Wikipedia Data (Preproc+Saving+Viz) (stable **v2.0.0**)         | |colab1_2|       |
+--------------------------------------------------------------------------------+------------------+
| Zero-Shot Cross-lingual Topic Modeling (Preproc+Viz) (stable **v2.0.0**)       | |colab2_2|       |
+--------------------------------------------------------------------------------+------------------+
| SuperCTM and  β-CTM (High-level usage) (stable **v2.1.0**)                     | |colab3_3|       |
+--------------------------------------------------------------------------------+------------------+

Overview
--------

TL;DR
~~~~~

+ In CTMs we have two models. CombinedTM and ZeroShotTM, which have different use cases.
+ CTMs work better when the size of the bag of words **has been restricted to a number of terms** that does not go over **2000 elements**. This is because we have a neural model that reconstructs the input bag of word, Moreover, in CombinedTM we project the contextualized embedding to the vocab space, the bigger the vocab the more parameters you get, with the training being more difficult and prone to bad fitting. This is **NOT** a strict limit, however, consider preprocessing your dataset. We have a preprocessing_ pipeline that can help you in dealing with this.
+ Check the contextual model you are using, the **multilingual model one used on English data might not give results that are as good** as the pure English trained one.
+ **Preprocessing is key**. If you give a contextual model like BERT preprocessed text, it might be difficult to get out a good representation. What we usually do is use the preprocessed text for the bag of word creating and use the NOT preprocessed text for BERT embeddings. Our preprocessing_ class can take care of this for you.

Installing
~~~~~~~~~~

**Important**: If you want to use CUDA you need to install the correct version of
the CUDA systems that matches your distribution, see pytorch_.

Install the package using pip

.. code-block:: bash

    pip install -U contextualized_topic_models

An important aspect to take into account is which network you want to use: the one that combines BERT and the BoW or the one that just uses BERT.
It's easy to swap from one to the other:

ZeroShotTM:

.. code-block:: python

    ZeroShotTM(bow_size=len(qt.vocab), contextual_size=embedding_dimension, n_components=number_of_topics)

CombinedTM:

.. code-block:: python

    CombinedTM(bow_size=len(qt.vocab), contextual_size=embedding_dimension, n_components=number_of_topics)


But remember that you can do zero-shot cross-lingual topic modeling only with the :code:`ZeroShotTM` model. See cross-lingual-topic-modeling_

References
----------

If you find this useful you can cite the following papers :)

**ZeroShotTM**

::

    @inproceedings{bianchi-etal-2021-cross,
        title = "Cross-lingual Contextualized Topic Models with Zero-shot Learning",
        author = "Bianchi, Federico and Terragni, Silvia and Hovy, Dirk  and
          Nozza, Debora and Fersini, Elisabetta",
        booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
        month = apr,
        year = "2021",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/2021.eacl-main.143",
        pages = "1676--1683",
    }

**CombinedTM**

::

    @inproceedings{bianchi2021pretraining,
        title={Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence},
        author={Federico Bianchi and Silvia Terragni and Dirk Hovy},
        year={2021},
       booktitle={ACL},
    }


Does it work for different languages? Of Course!
------------------------------------------------

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

Topic Models
------------

Combined Topic Model
~~~~~~~~~~~~~~~~~~~~

Here is how you can use the CombinedTM. This is a standard topic model that also uses contextualized embeddings. The good thing about CombinedTM is that it makes your topic much more coherent (see the paper https://arxiv.org/abs/2004.03974).

.. code-block:: python

    from contextualized_topic_models.models.ctm import CombinedTM
    from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
    from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file

    qt = TopicModelDataPreparation("paraphrase-distilroberta-base-v2")

    training_dataset = qt.fit(text_for_contextual=list_of_unpreprocessed_documents, text_for_bow=list_of_preprocessed_documents)

    ctm = CombinedTM(bow_size=len(qt.vocab), contextual_size=768, n_components=50)

    ctm.fit(training_dataset) # run the model

    ctm.get_topics()


**Advanced Notes:** Combined TM combines the BoW with SBERT, a process that seems to increase
the coherence of the predicted topics (https://arxiv.org/pdf/2004.03974.pdf).

Zero-Shot Topic Model
~~~~~~~~~~~~~~~~~~~~~

Our ZeroShotTM can be used for zero-shot topic modeling. It can handle words that are not used during the training phase.
More interestingly, this model can be used for cross-lingual topic modeling (See next sections)! See the paper (https://www.aclweb.org/anthology/2021.eacl-main.143)

.. code-block:: python

    from contextualized_topic_models.models.ctm import ZeroShotTM
    from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
    from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file

    text_for_contextual = [
        "hello, this is unpreprocessed text you can give to the model",
        "have fun with our topic model",
    ]

    text_for_bow = [
        "hello unpreprocessed give model",
        "fun topic model",
    ]

    qt = TopicModelDataPreparation("paraphrase-multilingual-mpnet-base-v2")

    training_dataset = qt.fit(text_for_contextual=text_for_contextual, text_for_bow=text_for_bow)

    ctm = ZeroShotTM(bow_size=len(qt.vocab), contextual_size=768, n_components=50)

    ctm.fit(training_dataset) # run the model

    ctm.get_topics(2)


As you can see, the high-level API to handle the text is pretty easy to use;
**text_for_bert** should be used to pass to the model a list of documents that are not preprocessed.
Instead, to **text_for_bow** you should pass the preprocessed text used to build the BoW.

**Advanced Notes:** in this way, SBERT can use all the information in the text to generate the representations.

Using The Topic Models
----------------------

Getting The Topics
~~~~~~~~~~~~~~~~~~

Once the model is trained, it is very easy to get the topics!

.. code-block:: python

    ctm.get_topics()

Predicting Topics For Unseen Documents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **transform** method will take care of most things for you, for example the generation
of a corresponding BoW by considering only the words that the model has seen in training.
However, this comes with some bumps when dealing with the ZeroShotTM, as we will se in the next section.

You can, however, manually load the embeddings if you like (see the Advanced part of this documentation).

Mono-Lingual Topic Modeling
===========================

If you use **CombinedTM** you need to include the test text for the BOW:

.. code-block:: python

    testing_dataset = qt.transform(text_for_contextual=testing_text_for_contextual, text_for_bow=testing_text_for_bow)

    # n_sample how many times to sample the distribution (see the doc)
    ctm.get_doc_topic_distribution(testing_dataset, n_samples=20) # returns a (n_documents, n_topics) matrix with the topic distribution of each document

If you use **ZeroShotTM** you do not need to use the `testing_text_for_bow` because if you are using
a different set of test documents, this will create a BoW of a different size. Thus, the best
way to do this is to pass just the text that is going to be given in input to the contexual model:

.. code-block:: python

    testing_dataset = qt.transform(text_for_contextual=testing_text_for_contextual)

    # n_sample how many times to sample the distribution (see the doc)
    ctm.get_doc_topic_distribution(testing_dataset, n_samples=20)


Cross-Lingual Topic Modeling
============================

Once you have trained the ZeroShotTM model with multilingual embeddings,
you can use this simple pipeline to predict the topics for documents in a different language (as long as this language
is covered by **paraphrase-multilingual-mpnet-base-v2**).

.. code-block:: python

    # here we have a Spanish document
    testing_text_for_contextual = [
        "hola, bienvenido",
    ]

    # since we are doing multilingual topic modeling, we do not need the BoW in
    # ZeroShotTM when doing cross-lingual experiments (it does not make sense, since we trained with an english Bow
    # to use the spanish BoW)
    testing_dataset = qt.transform(testing_text_for_contextual)

    # n_sample how many times to sample the distribution (see the doc)
    ctm.get_doc_topic_distribution(testing_dataset, n_samples=20) # returns a (n_documents, n_topics) matrix with the topic distribution of each document

**Advanced Notes:** We do not need to pass the Spanish bag of word: the bag of words of the two languages will not be comparable! We are passing it to the model for compatibility reasons, but you cannot get
the output of the model (i.e., the predicted BoW of the trained language) and compare it with the testing language one.

Extensions
----------

We have developed two extensions to CTM, one that supports supervision and another one that
uses a weight on the KL loss to generate disentangled representations.

**NOTE**: both model haven't been thoroughly validated. Use them with care and let us know if you find something cool!

SuperCTM
~~~~~~~~

Inspiration for SuperCTM has been taken directly from the work by `Card et al., 2018 <https://aclanthology.org/P18-1189/>`_ (you can read this as
"we essentially implemented their approach in our architecture"). SuperCTM should give better representations of the documents - this is somewhat expected, since we are using the labels to give more information to the model - and in theory should also make the model able to find topics more coherent with respect to the labels.
The model is super easy to use and requires minor modifications to the already implemented pipeline:


.. code-block:: python

    from contextualized_topic_models.models.ctm import ZeroShotTM
    from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation

    text_for_contextual = [
        "hello, this is unpreprocessed text you can give to the model",
        "have fun with our topic model",
    ]

    text_for_bow = [
        "hello unpreprocessed give model",
        "fun topic model",
    ]

    labels = [0, 1] # we need to have  a label for each document

    qt = TopicModelDataPreparation("paraphrase-multilingual-mpnet-base-v2")

    # training dataset should contain the labels
    training_dataset = qt.fit(text_for_contextual=text_for_contextual, text_for_bow=text_for_bow, labels=labels)

    # model should know the label size in advance
    ctm = CombinedTM(bow_size=len(qt.vocab), contextual_size=768, n_components=50, label_size=len(set(labels)))

    ctm.fit(training_dataset) # run the model

    ctm.get_topics(2)


β-CTM
~~~~~~

We also implemented the intuition found in the work by `Higgins et al., 2018 <https://openreview.net/forum?id=Sy2fzU9gl>`_, where a weight is applied
to the KL loss function. The idea is that giving more weight to the KL part of the loss function helps in creating disentangled representations
by forcing independence in the components. Again, the model should be straightforward to use:

.. code-block:: python

     ctm = CombinedTM(bow_size=len(qt.vocab), contextual_size=768, n_components=50, loss_weights={"beta" : 3})




Visualization
-------------

PyLda Visualization
~~~~~~~~~~~~~~~~~~~

We support pyLDA visualizations with few lines of code!

.. code-block:: python

    import pyLDAvis as vis

    lda_vis_data = ctm.get_ldavis_data_format(tp.vocab, training_dataset, n_samples=10)

    ctm_pd = vis.prepare(**lda_vis_data)
    vis.display(ctm_pd)

.. image:: https://raw.githubusercontent.com/MilaNLProc/contextualized-topic-models/master/img/pyldavis.png
   :align: center
   :width: 400px


Showing The Topic Word Cloud
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also create a word cloud of the topic!

.. code-block:: python

    ctm.get_wordcloud(topic_id=47, n_words=15)

.. image:: https://raw.githubusercontent.com/MilaNLProc/contextualized-topic-models/master/img/displaying_topic.png
   :align: center
   :width: 400px


More Advanced Stuff
-------------------

Can I load my own embeddings?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sure, here is a snippet that can help you. You need to create the embeddings (for bow and contextualized) and you also need
to have the vocab and an id2token dictionary (maps integers ids to words).

.. code-block:: python

    qt = TopicModelDataPreparation()

    training_dataset = qt.load(contextualized_embeddings, bow_embeddings, id2token)
    ctm = CombinedTM(bow_size=len(vocab), contextual_size=768, n_components=50)
    ctm.fit(training_dataset) # run the model
    ctm.get_topics()

You can give a look at the code we use in the TopicModelDataPreparation object to get an idea on how to create everything from scratch.
For example:

.. code-block:: python

        vectorizer = CountVectorizer() #from sklearn

        train_bow_embeddings = vectorizer.fit_transform(text_for_bow)
        train_contextualized_embeddings = bert_embeddings_from_list(text_for_contextual, "chosen_contextualized_model")
        vocab = vectorizer.get_feature_names()
        id2token = {k: v for k, v in zip(range(0, len(vocab)), vocab)}

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
    sp = WhiteSpacePreprocessing(documents, "english")
    preprocessed_documents, unpreprocessed_documents, vocab = sp.preprocess()


Development Team
----------------

* `Federico Bianchi`_ <f.bianchi@unibocconi.it> Bocconi University
* `Silvia Terragni`_ <s.terragni4@campus.unimib.it> University of Milan-Bicocca
* `Dirk Hovy`_ <dirk.hovy@unibocconi.it> Bocconi University


Software Details
----------------

* Free software: MIT license
* Documentation: https://contextualized-topic-models.readthedocs.io.
* Super big shout-out to `Stephen Carrow`_ for creating the awesome https://github.com/estebandito22/PyTorchAVITM package from which we constructed the foundations of this package. We are happy to redistribute this software again under the MIT License.



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
.. _SBERT: https://www.sbert.net/docs/pretrained_models.html
.. _HuggingFace: https://huggingface.co/models
.. _UmBERTo: https://huggingface.co/Musixmatch/umberto-commoncrawl-cased-v1
.. _medium: https://fbvinid.medium.com/contextualized-topic-modeling-with-python-eacl2021-eacf6dfa576

