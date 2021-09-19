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

* Bianchi, F., Terragni, S., & Hovy, D. (2021). `Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence`. ACL. https://aclanthology.org/2021.acl-short.96/
* Bianchi, F., Terragni, S., Hovy, D., Nozza, D., & Fersini, E. (2021). `Cross-lingual Contextualized Topic Models with Zero-shot Learning`. EACL. https://www.aclweb.org/anthology/2021.eacl-main.143/


.. image:: https://raw.githubusercontent.com/MilaNLProc/contextualized-topic-models/master/img/logo.png
   :align: center
   :width: 200px

Topic Modeling with Contextualized Embeddings
---------------------------------------------

Our new topic modeling family supports many different languages (i.e., the one supported by HuggingFace models) and comes in two versions: **CombinedTM** combines contextual embeddings with the good old bag of words to make more coherent topics; **ZeroShotTM** is the perfect topic model for task in which you might have missing words in the test data and also, if trained with muliglingual embeddings, inherits the property of being a multilingual topic model!

The big advantage is that you can use different embeddings for CTMs. Thus, when a new
embedding method comes out you can use it in the code and improve your results. We are not limited
by the BoW anymore.

We also have kitty! a new submodule that can be used to quickly create an human in the loop
classifier to quickly classify your documents and create named clusters.

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

.. |kitty_colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1ZO6y-laPMnIT6boMwNXK4WNiyAUWUK4L?usp=sharing
    :alt: Open In Colab

+--------------------------------------------------------------------------------+------------------+
| Name                                                                           | Link             |
+================================================================================+==================+
| Combined TM on Wikipedia Data (Preproc+Saving+Viz) (stable **v2.2.0**)         | |colab1_2|       |
+--------------------------------------------------------------------------------+------------------+
| Zero-Shot Cross-lingual Topic Modeling (Preproc+Viz) (stable **v2.2.0**)       | |colab2_2|       |
+--------------------------------------------------------------------------------+------------------+
| Kitty: Human in the loop Classifier (High-level usage) (stable **v2.2.0**)     | |kitty_colab|    |
+--------------------------------------------------------------------------------+------------------+
| SuperCTM and  β-CTM (High-level usage) (stable **v2.2.0**)                     | |colab3_3|       |
+--------------------------------------------------------------------------------+------------------+

Overview
--------

TL;DR
~~~~~

+ In CTMs we have two models. CombinedTM and ZeroShotTM, which have different use cases.
+ CTMs work better when the size of the bag of words **has been restricted to a number of terms** that does not go over **2000 elements**. This is because we have a neural model that reconstructs the input bag of word, Moreover, in CombinedTM we project the contextualized embedding to the vocab space, the bigger the vocab the more parameters you get, with the training being more difficult and prone to bad fitting. This is **NOT** a strict limit, however, consider preprocessing your dataset. We have a preprocessing_ pipeline that can help you in dealing with this.
+ Check the contextual model you are using, the **multilingual model one used on English data might not give results that are as good** as the pure English trained one.
+ **Preprocessing is key**. If you give a contextual model like BERT preprocessed text, it might be difficult to get out a good representation. What we usually do is use the preprocessed text for the bag of word creating and use the NOT preprocessed text for BERT embeddings. Our preprocessing_ class can take care of this for you.

Features
~~~~~~~~

An important aspect to take into account is which network you want to use:
the one that combines contextualized embeddings
and the BoW (CombinedTM) or the one that just uses contextualized embeddings (ZeroShotTM)

But remember that you can do zero-shot cross-lingual topic modeling only with the :code:`ZeroShotTM` model. See cross-lingual-topic-modeling_

We also have :code:`Kitty`: a utility you can use to do a simpler human in the loop classification of your
documents. This can be very useful to do document filtering. It also works in cross-lingual setting and
thus you might be able to filter documents in a language you don't know!

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

    @inproceedings{bianchi-etal-2021-pre,
        title = "Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence",
        author = "Bianchi, Federico  and
          Terragni, Silvia  and
          Hovy, Dirk",
        booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
        month = aug,
        year = "2021",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2021.acl-short.96",
        doi = "10.18653/v1/2021.acl-short.96",
        pages = "759--766",
    }


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

