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


The contextual neural topic model can be easily instantiated using few parameters (although there is a wide range of parameters you can use to change the behaviour of the neural topic model.

.. code-block:: python

    cotm = COTM(input_size=1000, bert_input_size=512, inferencetype="contextual")
    cotm.fit()


See the example notebook in the `contextualized_topic_models/examples` folder

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
