=======
History
=======


2.2.2 (2021-11-09)
------------------

* kitty now takes in input a stopword list instead of a language (from which it gathered the stopwords)
* solving a bug in the whitespace preprocessing function
* adding a new preprocessing function that supports passing the stopwords as a list
* deprecating whitespace preprocessing
* minor fixes to kitty API
* breaking change to kitty API, now uses WhiteSpacePreprocessingStopwords.

2.2.0 (2021-09-20)
------------------

* introducing kitty
* improving the documentation a lot

2.1.2 (2021-09-03)
------------------

* patching `Issue 38 <https://github.com/MilaNLProc/contextualized-topic-models/issues/38>`_
* improvements `PR 80 <https://github.com/MilaNLProc/contextualized-topic-models/pull/80>`_


2.1.0 (2021-07-16)
------------------

* new model introduced SuperCTM
* new model introduced β-CTM

2.0.0 (2021-xx-xx)
------------------

* warning, breaking changes were introduced:
    * the order of the parameters in CTMDataset was changed (now first is contextual embeddings)
    * CTM takes in input bow_size, contextual_size instead of input_size and bert_size
    * changed the name of the parameters in the dataset
* introduced early stopping
* introduced visualization with pyldavis

1.8.2 (2021-02-08)
------------------

* removed constraint over pytorch version. This should solve problems for Windows users

1.8.0 (2021-01-11)
------------------

* novel way to handle text, we now allow for an easy usage of training and testing data
* better visualization of the training progress and of the sampling process
* removed old stuff from the documentation

1.7.1 (2020-12-17)
------------------

* some minor updates to the documentation
* adding a new method to visualize the topic using a wordcloud
* save and load will now generate a warning since the feature has not been tested


1.7.0 (2020-12-10)
------------------

* adding a new and much simpler way to handle text for topic modeling

1.6.0 (2020-11-03)
------------------

* introducing the two different classes for ZeroShotTM and CombinedTM
* depracating CTM class in favor of ZeroShotTM and CombinedTM


1.5.3 (2020-11-03)
------------------

* adding support for Windows encoding by defaulting file load to UTF-8

1.5.2 (2020-11-03)
------------------

* updated sentence-transformers version to 0.3.6
* beta support for model saving and loading
* new evaluation metrics based on coherence

1.5.0 (2020-09-14)
------------------

* Introduced a method to predict the topics for a set of documents (supports multiple sampling to reduce variation)
* Adding some features to bert embeddings creation like increased batch size and progress bar
* Supporting training directly from lists without the need to deal with files
* Adding a simple quick preprocessing pipeline

1.4.3 (2020-09-03)
------------------

* Updating sentence-transformers package to avoid errors

1.4.2 (2020-08-04)
------------------

* Changed the encoding on file load for the SBERT embedding function

1.4.1 (2020-08-04)
------------------

* Fixed bug over sparse matrices

1.4.0 (2020-08-01)
------------------

* New feature handling sparse bow for optimized processing
* New method to return topic distributions for words

1.0.0 (2020-04-05)
------------------

* Released models with the main features implemented

0.1.0 (2020-04-04)
------------------

* First release on PyPI.
