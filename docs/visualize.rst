=============
Visualization
=============


PyLda Visualization
===================

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
============================

You can also create a word cloud of the topic!

.. code-block:: python

    ctm.get_wordcloud(topic_id=47, n_words=15)

.. image:: https://raw.githubusercontent.com/MilaNLProc/contextualized-topic-models/master/img/displaying_topic.png
   :align: center
   :width: 400px

