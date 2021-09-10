=======================
Published Papers on CTM
=======================

Contextualized Topic Models (CTM) are a family of topic models that use pre-trained representations of language (e.g., BERT) to
support topic modeling. See the papers for details:

* Bianchi, F., Terragni, S., & Hovy, D. (2021). `Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence`. ACL. https://aclanthology.org/2021.acl-short.96/
* Bianchi, F., Terragni, S., Hovy, D., Nozza, D., & Fersini, E. (2021). `Cross-lingual Contextualized Topic Models with Zero-shot Learning`. EACL. https://www.aclweb.org/anthology/2021.eacl-main.143/

If you want to replicate our results, you can use our code.
You will find the W1 dataset in the colab and here: https://github.com/vinid/data, if you need the W2 dataset, send us an email (it is a bit bigger than W1 and we could not upload it on github).

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
