from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
import numpy as np

class Kitty:

    def __init__(self):
        self._assigned_classes = {}
        self.ctm = None
        self.qt = None
        self.topics_num = 0

    def train(self, documents,
              topics=10,
              embedding_model="paraphrase-distilroberta-base-v2",
              epochs=10,
              contextual_size=768,
              n_words=2000,
              language="english"):

        self.topics_num = topics
        self._assigned_classes = {k: "other" for k in range(0, self.topics_num)}

        sp = WhiteSpacePreprocessing(documents, language, n_words)
        preprocessed_documents, unpreprocessed_documents, vocab = sp.preprocess()

        self.qt = TopicModelDataPreparation(embedding_model)
        training_dataset = self.qt.fit(text_for_contextual=unpreprocessed_documents,
                                  text_for_bow=preprocessed_documents)

        self.ctm = ZeroShotTM(bow_size=len(vocab), contextual_size=contextual_size, n_components=topics, num_epochs=epochs)

        self.ctm.fit(training_dataset)  # run the model

    def get_word_classes(self) -> list:
        return self.ctm.get_topic_lists(5)

    def pretty_print_word_classes(self):
        return "\n".join(str(a) + "\t" + ", ".join(b) for a, b in enumerate(self.get_word_classes()))

    @property
    def assigned_classes(self):
        return self._assigned_classes

    @assigned_classes.setter
    def assigned_classes(self, classes):
        self._assigned_classes = {k: "other" for k in range(0, self.topics_num)}
        self._assigned_classes.update(classes)

    def predict(self, text):
        data = self.qt.transform(text)
        topic_ids = np.argmax(self.ctm.get_doc_topic_distribution(data), axis=1)

        return [self._assigned_classes[k] for k in topic_ids]








