from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation


class Kitty:

    def __init__(self):
        self._assigned_classes = {}
        self.ctm = None

    def train(self, documents, topics=10,
              embedding_model="paraphrase-distilroberta-base-v2",
              contextual_size=768,
              n_words=2000,
              language="english"):

        sp = WhiteSpacePreprocessing(documents, language, n_words)
        preprocessed_documents, unpreprocessed_documents, vocab = sp.preprocess()

        qt = TopicModelDataPreparation(embedding_model)
        training_dataset = qt.fit(text_for_contextual=unpreprocessed_documents,
                                  text_for_bow=preprocessed_documents)

        self.ctm = ZeroShotTM(bow_size=len(vocab), contextual_size=contextual_size, n_components=topics)

        self.ctm.fit(training_dataset)  # run the model

    def get_classes(self) -> list:
        return self.ctm.get_topic_lists()

    @property
    def assigned_classes(self):
        return self._assigned_classes

    @assigned_classes.setter
    def assigned_classes(self, classes):
        self._assigned_classes = classes




