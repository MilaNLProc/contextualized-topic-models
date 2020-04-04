import numpy as np

def to_bow(data, min_length):
    """Convert index lists to bag of words representation of documents."""
    vect = [np.bincount(x[x != np.array(None)].astype('int'), minlength=min_length)
            for x in data if np.sum(x[x != np.array(None)]) != 0]
    return np.array(vect)

class VocabAndTextFromFile:

    def __init__(self, file_name):
        self.file_name = file_name
        self.vocab_dict = {}
        self.vocab = []

    def load_text_file(self):
        """
        Loads a text file
        :param text_file:
        :return:
        """
        with open(self.file_name, "r") as filino:
            data = filino.readlines()

        return data

    def create_vocab_and_index(self):
        data = self.load_text_file()

        concatenate_text = ""
        for line in data:
            line = line.strip()
            concatenate_text += line + " "
        concatenate_text = concatenate_text.strip()

        self.vocab = list(set(concatenate_text.split()))


        for index, vocab in list(zip(range(0, len(self.vocab)), self.vocab)):
            self.vocab_dict[vocab] = index

        self.index_dd = np.array(list(map(lambda y: np.array(list(map(lambda x : self.dict_to_dump[x], y.split()))), data)))


