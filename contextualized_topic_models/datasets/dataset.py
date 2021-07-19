import torch
from torch.utils.data import Dataset
import scipy.sparse


class CTMDataset(Dataset):

    """Class to load BoW and the contextualized embeddings."""

    def __init__(self, X_contextual, X_bow, idx2token, labels=None):

        if X_bow.shape[0] != len(X_contextual):
            raise Exception("Wait! BoW and Contextual Embeddings have different sizes! "
                            "You might want to check if the BoW preparation method has removed some documents. ")

        if labels is not None:
            if labels.shape[0] != X_bow.shape[0]:
                raise Exception(f"There is something wrong in the length of the labels (size: {labels.shape[0]}) "
                                f"and the bow (len: {X_bow.shape[0]}). These two numbers should match.")

        self.X_bow = X_bow
        self.X_contextual = X_contextual
        self.idx2token = idx2token
        self.labels = labels

    def __len__(self):
        """Return length of dataset."""
        return self.X_bow.shape[0]

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        if type(self.X_bow[i]) == scipy.sparse.csr.csr_matrix:
            X_bow = torch.FloatTensor(self.X_bow[i].todense())
            X_contextual = torch.FloatTensor(self.X_contextual[i])
        else:
            X_bow = torch.FloatTensor(self.X_bow[i])
            X_contextual = torch.FloatTensor(self.X_contextual[i])

        return_dict = {'X_bow': X_bow, 'X_contextual': X_contextual}

        if self.labels is not None:
            labels = self.labels[i]
            if type(labels) == scipy.sparse.csr.csr_matrix:
                return_dict["labels"] = torch.FloatTensor(labels.todense())
            else:
                return_dict["labels"] = torch.FloatTensor(labels)

        return return_dict


