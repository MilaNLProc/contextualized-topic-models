import os
from collections import defaultdict
import multiprocessing as mp
import requests

import numpy as np
import datetime

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from contextualized_topic_models.networks.decoding_network import DecoderNetwork


class CTM(object):

    """Class to train CTM model."""

    def __init__(self, input_size, bert_input_size, inference_type, n_components=10, model_type='prodLDA',
                 hidden_sizes=(100, 100), activation='softplus', dropout=0.2,
                 learn_priors=True, batch_size=64, lr=2e-3, momentum=0.99,
                 solver='adam', num_epochs=100, reduce_on_plateau=False, num_data_loader_workers=mp.cpu_count()):
        """
        Initialize CTM model.

        Args
            input_size : int, dimension of input
            n_components : int, number of topic components, (default 10)
            model_type : string, 'prodLDA' or 'LDA' (default 'prodLDA')
            hidden_sizes : tuple, length = n_layers, (default (100, 100))
            activation : string, 'softplus', 'relu', (default 'softplus')
            dropout : float, dropout to use (default 0.2)
            learn_priors : bool, make priors a learnable parameter (default True)
            batch_size : int, size of batch to use for training (default 64)
            lr : float, learning rate to use for training (default 2e-3)
            momentum : float, momentum to use for training (default 0.99)
            solver : string, optimizer 'adam' or 'sgd' (default 'adam')
            num_epochs : int, number of epochs to train for, (default 100)
            reduce_on_plateau : bool, reduce learning rate by 10x on plateau of 10 epochs (default False)
            num_data_loader_workers : int, number of data loader workers (default cpu_count). set it to 0 if you are
            using Windows
        """
        assert isinstance(input_size, int) and input_size > 0,\
            "input_size must by type int > 0."
        assert isinstance(n_components, int) and input_size > 0,\
            "n_components must by type int > 0."
        assert model_type in ['LDA', 'prodLDA'],\
            "model must be 'LDA' or 'prodLDA'."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu'], \
            "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."
        assert isinstance(learn_priors, bool), "learn_priors must be boolean."
        assert isinstance(batch_size, int) and batch_size > 0,\
            "batch_size must be int > 0."
        assert lr > 0, "lr must be > 0."
        assert isinstance(momentum, float) and momentum > 0 and momentum <= 1,\
            "momentum must be 0 < float <= 1."
        assert solver in ['adam', 'sgd'], "solver must be 'adam' or 'sgd'."
        assert isinstance(reduce_on_plateau, bool),\
            "reduce_on_plateau must be type bool."
        assert isinstance(num_data_loader_workers, int) and num_data_loader_workers >= 0, \
            "num_data_loader_workers must by type int >= 0. set 0 if you are using windows"

        self.input_size = input_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.batch_size = batch_size
        self.lr = lr
        self.bert_size = bert_input_size
        self.momentum = momentum
        self.solver = solver
        self.num_epochs = num_epochs
        self.reduce_on_plateau = reduce_on_plateau
        self.num_data_loader_workers = num_data_loader_workers

        # init inference avitm network
        self.model = DecoderNetwork(
            input_size, self.bert_size, inference_type, n_components, model_type, hidden_sizes, activation,
            dropout, learn_priors)

        # init optimizer
        if self.solver == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, betas=(self.momentum, 0.99))
        elif self.solver == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=self.momentum)

        # init lr scheduler
        if self.reduce_on_plateau:
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=10)

        # performance attributes
        self.best_loss_train = float('inf')

        # training atributes
        self.model_dir = None
        self.train_data = None
        self.nn_epoch = None

        # learned topics
        self.best_components = None

        # Use cuda if available
        if torch.cuda.is_available():
            self.USE_CUDA = True
        else:
            self.USE_CUDA = False

        if self.USE_CUDA:
            self.model = self.model.cuda()

    def _loss(self, inputs, word_dists, prior_mean, prior_variance,
              posterior_mean, posterior_variance, posterior_log_variance):

        # KL term
        # var division term
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)
        # diff means term
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum(
            (diff_means * diff_means) / prior_variance, dim=1)
        # logvar det division term
        logvar_det_division = \
            prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
        # combine terms
        KL = 0.5 * (
            var_division + diff_term - self.n_components + logvar_det_division)

        # Reconstruction term
        RL = -torch.sum(inputs * torch.log(word_dists + 1e-10), dim=1)

        loss = KL + RL

        return loss.sum()

    def _train_epoch(self, loader):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        samples_processed = 0

        for batch_samples in loader:
            # batch_size x vocab_size
            X = batch_samples['X']
            X_bert = batch_samples['X_bert']

            if self.USE_CUDA:
                X = X.cuda()
                X_bert = X_bert.cuda()

            # forward pass
            self.model.zero_grad()
            prior_mean, prior_variance, \
                posterior_mean, posterior_variance, posterior_log_variance, \
                word_dists = self.model(X, X_bert)

            # backward pass
            loss = self._loss(
                X, word_dists, prior_mean, prior_variance,
                posterior_mean, posterior_variance, posterior_log_variance)
            loss.backward()
            self.optimizer.step()

            # compute train loss
            samples_processed += X.size()[0]
            train_loss += loss.item()

        train_loss /= samples_processed

        return samples_processed, train_loss

    def fit(self, train_dataset, save_dir=None):
        """
        Train the AVITM model.

        Args
            train_dataset : PyTorch Dataset classs for training data.
            val_dataset : PyTorch Dataset classs for validation data.
            save_dir : directory to save checkpoint models to.
        """
        # Print settings to output file
        print("Settings: \n\
               N Components: {}\n\
               Topic Prior Mean: {}\n\
               Topic Prior Variance: {}\n\
               Model Type: {}\n\
               Hidden Sizes: {}\n\
               Activation: {}\n\
               Dropout: {}\n\
               Learn Priors: {}\n\
               Learning Rate: {}\n\
               Momentum: {}\n\
               Reduce On Plateau: {}\n\
               Save Dir: {}".format(
                   self.n_components, 0.0,
                   1. - (1./self.n_components), self.model_type,
                   self.hidden_sizes, self.activation, self.dropout, self.learn_priors,
                   self.lr, self.momentum, self.reduce_on_plateau, save_dir))

        self.model_dir = save_dir
        self.train_data = train_dataset

        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_data_loader_workers)

        # init training variables
        train_loss = 0
        samples_processed = 0

        # train loop
        for epoch in range(self.num_epochs):
            self.nn_epoch = epoch
            # train epoch
            s = datetime.datetime.now()
            sp, train_loss = self._train_epoch(train_loader)
            samples_processed += sp
            e = datetime.datetime.now()

            # report
            print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: {}\tTime: {}".format(
                epoch+1, self.num_epochs, samples_processed,
                len(self.train_data)*self.num_epochs, train_loss, e - s))

            # save best
            if train_loss < self.best_loss_train:
                self.best_loss_train = train_loss
                self.best_components = self.model.beta

                if save_dir is not None:
                    self.save(save_dir)

    def get_thetas(self, dataset):
        """Predict input."""
        self.model.eval()

        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_data_loader_workers)

        with torch.no_grad():
            collect_theta = []
            for batch_samples in loader:
                # batch_size x vocab_size
                X = batch_samples['X']
                X_bert = batch_samples['X_bert']

                if self.USE_CUDA:
                    X = X.cuda()
                    X_bert = X_bert.cuda()

                # forward pass
                self.model.zero_grad()
                collect_theta.extend(self.model.get_theta(X, X_bert).cpu().numpy().tolist())
            return collect_theta


    def predict(self, dataset, k=10):
        """Predict input."""
        self.model.eval()

        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_data_loader_workers)

        preds = []

        with torch.no_grad():
            for batch_samples in loader:
                # batch_size x vocab_size
                X = batch_samples['X']
                X_bert = batch_samples['X_bert']

                if self.USE_CUDA:
                    X = X.cuda()
                    X_bert = X_bert.cuda()

                # forward pass
                self.model.zero_grad()
                _, _, _, _, word_dists = self.model(X, X_bert)

                _, indices = torch.sort(word_dists, dim=1)
                preds += [indices[:, :k]]

            preds = torch.cat(preds, dim=0)

        return preds

    def score(self, scorer='coherence', k=10, topics=5):
        """Score model."""
        if scorer == 'perplexity':
            # score = perplexity_score(truth, preds)
            raise NotImplementedError("Not implemented yet.")
        elif scorer == 'coherence':
            score = self._get_coherence(k, topics=topics)
        else:
            raise ValueError("Unknown score type!")

        return score

    def _get_coherence(self, k=10, topics=5):
        """Get coherence using palmetto web service."""
        component_dists = self.best_components
        base_url = 'https://palmetto.demos.dice-research.org/service/cv?words='
        scores = []
        i = 0
        while i < topics:
            print(i)
            t = np.random.randint(0, self.n_components)
            _, idxs = torch.topk(component_dists[t], k)
            component_words = [self.train_data.idx2token[idx]
                               for idx in idxs.cpu().numpy()]
            url = base_url + '%20'.join(component_words)
            print(url)
            try:
                score = float(requests.get(url, timeout=300).content)
                scores += [score]
                i += 1
            except requests.exceptions.Timeout:
                print("Attempted scoring timed out.  Trying again.")
                continue
        return np.mean(scores)

    def get_topics(self, k=10):
        """
        Retrieve topic words.

        Args
            k : (int) number of words to return per topic, default 10.
        """
        assert k <= self.input_size, "k must be <= input size."
        component_dists = self.best_components
        topics = defaultdict(list)
        for i in range(self.n_components):
            _, idxs = torch.topk(component_dists[i], k)
            component_words = [self.train_data.idx2token[idx]
                               for idx in idxs.cpu().numpy()]
            topics[i] = component_words
        return topics

    def get_topic_lists(self, k=10):
        """
        Retrieve the lists of topic words.

        Args
            k : (int) number of words to return per topic, default 10.
        """
        assert k <= self.input_size, "k must be <= input size."
        component_dists = self.best_components
        topics = []
        for i in range(self.n_components):
            _, idxs = torch.topk(component_dists[i], k)
            component_words = [self.train_data.idx2token[idx]
                               for idx in idxs.cpu().numpy()]
            topics.append(component_words)
        return topics

    def _format_file(self):
        model_dir = "AVITM_nc_{}_tpm_{}_tpv_{}_hs_{}_ac_{}_do_{}_lr_{}_mo_{}_rp_{}".\
            format(self.n_components, 0.0, 1 - (1./self.n_components),
                   self.model_type, self.hidden_sizes, self.activation,
                   self.dropout, self.lr, self.momentum,
                   self.reduce_on_plateau)
        return model_dir

    def save(self, models_dir=None):
        """
        Save model.

        Args
            models_dir: path to directory for saving NN models.
        """
        if (self.model is not None) and (models_dir is not None):

            model_dir = self._format_file()
            if not os.path.isdir(os.path.join(models_dir, model_dir)):
                os.makedirs(os.path.join(models_dir, model_dir))

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(models_dir, model_dir, filename)
            with open(fileloc, 'wb') as file:
                torch.save({'state_dict': self.model.state_dict(),
                            'dcue_dict': self.__dict__}, file)

    def load(self, model_dir, epoch):
        """
        Load a previously trained model.

        Args
            model_dir: directory where models are saved.
            epoch: epoch of model to load.
        """
        epoch_file = "epoch_"+str(epoch)+".pth"
        model_file = os.path.join(model_dir, epoch_file)
        with open(model_file, 'rb') as model_dict:
            checkpoint = torch.load(model_dict)

        for (k, v) in checkpoint['dcue_dict'].items():
            setattr(self, k, v)

        self._init_nn() #TODO implement this method
        self.model.load_state_dict(checkpoint['state_dict'])
