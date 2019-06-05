# -*- coding: utf-8 -*-
import matlab_wrapper
import numpy as np
import os
import random
import sys
sys.path.insert(0, '../')
from data_utils import logger as log


class AKDE(object):

    def __init__(self, workspace=None):
        """
        Instantiate an AKDE model.

        Parameters
        ----------
        workspace : MatlabWorkspace
            The matlab workspace object on which to build the AKDE model.
            If None is used, a new MatlabWorkspace object is instantiated for this model.

        Returns
        -------
        AKDE
            Newly instantiated AKDE object.

        """
        # generate random ID which is most likely unique
        self.id = random.randint(0, 2**15)
        self.trained = False
        try:
            # create a new workspace if an existing workspace was not provided
            if workspace is None:
                workspace = MatlabWorkspace()
            self.workspace = workspace
        except Exception, e:
            log.warn("Failed to create AKDE: {}".format(e.message))

    def fit(self, data):
        """
        Fit the AKDE model to data.

        Parameters
        ----------
        data : ndarray
            The data against which to fit the model.
            Must be a NxM sized ndarray where N is the number of instances
            and M is the number of variables to fit.

        Returns
        -------
        AKDE
            The self AKDE object.

        """
        try:
            self.workspace.akde_fit(data, self.id)
            self.trained = True
        except Exception, e:
            log.warn("Failed to fit AKDE: {}".format(e.message))
        return self

    def predict(self, samples):
        """
        Make probability predictions for samples.

        Parameters
        ----------
        samples : ndarray

        Returns
        -------
        ndarray
            A Nx1 sized ndarray of probability predictions.
            If an exception is produced during operation, an empty ndarray is returned.

        """
        try:
            predictions = self.workspace.akde_predict(samples, self.id)
            if predictions is None:
                return np.array([])
            return predictions
        except Exception, e:
            log.warn("Failed to make AKDE predictions: {}".format(e.message))
            return np.array([])

    def sample(self, n_samples):
        """
        Generate samples from model.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
            Must be 1 or greater.

        Returns
        -------
        ndarray
            Samples from @kde as ndarray with a shape of (n_samples, variables).
            If an exception is produced during operation, an empty ndarray is returned.

        """
        try:
            samples = self.workspace.akde_sample(n_samples, self.id)
            if samples is None:
                return np.array([])
            return samples
        except Exception, e:
            log.warn("Failed to make generate samples: {}".format(e.message))
            return np.array([])


class MatlabWorkspace(object):

    def __init__(self):
        """
        Instantiate a MatlabWorkspace by opening a new matlab session.

        Parameters
        ----------

        Returns
        -------
        MatlabWorkspace

        """
        # start new matlab workspace
        self.session = matlab_wrapper.MatlabSession(options="-nodisplay -nodesktop -nosplash")

        # add akde matlab files to the workspace's search path
        path = os.path.dirname(__file__)
        self.session.eval('addpath(\"{}\")'.format(path))
        self.session.eval('addpath(\"{}\")'.format(os.path.join(path, '@kde')))

    def akde_fit(self, data, id):
        """
        Fit @kde on provided data and store in the matlab workspace.
        The model is identified using the ID provided to this function.

        Parameters
        ----------
        data : ndarray
            ndarray representing features for instances w/ shape (N,f).
        id : int
            Identifier used to identify the model in the shared workspace.

        Returns
        -------
        None

        """
        # discrete vector identifies which features in the data should be modeled as discrete
        # currently, this implementation only models continuous
        discrete_vector = np.zeros(data.shape[0])

        # add data and isDiscVec to matlab session
        self.session.put('data_{id}'.format(id=id), data)
        self.session.put('isDiscVec_{id}'.format(id=id), discrete_vector)

        # generate the AKDE
        self.session.eval('model_{id} = MakeKDE(data_{id}, isDiscVec_{id})'.format(id=id))

    def akde_predict(self, samples, id):
        """
        Make probability predictions for provided samples using the selected @kde.

        Parameters
        ----------
        samples : ndarray
            numpy array w/ shape (N,f)
        id : int
            integer identifier used to identify the model in the shared workspace

        Returns
        -------
        ndarray
            numpy array of probabilities

        """
        self.session.put('samples_{id}'.format(id=id), samples)
        self.session.eval('pred_{id} = evaluate(model_{id}, samples_{id})'.format(id=id))
        return self.session.get('pred_{id}'.format(id=id))

    def akde_sample(self, count, id):
        """
        Generate samples from selected @kde.

        Parameters
        ----------
        count : int
            number of samples to generate
        id : int
            identifier used to identify the model in the shared workspace

        Returns
        -------
        ndarray
            numpy array of samples

        """
        self.session.put('n_samples_{id}'.format(id=id), count)
        self.session.eval('[points_{id},ind] = sample(model_{id}, n_samples_{id})'.format(id=id))
        return self.session.get('points_{id}'.format(id=id))

    def pairwise_mi(self, data):
        """
        Use @kde to estimate pair-wise mutual information.

        Parameters
        ----------
        data : ndarray
            numpy array of dimensions Nx2

        Returns
        -------
        float
            Mutual information estimation.
            Returns 0.0 if there is an exception during operation.

        """
        try:
            data = data.transpose((1, 0))
            self.session.put('data', data)
            self.session.eval('mi = MutualInfo(data)')
            return self.session.get('mi')
        except Exception, e:
            log.warn("Failed to produce MI estimate: {}".format(e.message))
            return 0.0

    def __exit__(self):
        try:
            del self.session
        except:
            pass

