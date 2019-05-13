import matlab_wrapper
import numpy as np
import os
import random


class AKDE(object):

    def __init__(self, workspace=None):
        """
        Instantiate an AKDE model.
        :param workspace: matlab workspace on which to save the model
        """
        # generate random ID which is most likely unique
        self.id = random.randint(0, 2**15)

        # create a new workspace if an existing workspace was not provided
        if workspace is None:
            workspace = MatlabWorkspace()
        self.workspace = workspace
        self.trained = False

    def fit(self, data):
        """
        Fit the AKDE model to data.
        :param data: numpy array w/ shape (N,f)
        :return: self
        """
        self.workspace.akde_fit(data, self.id)
        self.trained = True
        return self

    def predict(self, samples):
        """
        Make probability predictions for samples.
        :param samples: numpy array w/ shape (n,f)
        :return: probability predictions for samples as numpy array w/ shape (n,)
        """
        predictions = self.workspace.akde_predict(samples, self.id)
        if predictions is None:
            return np.array([])
        return predictions

    def sample(self, n_samples):
        """
        Generate samples from model.
        :param n_samples: number of samples to generate
        :return: random samples from @kde as numpy array w/ shape (n_samples,f)
        """
        samples = self.workspace.akde_sample(n_samples, self.id)
        if samples is None:
            return np.array([])
        return samples


class MatlabWorkspace(object):

    def __init__(self):

        # start new matlab workspace
        self.session = matlab_wrapper.MatlabSession()

        # add akde matlab files to the workspace's search path
        path = os.path.dirname(__file__)
        self.session.eval('addpath(\"{}\")'.format(path))
        self.session.eval('addpath(\"{}\")'.format(os.path.join(path, '@kde')))

    def akde_fit(self, data, id):
        """
        Fit @kde on provided data and store in the matlab workspace.
        The model is identified using the ID provided to this function.
        :param data: numpy array representing features for instances w/ shape (N,f)
        :param id: integer identifier used to identify the model in the shared workspace
        :return: None
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
        :param samples: numpy array w/ shape (N,f)
        :param id: integer identifier used to identify the model in the shared workspace
        :return: numpy array of probabilities
        """
        self.session.put('samples_{id}'.format(id=id), samples)
        self.session.eval('pred_{id} = evaluate(model_{id}, samples_{id})'.format(id=id))
        return self.session.get('pred_{id}'.format(id=id))

    def akde_sample(self, count, id):
        """
        Generate samples from selected @kde.
        :param count: number of samples to generate
        :param id: integer identifier used to identify the model in the shared workspace
        :return: numpy array of samples
        """
        self.session.put('n_samples_{id}'.format(id=id), count)
        self.session.eval('[points_{id},ind] = sample(model_{id}, n_samples_{id})'.format(id=id))
        return self.session.get('points_{id}'.format(id=id))

    def pairwise_mi(self, data):
        """
        use @kde to estimate pair-wise mutual information
        :param data: numpy array of dimension w/ shape (N,2)
        :return: mutual information estimation
        """
        data = data.transpose((1, 0))
        self.session.put('data', data)
        self.session.eval('mi = MutualInfo(data)')
        return self.session.get('mi')

    def __exit__(self):
        del self.session

