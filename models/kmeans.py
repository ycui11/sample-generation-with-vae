"""Implements the k-means algorithm.
"""

import numpy as np
import scipy
from scipy import stats
import math


class KMeans(object):
    def __init__(self, n_dims, n_components=10, max_iter=100):
        """Initialize a KMeans GMM model
        Args:
            n_dims(int): The dimension of the feature.
            n_components(int): The number of cluster in the model.
            max_iter(int): The number of iteration to run EM.
        """
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter

        # Randomly initialize model parameters using Gaussian distribution of 0
        # mean and unit variance.
        # np.array of size (n_components, n_dims)
        self._mu=np.random.normal(0,1,(self._n_components,self._n_dims))
    def fit(self, x):
        """Runs EM step for max_iter number of times.

        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
            ndims).
        """
        for i in range(self._max_iter):
            r_ik=self._e_step(x)
            self._m_step(x,r_ik)
        pass

    def _e_step(self, x):
        """Update cluster assignment.

        Computes the posterior probability of p(z|x), z is the latent cluster
        variable.

        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            r_ik(numpy.ndarray): Array containing the cluster assignment of
                each example, dimension (N,).
        """
        prob=np.asarray([[np.dot(x_i-y_k, x_i-y_k) for y_k in self._mu] for x_i in x])
        dist=1/math.sqrt(2*math.pi)*np.exp((-1.0/2)*prob)
        r_ik=np.argmax(dist,axis=1)
        return r_ik
    def _m_step(self, x, r_ik):
        """Update cluster mean.

        Updates self_mu according to the cluster assignment.

        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            r_ik(numpy.ndarray): Array containing the cluster assignment of
                each example, dimension (N,).
        """
        cluster=list(set(r_ik))
        my= np.asarray([x[r_ik == k].mean(axis = 0) for k in cluster])
        for i in range(len(cluster)):
            self._mu[cluster[i]]=my[i]
        pass

    def get_posterior(self, x):
        """Computes cluster assignment.

        Computes the posterior probability of p(z|x), z is the latent cluster
        variable.

        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            r_ik(numpy.ndarray): Array containing the cluster assignment of
            each example, dimension (N,).
        """
        self.fit(x)
        prob=np.asarray([[np.dot(x_i-y_k, x_i-y_k) for y_k in self._mu] for x_i in x])
        dist=1/math.sqrt(2*math.pi)*np.exp((-1.0/2)*prob)
        r_ik=np.argmax(dist,axis=1)
        return r_ik

    def supervised_fit(self, x, y):
        """Assign each cluster with a label through counting.

        For each cluster, find the most common digit using the provided (x,y)
        and store it in self.cluster_label_map.

        self.cluster_label_map should be a list of length n_components,
        where each element maps to the most common digit in that cluster.
        (e.g. If self.cluster_label_map[0] = 9. Then the most common digit
        in cluster 0 is 9.

        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            y(numpy.ndarray): Array containing the label of dimension (N,)
        """
        r_ik=self.get_posterior(x)
        lookup_map=np.zeros((self._n_components,len(set(y))))
        self.cluster_label_map = [0]*self._n_components
        for i in range(y.shape[0]):
            lookup_map[int(r_ik[i])][int(y[i])]+=1
        for i in range(self._n_components):
            self.cluster_label_map[i]=np.argmax(lookup_map,axis=1)[i]
        print(self.cluster_label_map,lookup_map)
        pass

    def supervised_predict(self, x):
        """Predict a label for each example in x.
        Find the get the cluster assignment for each x, then use
        self.cluster_label_map to map to the corresponding digit.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            y_hat(numpy.ndarray): Array containing the predicted label for each
                x, dimension (N,)
        """
        y_hat = []
        r_ik = self.get_posterior(x)
        for i in range(len(x)):
            y_hat.append(self.cluster_label_map[int(r_ik[i])])
        return np.array(y_hat)
