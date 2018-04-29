"""Variation autoencoder."""

import numpy as np
import tensorflow as tf

from tensorflow import contrib
from tensorflow.contrib import layers


class VariationalAutoencoder(object):
    """Varational Autoencoder.
    """
    def __init__(self, ndims=784, nlatent=2):
        """Initializes a VAE

        Args:
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
        """

        self._ndims = ndims
        self._nlatent = nlatent

        # Create session
        self.session = tf.Session()
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])

        # Build graph.
        self.z_mean, self.z_log_var = self._encoder(self.x_placeholder)
        self.z = self._sample_z(self.z_mean, self.z_log_var)
        self.outputs_tensor = self._decoder(self.z)

        # Setup loss tensor, predict_tensor, update_op_tensor
        self.loss_tensor = self.loss(self.outputs_tensor, self.x_placeholder,
                                     self.z_mean, self.z_log_var)

        self.update_op_tensor = self.update_op(self.loss_tensor,
                                               self.learning_rate_placeholder)

        # Initialize all variables.
        self.session.run(tf.global_variables_initializer())
    
        
        
        
        
    def _sample_z(self, z_mean, z_log_var):
        """Sample z using reparametrize trick.

        Args:
            z_mean (tf.Tensor): The latent mean,
                tensor of dimension (None, 2)
            z_log_var (tf.Tensor): The latent log variance,
                tensor of dimension (None, 2)
        Returns:
            z (tf.Tensor): Random z sampled of dimension (None, 2)
        """
        eps=tf.random_normal(tf.shape(z_mean), 0, 1, dtype=tf.float32)
        z=tf.add(z_mean,tf.multiply(tf.sqrt(tf.exp(z_log_var)),eps))
        return z

    
    def _encoder(self, x,reuse=True):
        """Encoder block of the network.

        Build a two layer network of fully connected layers, with 100 nodes,
        then 50 nodes. Then two output branches each two 2 nodes representing
        the z_mean and z_log_var.

                             |-> 2 (z_mean)
        Input --> 100 --> 50 -
                             |-> 2 (z_log_var)

        Use activation of tf.nn.softplus.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, 784).
        Returns:
            z_mean(tf.Tensor): The latent mean, tensor of dimension (None, 2).
            z_log_var(tf.Tensor): The latent log variance, tensor of dimension
                (None, 2).
        """
        w0 = tf.Variable(tf.truncated_normal([int(self._ndims), 100],stddev=0.1,dtype=tf.float32))
        b0 = tf.constant(0.,shape=[100],dtype=tf.float32)
        h0 = tf.matmul(x, w0) + b0
        h0 = tf.nn.softplus(h0)
        w1 = tf.Variable(tf.truncated_normal([100, 50],stddev=0.1,dtype=tf.float32))
        b1 = tf.constant(0., shape=[50],dtype=tf.float32)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.softplus(h1)
        wo = tf.Variable(tf.truncated_normal([50, self._nlatent * 2],stddev=0.1,dtype=tf.float32))
        bo = tf.constant(0., shape=[self._nlatent * 2],dtype=tf.float32)
        output = tf.matmul(h1, wo) + bo
        z_mean = output[:, :self._nlatent]
        z_log_var = 1e-6 + output[:, self._nlatent:]

        return z_mean, z_log_var

    def _decoder(self, z):
        """From a sampled z, decode back into image.

        Build a three layer network of fully connected layers,
        with 50, 100, 784 nodes. Use activation tf.nn.softplus.
        z (2) --> 50 --> 100 --> 784.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, 2).
        Returns:
            f(tf.Tensor): Predicted score, tensor of dimension (None, 784).
        """

        w0 = tf.Variable(tf.truncated_normal([self._nlatent, 50],stddev=0.1,dtype=tf.float32))
        b0 = tf.constant(0., shape=[50],dtype=tf.float32)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.softplus(h0)
        w1 = tf.Variable(tf.truncated_normal( [50, 100],stddev=0.1,dtype=tf.float32))
        b1 = tf.constant(0, shape=[100],dtype=tf.float32)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.softplus(h1)
        wo = tf.Variable(tf.truncated_normal( [100, 784],stddev=0.1,dtype=tf.float32))
        bo = tf.constant(0, shape=[784],dtype=tf.float32)
        f = tf.nn.sigmoid(tf.matmul(h1, wo) + bo)
        f=tf.clip_by_value(f, 1e-8, 1 - 1e-8)

        return f

    def _latent_loss(self, z_mean, z_log_var):
        """Constructs the latent loss.

        Args:
            z_mean(tf.Tensor): Tensor of dimension (None, 2)
            z_log_var(tf.Tensor): Tensor of dimension (None, 2)

        Returns:
            latent_loss: Tensor of dimension (None,). Sum the loss over
            dimension 1.
        """
        latent_loss =0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_log_var) - tf.log(1e-8 + tf.square(z_log_var)) - 1, 1)
        latent_loss=tf.reduce_mean(latent_loss)
        return latent_loss

    def _reconstruction_loss(self, f, y):
        """Constructs the reconstruction loss.
        Args:
            f(tf.Tensor): Predicted score for each example, dimension (None,
                784).
            y(tf.Tensor): Ground truth for each example, dimension (None, 784)
        Returns:
            Tensor for dimension (None,). Sum the loss over dimension 1.
        """
        loss=tf.reduce_sum(tf.squared_difference(f, y))
        loss=tf.reduce_mean(loss)
        return loss
    def loss(self, f, y, z_mean, z_var):
        """Computes the total loss.

        Computes the sum of latent and reconstruction loss then average over
        dimension 0.

        Returns:
            (1) averged loss of latent_loss and reconstruction loss over
                dimension 0.
        """
        REC=self._reconstruction_loss( f, y)
        KLD=self._latent_loss(z_mean,z_var)
        loss=KLD+REC
        return loss
    def update_op(self, loss, learning_rate):
        """Creates the update optimizer.

        Use tf.train.AdamOptimizer to obtain the update op.

        Args:
            loss: Tensor containing the loss function.
            learning_rate: A scalar, learning rate for gradient descent.
        Returns:
            (1) Update opt tensorflow operation.
        """
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_step

    def generate_samples(self, z_np):
        """Generates a random sample from the provided z_np.

        Args:
            z_np(numpy.ndarray): Numpy array of dimension (batch_size, 2).

        Returns:
            (1) The sampled images (numpy.ndarray) of dimension (batch_size,
                748).
        """

        y=self.session.run(self.outputs_tensor,feed_dict={self.z:z_np})
        return y
