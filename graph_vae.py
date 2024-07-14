import tensorflow as tf
from graph_layers import GraphConvolution, GraphLinear, InnerProductDecoder
import numpy as np
import pyedflib


# We assume that the 3 loss function is defined and we include the new proposed decoder model

subject_num = 109 # Size of dataset
run_num = 14
task_num = 6
n = 64 # Number of EEG electrodes??
data_length = 9600 #
# Fs means sampling frequency



# Graph Variational Autoencoder
class GCNModelVAE(tf.keras.Model):
    def __init__(self, num_features, num_nodes, features_nonzero, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.hidden_dim = 16 # hyperparameter
        self.embedding_dimension = 1 # hyperparameter
        self.hidden1 = GraphConvolution(input_dim=self.input_dim,
                                        output_dim=self.hidden_dim, num=1,
                                        act=lambda x: x)  # Convolutional layer
        self.hidden2 = GraphConvolution(input_dim=self.hidden_dim,
                                        output_dim=self.embedding_dimension * 2, num=2,
                                        act=lambda x: x)
        self.d = InnerProductDecoder(act=lambda x: x)
        self.d1 = GraphConvolution(input_dim=1,
                                   output_dim=self.n_samples, num=3,
                                   act=lambda x: x) # Use new decoder model and loss function = 3???

    # Encoder feedforward
    def encoder(self, inputs, adj, rate):
        x = self.hidden1(inputs, adj, rate)
        x = self.hidden2(x, adj, rate)
        mean, logvar = tf.split(x, num_or_size_splits=2, axis=2)
        return mean, logvar

    # Reparametrization trick
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal([self.n_samples, self.dimension])
        return eps * (tf.exp(logvar)) + mean

    # Decoding model
    def decoder(self, z, adj, rate=0., apply_sigmoid=False):
        logits = z
        logits = self.d(logits, 0.)
        feature = tf.ones((logits.shape[0], logits.shape[1], 1))
        logits = self.d1(feature, logits, rate)
        logits = tf.reshape(logits, [-1, self.n_samples * self.n_samples])
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9)  # learning rate


# VAE optimizer model
class OptimizerVAE(object):
    def __init__(self, model, num_nodes, num_features, norm):
        self.norm = norm
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    def log_normal_pdf(self, sample, mean, logsd, raxis=[1, 2]):
        logvar = 2 * logsd
        log2pi = tf.math.log(2. * np.pi)
        out = tf.reduce_sum(-.5 * (tf.multiply((sample - mean) ** 2., tf.exp(-logvar)) + logvar + log2pi), axis=raxis)
        return out

    def bernoulli_log_density(self, logit, x):
        b = (x * 2) - 1
        return - tf.math.log(1 + tf.exp(-tf.multiply(b, logit)))

    def loss(self, y, x, adj, rate, model):
        mean, logvar = model.encoder(x, adj, rate)
        reparam = model.reparameterize(mean, logvar)
        reconstruct = model.decoder(reparam, adj, rate)
        preds_sub = tf.reshape(reconstruct, [-1, self.num_nodes, self.num_nodes])
        logpz = self.log_normal_pdf(reparam, 0., 0.)
        logqz_x = self.log_normal_pdf(reparam, mean, logvar)
        logpx_z = tf.reduce_sum(self.bernoulli_log_density(preds_sub, tf.cast(y, tf.float32)), [1, 2])
        return -tf.reduce_mean(logpx_z - ((logpz - logqz_x)))

    def loss2(self, y, x, adj, rate, model):
        mean, logvar = model.encoder(x, adj, rate)
        reparam = model.reparameterize(mean, logvar)
        reconstruct = model.decoder(reparam, adj, rate)
        preds_sub = tf.reshape(reconstruct, [-1, self.num_nodes, self.num_nodes])
        cost = self.norm * tf.reduce_mean(tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y, tf.float32),
                                                    logits=preds_sub), [1, 2]))
        kl = (0.5 / self.num_nodes) * \
             tf.reduce_mean(tf.reduce_sum(1 \
                                          + 2 * logvar \
                                          - tf.square(mean) \
                                          - tf.square(tf.exp(logvar)), [1, 2]))
        cost -= kl
        return cost

    def train_step(self, y, x, adj, rate, model):
        with tf.GradientTape() as tape:
            cost = self.loss(y, x, adj, rate, model)

        assert not np.any(np.isnan(cost.numpy()))
        gradients = tape.gradient(cost, model.trainable_variables)
        opt_op = self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return cost
