from copy import deepcopy
import scipy
import tensorflow as tf
from graph_clustering import A_binarize, creating_label
from graph_features import graph_norm
from graph_layers import GraphConvolution, GraphLinear, InnerProductDecoder
import numpy as np
import pyedflib
import time
from preprocessing_functions import preprocess_data
import os
import pandas as pd

# We assume that the 3 loss function is defined and we include the new proposed decoder model
# Experiment stuff
subject_num = 149  # Size of dataset
run_num = 1  # Number of "experiences" of each task (this should be one)
num_electrodes = 60  # Number of EEG electrodes (this is excluding ["Iz", "I1", "I2", "Resp", "PO4", "PO3", "FT9", "Status"])
data_length = 60500  # Minimum data length, cut off anything outside this data length
BASE_PATH = r"C:\Users\timmy\Downloads\park-eeg"

# Returns the dataset and labels
def load_dataset():
    # Format is sub-NUM_task-Rest_eeg.edf
    data = []
    for subject in range(subject_num):
        subject_list = []
        for run in range(run_num):
            file_name = r"\sub-{}_task-Rest_eeg.edf".format(str(subject + 1).zfill(3))
            try:
                f = pyedflib.EdfReader(BASE_PATH + file_name)
            except Exception as e:
                print(e)
                continue

            electrode_list = []
            for electrode in range(num_electrodes):  # Electrodes are zero-indexed
                # Iz, I1, I2, PO4, PO3, FT9 is not present in all subjects... exclude
                # Status/Resp... I'm assuming is ground/reference electrode
                # Regardless, both status/resp are not present in all subjects
                if f.getLabel(electrode) in ("Iz", "I1", "I2", "Resp", "PO4", "PO3", "FT9", "Status"):
                    continue
                electrode_list.append(f.readSignal(electrode)[:data_length])
            subject_list.append(electrode_list)
            f._close()
            del f  # Don't read all files into memory

        data.append(subject_list)
        print("Subject", subject, "done!")

    raw_labels = pd.read_csv("participants.tsv", sep="\t")
    # Get rid of #68 and add labels
    new_data = []
    labels = []
    for index, subject in enumerate(data):

        if not subject:
            continue

        new_data.append(subject)

        query = "sub-" + str(index + 1).zfill(3)
        results = raw_labels[(raw_labels["participant_id"] == query)]
        labels.append(results.iloc[0]["GROUP"])

        print("Label", index, "done!")

    return np.array(new_data), np.array([i == "PD" for i in labels])


Binary = True
partial_subject = False
part_channel = False
verbose = True
nb_run = 5  # 5-fold cross validation
step = 160 * 0 + 80  # 1-window*alpha%
Fs = 500  # Sampling frequency (hz)
Ts = 1 / Fs  # Time in seconds ??

# Initialized numpy arrays for in-train data
accuracy = np.zeros((nb_run, 1))
accuracy2 = np.zeros((nb_run, 1))
Computational_time = np.zeros((nb_run, 1))
num_epoch = np.zeros((nb_run, 1))
full_time = np.zeros((nb_run, 1))
roc_auc = np.zeros((nb_run, 1))
EER = np.zeros((nb_run, 1))

x_raw_all, labels = load_dataset()  # import data for all subject icluding all tasks

loss_function = 3  # 3 loss function is defined
decoder_adj = True  # include new decoder model

FLAGS_features = False  # Whether or not to include pre-defined features
features_init_train = None
features_init_test = None


def invlogit(z):  # convert decoded adjancy matrix to original space
    return 1 - 1 / (1 + np.exp(z))


# Graph Variational Autoencoder
class GCNModelVAE(tf.keras.Model):
    def __init__(self, num_features, num_nodes, features_nonzero, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.hidden_dim = 16  # hyperparameter
        self.embedding_dimension = 1  # hyperparameter
        self.hidden1 = GraphConvolution(input_dim=self.input_dim,
                                        output_dim=self.hidden_dim, num=1,
                                        act=lambda x: x)  # Convolutional layer
        self.hidden2 = GraphConvolution(input_dim=self.hidden_dim,
                                        output_dim=self.embedding_dimension * 2, num=2,
                                        act=lambda x: x)
        self.d = InnerProductDecoder(act=lambda x: x)
        self.d1 = GraphConvolution(input_dim=1,
                                   output_dim=self.n_samples, num=3,
                                   act=lambda x: x)  # Use new decoder model and loss function = 3???

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


def Adj_matrix(train_x, test_x):
    if (Binary):
        # Change weighted matrix to binary matrix with threshold
        percentile = 0.75
        adj_train = A_binarize(A_matrix=train_x, percent=percentile, sparse=False)
        adj_test = A_binarize(A_matrix=test_x, percent=percentile, sparse=False)
        # sparse matrix
    else:
        adj_train = deepcopy(train_x)
        adj_test = deepcopy(test_x)
    # consider part of the graph
    if (part_channel):
        index = creating_label(ztr, y_train, subject_num, method='mean_sort')
        adj_train = adj_train[:, :, index]
        adj_train = adj_train[:, index]
        adj_test = adj_test[:, :, index]
        adj_test = adj_test[:, index]
    print("sparsity: ", scipy.sparse.issparse(adj_train[9]))  # check sparsity
    print("rank: ", np.linalg.matrix_rank(adj_train[9]))  # check matrix rank
    return adj_train, adj_test


for run in range(nb_run):
    t_start = time.time()

    print("Applying ICA...")
    sec = 12  # Number of seconds in window?

    train_x, test_x, y_train, y_test = preprocess_data(x_raw_all[:, 0], labels, 0, Fs,
                                                       filt=False, ICA=True, A_Matrix='cov', sec=sec)

    print("Creating brain graph....")
    adj_train, adj_test = Adj_matrix(train_x, test_x)  # Creating functional connectivity graph

    print("Preprocessing...")
    # Compute number of nodes
    num_nodes = adj_train.shape[1]

    # If features are not used, replace feature matrix by identity matrix
    I_train = np.tile(np.eye(adj_train.shape[1]), adj_train.shape[0]).T.reshape(-1, adj_train.shape[1],
                                                                                adj_train.shape[1])
    I_test = np.tile(np.eye(adj_test.shape[1]), adj_test.shape[0]).T.reshape(-1, adj_test.shape[1], adj_test.shape[1])

    features = np.ones((adj_train.shape[0], adj_train.shape[1], 1))

    # Preprocessing on node features
    num_features = features.shape[2]
    features_nonzero = np.count_nonzero(features) // features.shape[0]

    # Normalization and preprocessing on adjacency matrix
    adj_norm_train = graph_norm(adj_train)
    adj_label_train = adj_train + I_train

    adj_norm_test = graph_norm(adj_test)
    adj_label_test = adj_test + I_test

    features_test = np.ones((adj_test.shape[0], adj_test.shape[1], 1))

    train_dataset = (tf.data.Dataset.from_tensor_slices((adj_norm_train, adj_label_train, features))
                     .shuffle(len(adj_norm_train)).batch(64))
    norm = adj_train.shape[1] * adj_train.shape[1] / float((adj_train.shape[1] * adj_train.shape[1]
                                                            - (adj_train.sum() / adj_train.shape[0])) * 2)

    print("Initializing...")
    # VAE model
    GVAE_model = GCNModelVAE(num_features, num_nodes, features_nonzero)
    # Optimizer
    optimizer = OptimizerVAE(model=GVAE_model, num_nodes=num_nodes,
                             num_features=num_features, norm=norm)

    print("Training...")

    prev_cost = float("inf")
    stop_val = 0
    stop_num = 10
    FLAGS_shuffle = False
    for epoch in range(1000):
        t = time.time()
        # Compute average loss
        loss = 0
        for adj, label, x in train_dataset:
            loss += optimizer.train_step(label, tf.cast(x, tf.float32), tf.cast(adj, tf.float32), 0.5, GVAE_model)
        avg_cost = loss.numpy() / (len(adj_train))

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(round(avg_cost, 3)),
              "time=", "{:.5f}".format(round(time.time() - t, 3)))
        Computational_time[run] += (time.time() - t)
        num_epoch[run] += 1

        # When to stop the iteration
        if (prev_cost < avg_cost):
            stop_val += 1
            if (stop_val == stop_num):
                break
        else:
            stop_val = 0
            prev_cost = avg_cost

    Computational_time[run] = Computational_time[run] / num_epoch[run]

    print("Time for each epoch:", np.round(Computational_time[run], 3))

    meanr, logvarr = GVAE_model.encoder(tf.cast(features, tf.float32), tf.cast(adj_norm_train, tf.float32), 0.)
    ztr = GVAE_model.reparameterize(meanr, logvarr)
    meane, logvare = GVAE_model.encoder(tf.cast(features_test, tf.float32), tf.cast(adj_norm_test, tf.float32), 0.)
    zte = GVAE_model.reparameterize(meane, logvare)

    train_feature = deepcopy(ztr).numpy().reshape(len(ztr), -1)
    test_feature = deepcopy(zte).numpy().reshape(len(zte), -1)

for i in range(nb_run):
    t_start = time.time()
    if verbose:
        print("Creating Adjacency matrix...")

    sec = 12  # # of seconds
    train_x, test_x, y_train, y_test = preprocess_data(x_raw_all[:, 0], labels, i, Fs, dataset2=False,
                                                       filt=False, ICA=True, A_Matrix='cov', sec=sec)
    # A_matrix = 'cov' 'plv' 'iplv' 'pli' 'AEC'

    adj_train, adj_test = Adj_matrix(train_x, test_x)  # Creating brain graph
    # Initialization
    if verbose:
        print("Preprocessing and Initializing...")

    # Compute number of nodes
    num_nodes = adj_train.shape[1]

    # If features are not used, replace feature matrix by identity matrix
    I_train = np.tile(np.eye(adj_train.shape[1]), adj_train.shape[0]).T.reshape(-1, adj_train.shape[1],
                                                                                adj_train.shape[1])
    I_test = np.tile(np.eye(adj_test.shape[1]), adj_test.shape[0]).T.reshape(-1, adj_test.shape[1], adj_test.shape[1])

    if not FLAGS_features:
        features = np.ones((adj_train.shape[0], adj_train.shape[1], 1))
        # features = deepcopy(I)
    else:
        features = deepcopy(features_init_train)

    # Preprocessing on node features
    num_features = features.shape[2]
    features_nonzero = np.count_nonzero(features) // features.shape[0]
    # Normalization and preprocessing on adjacency matrix
    adj_norm = graph_norm(adj_train)
    adj_label = adj_train + I_train

    adj_norm_test = graph_norm(adj_test)
    adj_label_test = adj_test + I_test
    if not FLAGS_features:
        features_test = np.ones((adj_test.shape[0], adj_test.shape[1], 1))
        # features_test = deepcopy(I_test)
    else:
        features_test = deepcopy(features_init_test)
    # """
    train_dataset = (tf.data.Dataset.from_tensor_slices((adj_norm, adj_label, features))
                     .shuffle(len(adj_norm)).batch(64))
    norm = adj_train.shape[1] * adj_train.shape[1] / float((adj_train.shape[1] * adj_train.shape[1]
                                                            - (adj_train.sum() / adj_train.shape[0])) * 2)
    rate_test = 0
    # VAE model
    VAEmodel = GCNModelVAE(num_features, num_nodes, features_nonzero)
    # Optimizer
    opt = OptimizerVAE(model=VAEmodel, num_nodes=num_nodes,
                       num_features=num_features, norm=norm)
    # Model training
    if verbose:
        print("Training...")
    prev_cost = 100000
    stop_val = 0
    stop_num = 10
    FLAGS_shuffle = False
    for epoch in range(1000):
        t = time.time()
        # Compute average loss
        loss = 0
        for adj, label, x in train_dataset:
            loss += opt.train_step(label, tf.cast(x, tf.float32), tf.cast(adj, tf.float32), 0.5, VAEmodel)
        avg_cost = loss.numpy() / (len(adj_train))
        if verbose:
            # Display epoch information
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(round(avg_cost, 3)),
                  "time=", "{:.5f}".format(round(time.time() - t, 3)))
        Computational_time[i] += (time.time() - t)
        num_epoch[i] += 1
        # When to stop the iteration
        if (prev_cost < avg_cost):
            stop_val += 1
            if (stop_val == stop_num):
                break
        else:
            stop_val = 0
            prev_cost = avg_cost
    Computational_time[i] = Computational_time[i] / num_epoch[i]
    print("computational time for each epoch: ", np.round(Computational_time[i], 3))
    if (partial_subject and part_channel):
        test_index = np.where(y_test >= 5)[0]
        n_partial = 32
        n = adj_train.shape[1]
        prev_norm = tf.cast((np.mean(adj_train, keepdims=True, axis=0)), tf.float32)
        A_test = np.tile(graph_norm(prev_norm), len(test_index)).reshape(-1, n, n)
        A_test[:, :n_partial, :n_partial] = graph_norm(adj_test[test_index, :n_partial, :n_partial])
        adj_norm_test[test_index] = A_test
    meanr, logvarr = VAEmodel.encoder(tf.cast(features, tf.float32), tf.cast(adj_norm, tf.float32), 0.)
    ztr = VAEmodel.reparameterize(meanr, logvarr)
    meane, logvare = VAEmodel.encoder(tf.cast(features_test, tf.float32), tf.cast(adj_norm_test, tf.float32), 0.)
    zte = VAEmodel.reparameterize(meane, logvare)
    train_feature = deepcopy(ztr).numpy().reshape(len(ztr), -1)
    test_feature = deepcopy(zte).numpy().reshape(len(zte), -1)
