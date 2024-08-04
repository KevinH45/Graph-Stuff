from copy import deepcopy

import numpy as np
import pyedflib
import pandas as pd
import scipy

from graph_clustering import A_binarize
from preprocessing_functions import preprocess_data

# We assume that the 3 loss function is defined and we include the new proposed decoder model
# Experiment stuff
subject_num = 149 # Size of dataset
run_num = 1 # Number of "experiences" of each task (this should be one)
num_electrodes = 60 # Number of EEG electrodes (this is excluding ["Iz", "I1", "I2", "Resp", "PO4", "PO3", "FT9", "Status"])
data_length = 60500 # Minimum  data length
BASE_PATH = r"C:\Users\timmy\Downloads\park-eeg"

# Returns the dataset and labels

def load_dataset():
    # Format is sub-NUM_task-Rest_eeg.edf
    data = []
    for subject in range(subject_num):
        subject_list = []
        for run in range(run_num):
            file_name = r"\sub-{}_task-Rest_eeg.edf".format(str(subject+1).zfill(3))
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

        query = "sub-"+str(index+1).zfill(3)
        results = raw_labels[(raw_labels["participant_id"] == query)]
        labels.append(results.iloc[0]["GROUP"])

        print("Label", index, "done!")

    return np.array(new_data), np.array([i == "PD" for i in labels])

data, labels = load_dataset()
sec = 12 # Window seconds?
Fs = 500 # Sampling freq in hz
i = 0 # Fake data for iteration
train_x, test_x, y_train, y_test = preprocess_data(data[:, 0], labels, 0, Fs,
                                                   filt=False, ICA=True, A_Matrix='cov', sec=sec)


def Adj_matrix(train_x, test_x):
    #Change weighted matrix to binary matrix with threshold
    percentile = 0.75
    adj_train = A_binarize(A_matrix=train_x,percent=percentile,sparse=False)
    adj_test  = A_binarize(A_matrix=test_x,percent=percentile,sparse=False)
    #sparse matrix

    print("sparsity: ",scipy.sparse.issparse(adj_train[9])) #check sparsity
    print("rank: ",np.linalg.matrix_rank(adj_train[9])) #check matrix rank
    return adj_train, adj_test

adj_train, adj_test = Adj_matrix(train_x, test_x)  # Creating brain graph

import pickle

output = open('train_x.pkl', 'wb')
pickle.dump(train_x, output)

output = open('test_x.pkl', 'wb')
pickle.dump(test_x, output)

output = open('y_train.pkl', 'wb')
pickle.dump(y_train, output)

output = open('y_test.pkl', 'wb')
pickle.dump(y_train, output)

print(adj_train, adj_test)