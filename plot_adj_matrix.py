import pickle

import scipy
import numpy as np
from graph_clustering import A_binarize


def Adj_matrix(train_x, test_x):
    #Change weighted matrix to binary matrix with threshold
    percentile = 0.75
    adj_train = A_binarize(A_matrix=train_x,percent=percentile,sparse=False)
    adj_test  = A_binarize(A_matrix=test_x,percent=percentile,sparse=False)
    #sparse matrix

    print("sparsity: ",scipy.sparse.issparse(adj_train[9])) #check sparsity
    print("rank: ",np.linalg.matrix_rank(adj_train[9])) #check matrix rank
    return adj_train, adj_test

f = open('train_x.pkl', 'rb')
train_x = pickle.load(f)
f1 = open('test_x.pkl', 'rb')
test_x = pickle.load(f1)
adj_train, adj_test = Adj_matrix(train_x, test_x)  # Creating brain graph

matrix = adj_train[15]
import matplotlib.pyplot as plt

# Example boolean matrix
print(matrix)
# Plotting the boolean matrix
plt.imshow(matrix, cmap='binary', interpolation='nearest')
plt.title('Binary Functional Connectivity Matrix')
plt.colorbar()
plt.show()