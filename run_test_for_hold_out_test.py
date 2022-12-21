import pickle
import sys
import timeit
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools
import pandas as pd

#from sklearn.metrics import roc_auc_score, precision_score, recall_score


class GraphNeuralNetwork(nn.Module):
    def __init__(self):
        super(GraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(hidden_layer)])
        self.W_output = nn.ModuleList([nn.Linear(dim+2, dim+2)
                                       for _ in range(output_layer)])
        self.W_property = nn.Linear(dim+2, 2)

    def pad(self, matrices, pad_value):
        """Pad adjacency matrices for batch processing."""
        sizes = [d.shape[0] for d in matrices]
        D = sum(sizes)
        pad_matrices = pad_value + np.zeros((D, D))
        m = 0
        for i, d in enumerate(matrices):
            s_i = sizes[i]
            pad_matrices[m:m+s_i, m:m+s_i] = d
            m += s_i
        return torch.FloatTensor(pad_matrices).to(device)

    def sum_axis(self, xs, axis):
        y = list(map(lambda x: torch.sum(x, 0), torch.split(xs, axis)))
        return torch.stack(y)

    def mean_axis(self, xs, axis):
        y = list(map(lambda x: torch.mean(x, 0), torch.split(xs, axis)))
        return torch.stack(y)

    def gnn(self, xs, A, M, i):
        hs = torch.relu(self.W_fingerprint[i](xs))
        if update == 'sum':
            return xs + torch.matmul(A, hs)
        if update == 'mean':
            return xs + torch.matmul(A, hs) / (M-1)

    def forward(self, inputs):

        Smiles, fingerprints, adjacencies, docking_scores, properties = inputs
        axis = list(map(lambda x: len(x), fingerprints))

        M = np.concatenate([np.repeat(len(f), len(f)) for f in fingerprints])
        M = torch.unsqueeze(torch.FloatTensor(M), 1)

        fingerprints = torch.cat(fingerprints)
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        for i in range(hidden_layer):
            fingerprint_vectors = self.gnn(fingerprint_vectors,
                                           adjacencies, M, i)

        if output == 'sum':
            molecular_vectors = self.sum_axis(fingerprint_vectors, axis)
        if output == 'mean':
            molecular_vectors = self.mean_axis(fingerprint_vectors, axis)

        """getting docking scores and concatenate them with molecular vectors"""
        
        docking_scores = torch.from_numpy(np.asarray(docking_scores)).to(device)
            
        y_cat = torch.cat((docking_scores, molecular_vectors), 1)
        
        for j in range(output_layer):
            y_cat = torch.relu(self.W_output[j](y_cat))

#        print(y_cat)

        predicted_properties = self.W_property(y_cat)

        return Smiles, predicted_properties

    def __call__(self, data_batch):

        inputs = data_batch[:]
        Smiles, predicted_properties = self.forward(inputs)

        ys = F.softmax(predicted_properties, 1).to('cpu').data.numpy()
        predicted_labels = list(map(lambda x: np.argmax(x), ys))
        predicted_scores = list(map(lambda x: x[1], ys))
        return predicted_labels, predicted_scores


def load_tensor(filename, dtype, allow_pickle=True):
    return [dtype(d).to(device) for d in np.load(filename + '.npy', allow_pickle=True)]


def load_numpy(filename, allow_pickle=True):
    return np.load(filename + '.npy', allow_pickle=True)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":

    """Hyperparameters."""
    (DATASET, radius, update, output, dim, hidden_layer, output_layer, batch,
     lr, lr_decay, decay_interval, weight_decay, iteration,
     setting) = sys.argv[1:]
    (dim, hidden_layer, output_layer, batch, decay_interval,
     iteration) = map(int, [dim, hidden_layer, output_layer, batch,
                            decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    with open('train/radius' + str(radius) +'/'+ 'fingerprint_dict.pickle', 'rb') as f:
        fingerprint_dict = pickle.load(f)
    fingerprint_dict = load_pickle('train/radius'+ str(radius) + '/' + 'fingerprint_dict.pickle')
    n_fingerprint = len(fingerprint_dict)

    """load test data from the test directory"""

    dataset = load_numpy('test/radius' + str(radius)+ '/' + 'Test_data')
    Smiles = [line.rstrip('\n') for line in open('test/radius' + str(radius)+ '/' + 'test_smiles.txt')]
    print(dataset[0])

    """Set a model."""
    torch.manual_seed(1234)
    model = GraphNeuralNetwork()
    all_model_preds = []
    dataframe = pd.DataFrame()
#    dataframe['Smiles'] = Smiles   ####removed to take all predictions only
    for filename in os.listdir('fullmodel/'):
        if filename.endswith('.txt'):
            pass
        else:
            checkpoint = torch.load('fullmodel/'+ filename)
            model.load_state_dict(checkpoint)
            model = model.to(device)
    
            print('Testing model')
            start = timeit.default_timer()

            N = len(dataset)
            Correct_labels, Predicted_labels, Predicted_scores = [], [], []
            batch = 150

#        data_batch = list(zip(*dataset[:]))
        
            predicts = []
            for i in range(0, N, batch):
                data_batch = list(zip(*dataset[i:i+batch]))
                predicts.append(model(data_batch))
#        print(predicts)

            predictions = []
            softmax_scores = []
       
            all150s = []
            for n in range(0,len(predicts)):
                for m in range(0,2):
                    all150s.append(predicts[n][m])
#    print("150s=",all150s)
    
            labels = []
            scores = []
            for l in range(0,len(all150s)):
                if l%2 == 0:
                    labels.append(all150s[l])
                else:
                    scores.append(all150s[l])
#    print("labels=",labels)
#    print("scores=",scores)
    
    
            merged_labels = list(itertools.chain(*labels))
            merged_scores = list(itertools.chain(*scores))
    
#    print(merged_labels)
#    print(len(merged_labels))
#    print(merged_scores)
#    print(len(merged_scores))

            results = pd.DataFrame(
                    {'Smiles': Smiles,
                     'prediction_label': merged_labels,
                     'softmax_score': merged_scores,
                     })
            print(results)
            results.to_csv('bootstrapping_results_test/' + str(filename) +'EGNN_results.csv', header=True, index=False)
            all_model_preds.append(merged_scores)
            dataframe[str(filename)] = merged_scores
   
    
    
    
    dataframe.to_csv('bootstrapping_results_test/' + 'all_bootstrapping_predictions.csv', header=True, index=False)
    dataframe['average'] = dataframe.mean(axis=1)
    dataframe['std'] = dataframe.std(axis=1)
    dataframe['Smiles'] = Smiles


    final=dataframe[['Smiles','average','std']]
    final.to_csv('bootstrapping_results_test/' + 'EGNN_Bootstrapping_average_results.csv', header=True, index=False)

        
 
