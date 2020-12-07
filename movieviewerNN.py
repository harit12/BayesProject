import numpy as np
import pandas as pd
from movieviewerModel import Vocab, DataSet, Corpus
import datetime
import random

labelSet = ['pos', 'neg']

def init_features(dataDir):
    """
    args: dataDir(String of data directory path)
    """
    data = DataSet(dataDir)
    corpus = Corpus(data.data_prep)
    return corpus, data
def labelToClass(label):
    """
    args: label(String of a label)
    Turns label into a number based on label
    """
    res = 0
    if label=='pos':
        res = 1
    return res
def formatData(labeled_data, cv):
    """
    Format the data to be a tuple containing the bag of words vector and its class
    """
    res = [(add_bias((cv.transform(data)).toarray()), labelToClass(label)) for data, label in labeled_data]
    return res

def init_weights(inputs_nums):
    """
    Initialize weights of NN. Currently only one hidden layer
    """
    weightsXZ = np.random.random((5, inputs_nums))
    weightsZY = np.random.random((2, 5))
    bias = np.random.random((2))
    return weightsXZ, weightsZY, bias

def add_bias(matrix):
    """
    Adds bias to vector/matrix
    """
    matrix = np.append(matrix, 1)
    return matrix

def relu(val):
    """
    ReLu activation func
    """
    res = 0
    if val>=0:
        res=val
    else:
        res = 0
    return res
#Vectorize ReLu func
vrelu = np.vectorize(relu)

def SoftMax(vals,weights, inputs, bias,labelSet):
    """
    SoftMax function
    """
    numer = np.exp(vals)
    denom = 0
    for label in range(len(labelSet)):
        denom+=np.exp(weights[label].dot(inputs)+bias[label])
    res = numer/denom
    return res

def forward(inputs, weights):
    """
    Forward Propagation
    """
    weightsHidden, weightsOutputs, bias = weights
    unactivatedHidden = weightsHidden.dot(inputs)
    activatedHidden = vrelu(unactivatedHidden)
    unactivatedOutput = weightsOutputs.dot(activatedHidden)+bias
    probs = []
    for row in unactivatedOutput:
        probs.append(SoftMax(row, weightsOutputs, activatedHidden, bias, labelSet))
    res = np.asarray(probs)
    return res
def loss(outputs, hypo):
    """
    Cross Entropy Loss
    """
    res = 0
    for output in outputs:
        


def main():
    corpus,data = init_features('dev/train')
    inputs = formatData(data.data_prep_labeled, corpus.cv)
    #Shuffle training data
    random.shuffle(inputs)
    print(datetime.datetime.now(), 'format data')
    print(inputs[0][0])
    weights = init_weights(inputs[0][0].shape[0])
    forward(inputs[0][0], weights)
    
if __name__ == '__main__':
    main()
