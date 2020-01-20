"""This module includes utilities to run cross-validation on general supervised learning methods."""
from __future__ import division
import numpy as np

'''
-------------------------------------------------------------------------
This is the programming part homework of CS5824 Advanced Machine Learning.
This homework is finished by Shanghao Shi alone.
This .py file implement the training method and predicting method for decision tree algorithm.
The file has sucessfully passed the test code.
If this is any problem, please contact: shanghaos@vt.edu
-------------------------------------------------------------------------
'''


def cross_validate(trainer, predictor, all_data, all_labels, folds, params):
    """Perform cross validation with random splits.

    :param trainer: function that trains a model from data with the template
             model = function(all_data, all_labels, params)
    :type trainer: function
    :param predictor: function that predicts a label from a single data point
                label = function(data, model)
    :type predictor: function
    :param all_data: d x n data matrix
    :type all_data: numpy ndarray
    :param all_labels: n x 1 label vector
    :type all_labels: numpy array
    :param folds: number of folds to run of validation
    :type folds: int
    :param params: auxiliary variables for training algorithm (e.g., regularization parameters)
    :type params: dict
    :return: tuple containing the average score and the learned models from each fold
    :rtype: tuple
    """
    scores = np.zeros(folds)
    d, n = all_data.shape
    indices = np.array(range(n), dtype=int)
    # pad indices to make it divide evenly by folds
    examples_per_fold = int(np.ceil(n / folds))
    ideal_length = int(examples_per_fold * folds)
    # use -1 as an indicator of an invalid index
    indices = np.append(indices, -np.ones(ideal_length - indices.size, dtype=int))
    assert indices.size == ideal_length
    indices = indices.reshape((examples_per_fold, folds))
    models = []
    # TODO: INSERT YOUR CODE FOR CROSS VALIDATION HERE


    for i in range(folds):
        #get the train data indices and test data indices
        train_indices = np.delete(indices, i, 1).flatten();
        test_indices = indices[:, i]

        #delete element==-1 in train indices and test indices
        train_indices = np.delete(train_indices, np.where(train_indices == -1))
        test_indices = np.delete(test_indices, np.where(test_indices == -1))

        #get train data, train label and test data, test label
        train_data = all_data[:,train_indices]
        train_label = all_labels[train_indices]
        test_data = all_data[:,test_indices]
        test_label = all_labels[test_indices]

        #use train data to train a model
        model=trainer(train_data,train_label,params)

        #add the model in the model list
        models.append(model)

        #use test data to get the prediction labels
        predict_labels=predictor(test_data,model)

        #get score of the ith fold
        scores[i]=np.mean(predict_labels==test_label)

    #get the average score of all folds
    score = np.mean(scores)

    return score, models
