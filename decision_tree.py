"""This module includes methods for training and predicting using decision trees."""
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


def calculate_information_gain(data, labels):
    """
    Computes the information gain on label probability for each feature in data

    :param data: d x n matrix of d features and n examples
    :type data: ndarray
    :param labels: n x 1 vector of class labels for n examples
    :type labels: array
    :return: d x 1 vector of information gain for each feature (H(y) - H(y|x_d))
    :rtype: array
    """
    all_labels = np.unique(labels)
    num_classes = len(all_labels)
    class_count = np.zeros(num_classes)
    d, n = data.shape
    full_entropy = 0
    for c in range(num_classes):
        class_count[c] = np.sum(labels == all_labels[c])
        if class_count[c] > 0:
            class_prob = class_count[c] / n
            full_entropy -= class_prob * np.log(class_prob)

    # print("Full entropy is %d\n" % full_entropy)

    gain = full_entropy * np.ones(d)

    # we use a matrix dot product to sum to make it more compatible with sparse matrices
    num_x = data.dot(np.ones(n))
    prob_x = num_x / n
    prob_not_x = 1 - prob_x

    for c in range(num_classes):
        # print("Computing contribution of class %d." % c)
        num_y = np.sum(labels == all_labels[c])
        # this next line sums across the rows of data, multiplied by the
        # indicator of whether each column's label is c. It counts the number
        # of times each feature is on among examples with label c.
        # We again use the dot product for sparse-matrix compatibility
        data_with_label = data[:, labels == all_labels[c]]
        num_y_and_x = data_with_label.dot(np.ones(data_with_label.shape[1]))

        # Prevents Python from outputting a divide-by-zero warning
        with np.errstate(invalid='ignore'):
            prob_y_given_x = num_y_and_x / (num_x + 1e-8)
        prob_y_given_x[num_x == 0] = 0

        nonzero_entries = prob_y_given_x > 0
        if np.any(nonzero_entries):
            with np.errstate(invalid='ignore', divide='ignore'):
                cond_entropy = - np.multiply(np.multiply(prob_x, prob_y_given_x), np.log(prob_y_given_x))
            gain[nonzero_entries] -= cond_entropy[nonzero_entries]

        # The next lines compute the probability of y being c given x = 0 by
        # subtracting the quantities we've already counted
        # num_y - num_y_and_x is the number of examples with label y that
        # don't have each feature, and n - num_x is the number of examples
        # that don't have each feature
        with np.errstate(invalid='ignore'):
            prob_y_given_not_x = (num_y - num_y_and_x) / ((n - num_x) + 1e-8)
        prob_y_given_not_x[n - num_x == 0] = 0

        nonzero_entries = prob_y_given_not_x > 0
        if np.any(nonzero_entries):
            with np.errstate(invalid='ignore', divide='ignore'):
                cond_entropy = - np.multiply(np.multiply(prob_not_x, prob_y_given_not_x), np.log(prob_y_given_not_x))
            gain[nonzero_entries] -= cond_entropy[nonzero_entries]

    return gain


def decision_tree_train(train_data, train_labels, params):
    """Train a decision tree to classify data using the entropy decision criterion.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. Must include a 'max_depth' value
    :type params: dict
    :return: dictionary encoding the learned decision tree
    :rtype: dict
    """
    max_depth = params['max_depth']
    labels = np.unique(train_labels)
    num_classes = labels.size
    model = recursive_tree_train(train_data, train_labels, depth=0, max_depth=max_depth, num_classes=num_classes)
    return model


def recursive_tree_train(data, labels, depth, max_depth, num_classes):
    """Helper function to recursively build a decision tree by splitting the data by a feature.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param labels: length n numpy array with integer labels
    :type labels: array_like
    :param depth: current depth of the decision tree node being constructed
    :type depth: int
    :param max_depth: maximum depth to expand the decision tree to
    :type max_depth: int
    :param num_classes: number of classes in the classification problem
    :type num_classes: int
    :return: dictionary encoding the learned decision tree node
    :rtype: dict
    """
    # TODO: INSERT YOUR CODE FOR LEARNING THE DECISION TREE STRUCTURE HERE

    node = {} #node is the dict to save the model

    #get the number list of classes from labels
    #classes: ndarray, list of the classes
    #classes_num: ndarray, list of how many examples a class have
    classes,classes_num=np.unique(labels,return_counts=True)

    if depth >= max_depth:
        # return the class that have the maximum number of examples as prediction
        node['prediction'] = classes[np.argmax(classes_num)]
        node['feature'] = -1 #The label of whether a node is the decision part or prediction part
        return node
    elif num_classes == 1:
        # return the class that have the maximum number of examples as prediction
        node['prediction'] = classes[np.argmax(classes_num)]
        node['feature'] = -1
        return node
    else:

        gain = calculate_information_gain(data, labels)
        #find the feature that have the maximum information gain
        node['feature'] = np.where(gain == max(gain))[0][0]
        m = node['feature']

        #divide the data and the labels to two different part
        data_left = np.delete(data[:,np.where(data[m, :] == True)[0]], m, 0)
        data_right = np.delete(data[:,np.where(data[m, :] == False)[0]], m, 0)
        label_left = labels[np.where(data[m, :] == True)]
        label_right = labels[np.where(data[m, :] == False)]
        num_classes_left = len(set(label_left))
        num_classes_right = len(set(label_right))

        #call the function recursively
        if label_left.size!=0:
            node['left'] = recursive_tree_train(data_left, label_left, depth + 1, max_depth, num_classes_left)
        else:
            node['left'] = {'prediction': classes[np.argmax(num_classes)], 'feature': -1}

        if label_right.size!=0:
            node['right'] = recursive_tree_train(data_right, label_right, depth + 1, max_depth, num_classes_right)
        else:
            node['right'] = {'prediction': classes[np.argmax(num_classes)], 'feature': -1}


    return node


def decision_tree_predict(data, model):
    """Predict most likely label given computed decision tree in model.

    :param data: d x n ndarray of d binary features for n examples.
    :type data: ndarray
    :param model: learned decision tree model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    # TODO: INSERT YOUR CODE FOR COMPUTING THE DECISION TREE PREDICTIONS HERE

    d,n=data.shape

    #variable labels is the ndarray to get the predict result
    labels=np.zeros(n)

    for i in range(n):
        tree=model
        onedata=data[:,i] #getting one piece of testing data
        while(1):
            if tree['feature']==-1:
                labels[i]=tree['prediction'] #get the prediction at the leaf node
                break;
            else:
                #judge which position the decision tree will go: the left tree or the right tree
                if onedata[tree['feature']]==True:
                    onedata=np.delete(onedata,tree['feature']) #one feature will be used for once
                    tree=tree['left']
                else:
                    onedata=np.delete(onedata,tree['feature'])
                    tree=tree['right']

    return labels


