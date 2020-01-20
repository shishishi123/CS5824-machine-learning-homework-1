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


def naive_bayes_train(train_data, train_labels, params):
    """Train naive Bayes parameters from data.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. Must include an 'alpha' value
    :type params: dict
    :return: model learned with the priors and conditional probabilities of each feature
    :rtype: model
    """
    alpha = params['alpha']
    labels,result = np.unique(train_labels,return_counts=True)
    d,n=train_data.shape
    num_classes = labels.size
    # TODO: INSERT YOUR CODE HERE TO LEARN THE PARAMETERS FOR NAIVE BAYES

    #model is the dict to save probability distribution matrix
    model={}

    #num is the matrix to save class conditional probability
    num=np.zeros((num_classes,d,2))

    for i in range(num_classes):
        #get the indice where labels==i
        indice=np.where(train_labels==i)

        #get the number of features that ==true
        number=np.count_nonzero(train_data[:,indice[0]],axis=1)

        #get the class conditional probability distribution matrix
        num[i,:,1]=np.log((alpha+number)/(result[i]+2*alpha))
        num[i,:,0]=np.log((alpha+result[i]-number)/(result[i]+2*alpha))

    #save the probability matrix in model
    model['feature']=num

    #get the probability distribution of all classes
    model['class_distribution']=np.log(result/n)


    return model

def naive_bayes_predict(data, model):
    """Use trained naive Bayes parameters to predict the class with highest conditional likelihood.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param model: learned naive Bayes model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    # TODO: INSERT YOUR CODE HERE FOR USING THE LEARNED NAIVE BAYES PARAMETERS
    # TO CLASSIFY THE DATA

    #multiply data matrix and distribution matrix to get the log result
    #result on data matrix where feature==true
    result=model['feature'][:,:,1].dot(data)

    #reverse the value of data matrix: 0 to 1 and 1 to 0
    data=np.where(data==0,1,0)

    # result on data matrix where feature==false
    #add the result together
    result=result+(model['feature'][:,:,0].dot(data))

    #add the class distribution
    result=np.transpose(result)
    result=result+model['class_distribution']

    #use the maximum value class as the prediction result
    prediction = np.argmax(result,axis=1)

    return prediction
