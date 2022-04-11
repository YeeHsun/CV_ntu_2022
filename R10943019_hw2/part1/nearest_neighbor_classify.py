from __future__ import print_function

import numpy as np
import scipy.spatial.distance as distance
from scipy.stats import mode

#return the nearst k neighbors
def minN(elements):
    s = sorted(range(len(elements)), key=lambda k: elements[k])
    return s

#return the most frequnet element in a list
def most_frequent(List):
    return max(set(List), key = List.count)

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    ###########################################################################
    # TODO:                                                                   #
    # This function will predict the category for every test image by finding #
    # the training image with most similar features. Instead of 1 nearest     #
    # neighbor, you can vote based on k nearest neighbors which will increase #
    # performance (although you need to pick a reasonable value for k).       #
    ###########################################################################
    ###########################################################################
    # NOTE: Some useful functions                                             #
    # distance.cdist :                                                        #
    #   This function will calculate the distance between two list of features#
    #       e.g. distance.cdist(? ?)                                          #
    ###########################################################################
    '''
    Input : 
        train_image_feats : 
            image_feats is an (N, d) matrix, where d is the 
            dimensionality of the feature representation.

        train_labels : 
            image_feats is a list of string, each string
            indicate the ground truth category for each training image. 

        test_image_feats : 
            image_feats is an (M, d) matrix, where d is the 
            dimensionality of the feature representation.
    Output :
        test_predicts : 
            a list(M) of string, each string indicate the predict
            category for each testing image.
    '''

    k = 5
    test_predicts = []
    #reshape the matrix to the input format of distance.cdist
    train_image_feats = np.array(train_image_feats)
    train_image_feats = np.reshape(train_image_feats, (train_image_feats.shape[0], 1, train_image_feats.shape[1]))
    test_image_feats = np.array(test_image_feats)
    test_image_feats = np.reshape(test_image_feats, (test_image_feats.shape[0], 1, test_image_feats.shape[1]))
    print("knn calculating.....")
    for i in range(len(test_image_feats)):
        distance_map = []
        for j in range(len(train_image_feats)):
            #print(test_image_feat)
            distance_map.append(distance.cdist(test_image_feats[i], train_image_feats[j], 'braycurtis'))
        sorted_score = minN(distance_map)
        k_labels = [train_labels[m] for m in sorted_score[0:k]]
        k_labels = k_labels
        '''print(k_labels)
        print(most_frequent(k_labels))'''
        test_predicts.append(most_frequent(k_labels))
        #test_predicts.append(train_labels[distance_map.index(min(distance_map))])
    #print(test_image_feats[0])

    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return test_predicts
