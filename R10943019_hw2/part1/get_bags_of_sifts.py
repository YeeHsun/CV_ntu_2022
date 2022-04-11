from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from time import time
import cv2

def get_bags_of_sifts(image_paths):
    ############################################################################
    # TODO:                                                                    #
    # This function assumes that 'vocab.pkl' exists and contains an N x 128    #
    # matrix 'vocab' where each row is a kmeans centroid or visual word. This  #
    # matrix is saved to disk rather than passed in a parameter to avoid       #
    # recomputing the vocabulary every time at significant expense.            #
    #                                                                          #                                                               
    # image_feats is an N x d matrix, where d is the dimensionality of the     #
    # feature representation. In this case, d will equal the number of clusters#
    # or equivalently the number of entries in each image's histogram.         #
    #                                                                          #
    # You will construct SIFT features here in the same way you did in         #
    # build_vocabulary (except for possibly changing the sampling rate)        #
    # and then assign each local feature to its nearest cluster center         #
    # and build a histogram indicating how many times each cluster was used.   #
    # Don't forget to normalize the histogram, or else a larger image with more#
    # SIFT features will look very different from a smaller version of the same#
    # image.                                                                   #
    ############################################################################
    '''
    Input : 
        image_paths : a list(N) of training images
    Output : 
        image_feats : (N, d) feature, each row represent a feature of an image
    '''
    
    image_feats = []
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    print("computing histogram....")
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        #make to image to float32 for the input format of kmean
        image = np.asarray(image,dtype='float32')           
        #normalization
        image = (image - np.mean(image))/np.std(image)
        keypoints, descriptors = dsift(image, step=[1,1], fast=True)

        #find the nearest vocabulary(cluster)
        distance_map = distance.cdist(vocab, descriptors, metric='euclidean')
        nearest_vocab_index = np.argmin(distance_map, axis=0)

        #build the histogram'''
        '''
        hist : 
            The values of the histogram.
        bin_edges : 
            Return the bin edges (length(hist)+1)
        '''
        hist, bin_edges = np.histogram(nearest_vocab_index, bins=len(vocab))
        #normalization to turn it into list
        hist = ((hist - np.mean(hist))/np.std(hist)).tolist()
        image_feats.append(hist)
    
    image_feats = np.array(image_feats)



    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return image_feats
