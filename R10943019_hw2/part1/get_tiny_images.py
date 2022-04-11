from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2

def get_tiny_images(image_paths):
    #############################################################################
    # TODO:                                                                     #
    # To build a tiny image feature, simply resize the original image to a very #
    # small square resolution, e.g. 16x16. You can either resize the images to  #
    # square while ignoring their aspect ratio or you can crop the center       #
    # square portion out of each image. Making the tiny images zero mean and    #
    # unit length (normalizing them) will increase performance modestly.        #
    #############################################################################
    '''
    Input : 
        image_paths: a list(N) of string where each string is an image 
        path on the filesystem.
    Output :
        tiny image features : (N, d) matrix of resized and then vectorized tiny
        images. E.g. if the images are resized to 16x16, d would equal 256.
    '''
    tiny_images = []
    resize_size = 16
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        #normalization
        image = (image - np.mean(image))/np.std(image)
        tiny_images.append(np.reshape(cv2.resize(image, (resize_size, resize_size), interpolation=cv2.INTER_AREA), (resize_size*resize_size, 1)))
    
    #tiny_images = np.squeeze(np.array(tiny_images))
    
    ##############################################################################
    #                                END OF YOUR CODE                            #
    ##############################################################################
    
    return tiny_images
