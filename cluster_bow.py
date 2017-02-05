import cv2
import numpy as np

def kmeans_clusters(data, k_size, n_attempts = 10):
    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags
    flags = cv2.KMEANS_RANDOM_CENTERS

    # Apply KMeans
    print(k_size)
    compactness, labels, centers = cv2.kmeans(data=data, K=k_size,
                                              bestLabels=None, criteria=criteria,
                                              attempts=n_attempts, flags=flags)

    return compactness, labels, centers


def bag_of_words(centers):
    sift2 = cv2.xfeatures2d.SIFT_create()
    bowDiccion = cv2.BOWImgDescriptorExtractor(dextractor=sift2, dmatcher=cv2.BFMatcher(normType=cv2.NORM_L2))
    bowDiccion.setVocabulary(centers)
    # for each image do its bag of words

