import cv2
import numpy as np

def kmeans_clusters(df_labels):
    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags
    flags = cv2.KMEANS_RANDOM_CENTERS

    # Apply KMeans
    compactness, labels, centers = cv2.kmeans(data=df_labels, K=17, bestLabels=None, criteria=criteria,
                                              attempts=10, flags=flags)

    return compactness, labels, centers


def bag_of_words(centers):
    sift2 = cv2.xfeatures2d.SIFT_create()
    bowDiccion = cv2.BOWImgDescriptorExtractor(dextractor=sift2, dmatcher=cv2.BFMatcher(normType=cv2.NORM_L2))
    bowDiccion.setVocabulary(centers)
    # for each image do its bag of words

