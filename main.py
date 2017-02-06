import cv2
import sys
import model as ml
import flower as fl
from time import time
import numpy as np
from sklearn.svm import SVC


if __name__ == '__main__':
    # Check the parameters
    if len(sys.argv) < 2:
        sys.exit('Usage: %s <descriptor type>' % sys.argv[0])

    # Labels of the differents class
    labels = [
        'Daffodil', 'Snowdrop', 'LillyValley', 'Bluebell',
        'Crocus', 'Iris', 'Tigerlily', 'Tulip', 'Fritillary',
        'Sunflower', 'Daisy', 'Colts_Foot', 'Dandelion',
        'Cowslip', 'Buttercup', 'Windflower', 'Pansy',
    ]

    # Load all the images
    images = [cv2.imread('Dataset/image_' + '%0*d' % (4, i) + '.jpg',
                         flags=cv2.IMREAD_COLOR) for i in range(1, 1360)]

    # create numeric labels for each class
    nlabels = ml.generate_num_labels()

    # Create geometric vocabulary of the images and then, we do K-Means
    # clustering to create the Bag Of Words and get the
    # labels and histograms of every class
    x=time()
    BOW, keypoints, descriptors = fl.create_bag_of_words(images, sys.argv[1].upper(), k_size=50)
    y=time()
    print("Create BOW: ", y-x, ".s")
    w=time()
    BOW_descriptors = fl.compute_BOW_response(BOW, images, sys.argv[1].upper(),
                                              keypoints, descriptors)
    z=time()
    print("Create BOW: ", w - z, ".s")

    # Declare the index for the training and test subset
    training, test = ml.generate_train_test_masks(len(images))

    # Declare the svm model
    svm_onevsall = SVC(cache_size=200, verbose=True, decision_function_shape='ovr')
    svm_onevsone = SVC(cache_size=200, verbose=True, decision_function_shape='ovc')

    # svm.fit(BOW_descriptors, )