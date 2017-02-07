import cv2
import sys
import model as ml
import flower as fl
from time import time
from os.path import isfile
import numpy as np

def train_model(images, nlabels, bow_filename="bow"):
    k_size=20
    # Create geometric vocabulary of the images and then, we do K-Means
    # clustering to create the Bag Of Words and get the
    # labels and histograms of every class
    if not isfile(bow_filename+".npy"):
        x = time()
        BOW, keypoints, descriptors = fl.create_bag_of_words(images, sys.argv[1].upper(), k_size=k_size)
        y = time()
        print("Create BOW: ", y - x, ".s")
        w = time()
        BOW_descriptors = fl.compute_BOW_response(BOW, images, sys.argv[1].upper(),
                                                  keypoints, descriptors, k_size)
        np.save(file=bow_filename, arr=BOW_descriptors)
        z = time()
        print("Create BOW: ", z - w, ".s")
    else:
        BOW_descriptors = np.load(bow_filename+".npy")
    data = BOW_descriptors

    # Declare the index for the training and test subset
    training, test = ml.generate_train_test_masks(len(images))

    errors_svm = ml.svm(data=data, nlabels=nlabels, training=training, test=test)
    errors_rf = ml.rf(data=data, nlabels=nlabels, training=training, test=test)
    ml.cv_rf(data, nlabels, 30)
    ml.cv_svm(data, nlabels, 30)

    ml.paint_roc_curve(data=data, labels=nlabels, model=errors_rf[0], training=training, test=test, svm=False)

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
                         flags=cv2.IMREAD_COLOR) for i in range(1, 1361)]

    # create numeric labels for each class
    nlabels = ml.generate_num_labels()

    # train with images without any color modification
    train_model(images=images, nlabels=nlabels)
    # train with color quantization
    if not isfile("ColorQuantization/image_0001.jpg"):
        qimages = fl.convert_to_HSV_and_quantize(images=images)
    else:
        qimages = [cv2.imread('ColorQuantization/image_' + '%0*d' % (4, i) + '.jpg',
                             flags=cv2.IMREAD_COLOR) for i in range(1, 1361)]

    train_model(images=qimages, nlabels=nlabels, bow_filename="bow_hsv")
