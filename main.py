import cv2
import sys
import model as ml
import flower as fl
import numpy as np


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
                         flags=cv2.IMREAD_COLOR) for i in range(1, 50)]

    # create numeric labels for each class
    nlabels = ml.generate_num_labels()

    # Create geometric vocabulary of the images and then, we do K-Means
    # clustering to create the Bag Of Words and get the
    # labels and histograms of every class
    BOW, keypoints, descriptors = fl.create_bag_of_words(images, sys.argv[1].upper())

    BOW_descriptors = fl.compute_BOW_response(BOW, images, sys.argv[1].upper(),
                                              keypoints, descriptors)
    # # Declare the svm model
    # svm = ml.create_SVM()
    #
    # # Declare the index for the training and test subset
    # training, test = ml.generate_train_test_masks(len(images))

    # # Train
    # ml.train_model(model=svm, label=nlabels, data=BOW, mask=training)
    # results = ml.predict_model(svm, nlabels, BOW, mask=training)
    # print(results)
    # train_error = ml.error(nlabels[training], results[1])
    # print("Error en train = ", train_error)
    # # test the svm model
    # test_results = ml.predict_model(svm, nlabels, BOW, mask=test)
    # test_error = ml.error(BOW[test], test_results[1])
    # print("Error en test = ", test_error)
