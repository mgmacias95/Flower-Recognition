import cv2
import sys
import model as ml
import flower as fl
from time import time
from os.path import isfile
import numpy as np

def train_model(images, nlabels, bow_filename="bow", roc_filename="roc", k_size=20):
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

    # errors_svm = ml.svm(data=data, nlabels=nlabels, training=training, test=test)
    # errors_rf = ml.rf(data=data, nlabels=nlabels, training=training, test=test)
    rf = ml.cv_rf(data, nlabels, 30)
    svm = ml.cv_svm(data, nlabels, 30)

    ml.paint_roc_curve(data=data, labels=nlabels, training=training, test=test,
                       model_list=[svm[0], svm[1], rf[0], rf[1]],
                       filename=roc_filename, svm_list=[True, True, False, False],
                       label_list=["SVM One VS All", "SVM One VS One", "Boosting", "RF"])


def train_both_models(nlabels, roc_filename, geom_name, hsv_name):
    data = np.load(geom_name+".npy")
    qdata = np.load(hsv_name+".npy")
    both = np.concatenate((data, qdata), axis=1)

    training, test = ml.generate_train_test_masks(size=len(images))

    rf = ml.cv_rf(data, nlabels, 30)
    svm = ml.cv_svm(data, nlabels, 30)

    ml.paint_roc_curve(data=data, labels=nlabels, training=training, test=test,
                       model_list=[svm[0], svm[1], rf[0], rf[1]],
                       filename=roc_filename, svm_list=[True, True, False, False],
                       label_list=["SVM One VS All", "SVM One VS One", "Boosting", "RF"])

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

    if not isfile("ColorQuantization/image_0001.jpg"):
        qimages = fl.convert_to_HSV_and_quantize(images=images)
    else:
        qimages = [cv2.imread('ColorQuantization/image_' + '%0*d' % (4, i) + '.jpg',
                             flags=cv2.IMREAD_COLOR) for i in range(1, 1361)]
    # create numeric labels for each class
    nlabels = ml.generate_num_labels()

    ks = [200]

    for k in ks:
        print("\n\nK = " + str(ks))
        bfilename = "numpydata/bow_"+sys.argv[1].lower()+"k"+str(k)
        bhfilename = "numpydata/bow_hsv_"+sys.argv[1].lower()+"k"+str(k)
        rfilename = "doc/img/shape"+sys.argv[1].lower()+"_"+str(k)
        rhfilename = "doc/img/color" + sys.argv[1].lower() + "_" + str(k)
        rbfilename = "doc/img/both" + sys.argv[1].lower() + "_" + str(k)
        # train with images without any color modification
        train_model(images=images, nlabels=nlabels, roc_filename=rfilename, bow_filename=bfilename, k_size=k)
        # train with color quantization
        train_model(images=qimages, nlabels=nlabels, bow_filename=bhfilename, roc_filename=rhfilename, k_size=k)
        # train with both
        train_both_models(nlabels, rbfilename, bfilename, bhfilename)
