import cv2
import numpy as np

from flower import *


"""
function to divide the data into test and training
"""
def create_test_subset(size, num_photos_class, objs_class):
    subset = []
    for i in range(0, size, num_photos_class):
        subset += np.random.randint(low=i, high=i+80, size=objs_class).tolist()

    return np.array(subset)


"""
interface to OpenCV machine learning train function
"""
def train_model(model, label, data, mask, lay=cv2.ml.ROW_SAMPLE):
    model.train(samples=data[mask], layout=lay,
                responses=label[mask])


"""
interface to OpenCV machine learning predict function
"""
def predict_model(model, label, data, mask):
    return model.predict(samples=data[mask], results = label[mask])


"""
function to compute the square error of the results
"""
def error(labels, results):
    return ((labels - results)**2).mean(axis=None)


# Declare a parameterized SVM model
def create_SVM(gamma=0.1, C=3.01, Nu=0.68,
               Kernel=cv2.ml.SVM_RBF, type=cv2.ml.SVM_NU_SVC):
    svm_model = cv2.ml.SVM_create()
    svm_model.setGamma(gamma)
    svm_model.setC(C)
    svm_model.setNu(Nu)
    svm_model.setKernel(Kernel)
    svm_model.setType(type)

    return svm_model


"""
generate numeric labels
"""
def generate_num_labels(num_classes=17, num_photos_class=80):

    total = num_classes*num_photos_class
    numeric_labels = np.zeros(total, np.int32)

    for i in range(total):
        numeric_labels[i] = i // num_photos_class

    return numeric_labels


def generate_train_test_masks(size,
                              num_photos_class = 80,
                              objs_class = 4):
    # Test index
    test_subset = create_test_subset(size, num_photos_class, objs_class)

    aux = np.arange(size)
    # Training index
    training_subset = np.in1d(aux, test_subset) * 1
    training_subset = np.where(training_subset == 0)[0]

    # Return both of them
    return training_subset, test_subset


# Declare Random Forest Model
rt_model = cv2.ml.RTrees_create()

