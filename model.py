import cv2
import numpy as np

from flower import *


"""
function to divide the data into test and training
"""
def create_train_subset():
    subset = []
    for i in range(0, 1360, 80):
        subset += np.random.randint(low=i, high=i+80, size=4).tolist()

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

# Declare SVM model
def create_SVM(gamma=0.1, C=3.01, Nu=0.68,
               Kernel=cv2.ml.SVM_RBF, type=cv2.ml.SVM_NU_SVC):
    svm_model = cv2.ml.SVM_create()
    svm_model.setGamma(gamma)
    svm_model.setC(C)
    svm_model.setNu(Nu)
    svm_model.setKernel(Kernel)
    svm_model.setType(type)

    return svm_model

# # train the svm model
# train_model(model=svm_model, label=df_labels, data=df_data_array)
# results = predict_model(svm_model, df_labels, df_data_array, mask=training_mask)
# train_error = error (df_labels[training_mask], results[1])
# print("Error en train = ", train_error)
# # test the svm model
# test_results = predict_model(svm_model, df_labels, df_data_array, mask=test_mask)
# test_error = error (df_labels[test_mask], test_results[1])
# print("Error en test = ", test_error)

# Declare Random Forest Model
rt_model = cv2.ml.RTrees_create()

# # train the rt model
# train_model(model=rt_model, label=df_labels, data=df_data_array)
# results_rt = predict_model(model=rt_model, label=df_labels, data=df_data_array, mask=training_mask)
# train_error_rt = error(df_labels[training_mask], results_rt[1])
# print("Error en train de Random Forest = ", train_error_rt)
#
# test_results_rt = predict_model(model=rt_model, label=df_labels, data=df_data_array, mask=test_mask)
# test_error_rt = error(df_labels[test_mask], test_results_rt[1])
# print("Error en test de Random Forest = ", test_error_rt)
#

