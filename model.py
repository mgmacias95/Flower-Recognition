import cv2
import numpy as np

from flower import *

# train a model
def train_model(model, label, data, mask=training_mask,
                lay=cv2.ml.ROW_SAMPLE):
    model.train(samples=data[mask], layout=lay,
                responses=label[mask])


def predict_model(model, label, data, mask):
    return model.predict(samples=data[mask], results = label[mask])

def error(labels, results):
    return ((labels - results)**2).mean(axis=None)

# Declare SVM model
svm_model = cv2.ml.SVM_create()
svm_model.setGamma(0.1)
svm_model.setC(3.01)
svm_model.setNu(0.68)
svm_model.setKernel(cv2.ml.SVM_RBF)
svm_model.setType(cv2.ml.SVM_NU_SVC)

# train the svm model
train_model(model=svm_model, label=df_labels, data=df_data_array)
results = predict_model(svm_model, df_labels, df_data_array, mask=training_mask)
train_error = error (df_labels[training_mask], results[1])
print("Error en train = ", train_error)
# test the svm model
test_results = predict_model(svm_model, df_labels, df_data_array, mask=test_mask)
test_error = error (df_labels[test_mask], test_results[1])
print("Error en test = ", test_error)

# Declare Random Forest Model
rt_model = cv2.ml.RTrees_create()

# train the rt model
train_model(model=rt_model, label=df_labels, data=df_data_array)
results_rt = predict_model(model=rt_model, label=df_labels, data=df_data_array, mask=training_mask)
train_error_rt = error(df_labels[training_mask], results_rt[1])
print("Error en train de Random Forest = ", train_error_rt)

test_results_rt = predict_model(model=rt_model, label=df_labels, data=df_data_array, mask=test_mask)
test_error_rt = error(df_labels[test_mask], test_results_rt[1])
print("Error en test de Random Forest = ", test_error_rt)

# ---------------------- Applying k-means to create clusters -------------------------------------- #
# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS

# Apply KMeans
compactness, labels, centers = cv2.kmeans(data=df_labels, K=17, bestLabels=None, criteria=criteria,
                                          attempts=10, flags=flags)

