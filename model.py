import numpy as np
from sklearn.svm import SVC

"""
function to divide the data into test and training
"""
def create_test_subset(size, num_photos_class, objs_class):
    subset = []
    for i in range(0, size, num_photos_class):
        subset += np.random.randint(low=i, high=i+80, size=objs_class).tolist()

    return np.array(subset)


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


"""
generate numeric labels
"""
def generate_num_labels(num_classes=17, num_photos_class=80):

    total = num_classes*num_photos_class
    numeric_labels = np.zeros(total, np.int32)

    for i in range(total):
        numeric_labels[i] = i // num_photos_class
    return numeric_labels

"""
train a model
"""
def fit_and_error(model, data, labels, mask):
    model.fit(X=data[mask], y=labels[mask])
    fit_labels = model.predict(data[mask])
    return model.score(X=fit_labels, y=labels[mask])

"""
train and test a svm model
"""
def svm(BOW_descriptors, nlabels, training, test):
    # Declare the svm models
    svm_onevsall = SVC(cache_size=200, verbose=True, decision_function_shape='ovr')
    svm_onevsone = SVC(cache_size=200, verbose=True, decision_function_shape='ovc')
    # Fit and get the error of the models
    error_onevsall = fit_and_error(model=svm_onevsall, data=BOW_descriptors, labels=nlabels, mask=training)
    error_onevsone = fit_and_error(model=svm_onevsone, data=BOW_descriptors, labels=nlabels, mask=training)
    print("Error en training:\n\tOne VS All: \t", error_onevsall, "\n\tOne VS One: \t", error_onevsone)
    # Fit and get the error of the models
    error_onevsall_test = fit_and_error(model=svm_onevsall, data=BOW_descriptors, labels=nlabels, mask=test)
    error_onevsone_test = fit_and_error(model=svm_onevsone, data=BOW_descriptors, labels=nlabels, mask=test)
    print("Error en training:\n\tOne VS All: \t", error_onevsall_test, "\n\tOne VS One: \t", error_onevsone_test)
    return error_onevsall, error_onevsone, error_onevsall_test, error_onevsone_test
