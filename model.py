import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
# paint ROC curve
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle

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
    # fit_labels = model.predict(data[mask])
    return model.score(X=data[mask], y=labels[mask])

"""
train and test a svm model
"""
def svm(data, nlabels, training, test, verbose=False):
    # Declare the svm models
    svm_onevsall = SVC(cache_size=200, verbose=verbose, decision_function_shape='ovr')
    svm_onevsone = SVC(cache_size=200, verbose=verbose, decision_function_shape='ovc')
    # Fit and get the error of the models
    error_onevsall = fit_and_error(model=svm_onevsall, data=data, labels=nlabels, mask=training)
    error_onevsone = fit_and_error(model=svm_onevsone, data=data, labels=nlabels, mask=training)
    print("Error en training:\n\tOne VS All: \t", error_onevsall, "\n\tOne VS One: \t", error_onevsone)
    # Fit and get the error of the models
    error_onevsall_test = fit_and_error(model=svm_onevsall, data=data, labels=nlabels, mask=test)
    error_onevsone_test = fit_and_error(model=svm_onevsone, data=data, labels=nlabels, mask=test)
    print("Error en test:\n\tOne VS All: \t", error_onevsall_test, "\n\tOne VS One: \t", error_onevsone_test)
    return svm_onevsall, svm_onevsone, error_onevsall, error_onevsone, error_onevsall_test, error_onevsone_test

"""
train and test a random forest model
"""
def rf(data, nlabels, training, test):
    # declare the rf model
    rfb = RandomForestClassifier(n_jobs=-1)
    rfn = RandomForestClassifier(n_jobs=-1, bootstrap=False)
    # fit both models and get its error
    error_boots = fit_and_error(model=rfb, data=data, labels=nlabels, mask=training)
    error_noboots = fit_and_error(model=rfn, data=data, labels=nlabels, mask=training)
    # print("Error en training:\n\tWith Bootstrap:\t",error_boots,"\n\tWithout Bootstrap:\t",error_noboots)
    # fit both models and get its test error
    error_boots_test = fit_and_error(model=rfb, data=data, labels=nlabels, mask=test)
    error_noboots_test = fit_and_error(model=rfn, data=data, labels=nlabels, mask=test)
    # print("Error en test:\n\tWith Bootstrap:\t", error_boots_test, "\n\tWithout Bootstrap:\t", error_noboots_test)
    return rfb, rfn, error_boots, error_noboots, error_boots_test, error_noboots_test

"""
paint a ROC curve
"""
def paint_roc_curve(data, labels, model_list, training, test, filename, svm_list, label_list, n_classes=17):
    fpr = []
    tpr = []
    roc_auc = []

    for model,svm,l in zip(model_list, svm_list,label_list):
        print(l)
        f, t, r = calculate_roc_curve(data, labels, model, training, test, svm, n_classes)
        fpr.append(f)
        tpr.append(t)
        roc_auc.append(r)

    plt.figure()
    lw = 2
    plt.plot([0,1],[0,1],lw=lw,linestyle='--')
    for f, t, r, l in zip(fpr, tpr, roc_auc, label_list):
        plt.plot(f[2], t[2], lw=lw, label=l+'(area = %0.2f)'%r[2])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC de los modelos estudiados')
    plt.legend(loc="lower right")
    plt.savefig(filename + ".png")


def calculate_roc_curve(data, labels, model, training, test, svm, n_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    if svm:
        y_score = model.fit(X=data[training], y=labels[training]).decision_function(data[test])
    else:
        rt_lm = LogisticRegression()
        pipeline = make_pipeline(model, rt_lm)
        pipeline.fit(data[training],labels[training])
        y_score = pipeline.predict_proba(data[test])

    y = label_binarize(y=labels, classes=list(range(y_score[0].size)))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[test][:,i], y_score[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y[test].ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr, tpr, roc_auc


def multiclass_roc_curve(data, labels, model, filename, training, test, svm, label_list, n_classes=17):
    fpr, tpr, roc_auc = calculate_roc_curve(data, labels, model, training, test, svm, n_classes)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    lw=2
    # Plot all ROC curves
    plt.figure(figsize=(15, 10))
    ax = plt.subplot(111)
    ax.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    ax.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'mediumseagreen', 'orchid', 'lightpink',
                    'darkslateblue', 'purple', 'darkblue', 'skyblue', 'red', 'firebrick',
                    'coral', 'maroon', 'grey', 'skyblue', 'seagreen'])
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='Class {} (area = {1:0.2f})'
                       ''.format(label_list[i], roc_auc[i]))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC por clases para KAZE con k=500')
    # plt.legend(loc="lower right")
    plt.savefig(filename + ".png")

"""
Cross validation functions to test SVM and RF
"""
def cv_rf(data,nlabels, kfold):
    rfb = RandomForestClassifier(n_estimators=60, criterion="entropy", oob_score=True, n_jobs=-1)
    rfn = RandomForestClassifier(n_estimators=60, criterion="entropy", oob_score=False, n_jobs=-1, bootstrap=False)
    scoresrfb = cross_val_score(rfb, data, nlabels, cv=kfold)
    scoresrfn = cross_val_score(rfn, data, nlabels, cv=kfold)
    print("Random Forest cross validation accuracy:")
    print("\tWith boosting")
    print("\t\tBest: %0.2f" % scoresrfb.max())
    print("\t\tAccuracy: %0.2f (+/- %0.2f)" % (scoresrfb.mean(), scoresrfb.std() * 2))
    print("\tWithout boosting")
    print("\t\tBest: %0.2f" % scoresrfn.max())
    print("\t\tAccuracy: %0.2f (+/- %0.2f)" % (scoresrfn.mean(), scoresrfn.std() * 2))
    return rfb, rfn


def cv_svm(data, nlabels, kfold):
    svm_onevsall = SVC(cache_size=200, C=180, gamma=0.5, tol=1e-7, shrinking=False, decision_function_shape='ovr')
    svm_onevsone = SVC(cache_size=200, C=180, gamma=0.5, tol=1e-7, shrinking=False, decision_function_shape='ovo')
    print("SVM cross validation accuracy:")
    scores_onevsall = cross_val_score(svm_onevsall, data, nlabels, cv=kfold)
    print("\tSVM one vs all:")
    print("\t\tBest: %0.2f"%scores_onevsall.max())
    print("\t\tAccuracy: %0.2f (+/- %0.2f)" % (scores_onevsall.mean(), scores_onevsall.std() * 2))
    scores_onevsone = cross_val_score(svm_onevsone, data, nlabels, cv=kfold)
    print("\tSVM one vs one:")
    print("\t\tBest: %0.2f"%scores_onevsone.max())
    print("\t\tAccuracy: %0.2f (+/- %0.2f)" % (scores_onevsone.mean(), scores_onevsone.std() * 2))
    return svm_onevsall, svm_onevsone