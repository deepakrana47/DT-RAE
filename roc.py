"""
=======================================
Receiver Operating Characteristic (ROC)
=======================================

Example of Receiver Operating Characteristic (ROC) metric to evaluate
classifier output quality.

ROC curves typically feature true positive rate on the Y axis, and false
positive rate on the X axis. This means that the top left corner of the plot is
the "ideal" point - a false positive rate of zero, and a true positive rate of
one. This is not very realistic, but it does mean that a larger area under the
curve (AUC) is usually better.

The "steepness" of ROC curves is also important, since it is ideal to maximize
the true positive rate while minimizing the false positive rate.

Multiclass settings
-------------------

ROC curves are typically used in binary classification to study the output of
a classifier. In order to extend ROC curve and ROC area to multi-class
or multi-label classification, it is necessary to binarize the output. One ROC
curve can be drawn per label, but one can also draw a ROC curve by considering
each element of the label indicator matrix as a binary prediction
(micro-averaging).

Another evaluation measure for multi-class classification is
macro-averaging, which gives equal weight to the classification of each
label.

.. note::

    See also :func:`sklearn.metrics.roc_auc_score`,
             :ref:`sphx_glr_auto_examples_model_selection_plot_roc_crossval.py`.

"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


def ROC_plot(ddir, stp, num_feat, nhlayer, v_size):
    # ddir = '/media/zero/41FF48D81730BD9B/Final_Thesies/results/SVM_results/deep21_v_200__0__15_1_min/'
    trainset = np.loadtxt(ddir+'train_vector_dataset.csv', delimiter=',')
    X_train = trainset[:, 1:]
    ty_train = trainset[:,0]
    y_train = np.zeros((trainset.shape[0],2))
    for i in range(trainset.shape[0]):
        if ty_train[i] == 1:
            y_train[i,0] = 1
        else:
            y_train[i,1] = 1

    testset = np.loadtxt(ddir+'test_vector_dataset.csv', delimiter=',')
    X_test = testset[:, 1:]
    ty_test = testset[:,0]
    y_test = np.zeros((testset.shape[0],2))
    for i in range(testset.shape[0]):
        if ty_test[i] == 1:
            y_test[i,0] = 1
        else:
            y_test[i,1] = 1

    n_classes = 2

    classifier = OneVsRestClassifier(svm.LinearSVC(penalty='l2',tol=0.001, C=1.0, loss='hinge',
                    fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, random_state=None, max_iter=100000))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    lw = 2



    ##############################################################################
    # Plot ROC curves for the multiclass problem

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    plt.plot(fpr[0], tpr[0], color='aqua', lw=lw,
                 label='ROC curve of paraphrase {0} (area = {1:0.2f})'
                 ''.format(0, roc_auc[0]))
    plt.plot(fpr[1], tpr[1], color='darkorange', lw=lw,
                 label='ROC curve of nonparaphrase {0} (area = {1:0.2f})'
                 ''.format(1, roc_auc[1]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for (hidden layer='+str(nhlayer)+'; stopword='+str(stp)+'; number_feature='+str(num_feat)+'; vect_size='+str(v_size)+')')
    plt.legend(loc="lower right")
    plt.savefig(ddir+'ROC.png',dpi=1000)
    plt.close()

if __name__ == '__main__':
    ddir = '/media/zero/41FF48D81730BD9B/Final_Thesies/results/SVM_results/deep21_v_200__0__15_1_min/'
    ROC_plot(ddir)
