import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from sklearn.neural_network import MLPClassifier
from sklearn import svm
import numpy as np
import pickle

def test_nn(test_fname, train_fname, test_sent_f, fd, out_dir, pool_size):

    sents = open(test_sent_f,'r').read().split('\n')
    trainset = np.loadtxt(train_fname, delimiter=',')
    X_train = trainset[:, 1:]
    y_train = trainset[:,0]


    testset = np.loadtxt(test_fname, delimiter=',')
    X_test = testset[:, 1:]
    y_test = testset[:,0]


    # clf = MLPClassifier(activation='logistic', solver='adam',alpha=0.0001,batch_size=100,learning_rate='adaptive',max_iter=10000,tol=1e-5, verbose=0)
    #
    clf = svm.LinearSVC(penalty='l2',tol=0.001, C=1.0, loss='hinge', fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, random_state=7, max_iter=100000)
    #
    # clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)

    clf.fit(X_train, y_train)
    # pickle.dump(clf, open(out_dir+'nn_'+str(pool_size),'w'))

    score = clf.predict(X_test)
    # oscore = clf.decision_function(X_test)

    pickle.dump(score, open(out_dir+'test_nn_obtain_score.pickle','wb'))
    # pickle.dump(oscore, open(out_dir+'test_nn_obtain_dec.pickle','wb'))

    zndet = 0;ondet=0;tp=0.0;fp=0.0;fn=0.0;tn=0.0;
    for i in range(len(y_test)):
        # print "desire score :",y_test[i],"obtained :",score[i],"sentences :",sents[i]
        fd.write("\ndesire score : " + str(y_test[i]) + " obtained : " + str(score[i]) + " sentences : " + sents[i] + '\n')
        if y_test[i] == 1:
            if score[i] == 1:
                tp += 1
            elif score[i] == 0:
                fn += 1
                # print "desire score :", y_test[i], "obtained :", score[i], "value :", oscore[i], "sentences :", sents[i]
        elif y_test[i] == 0:
            if score[i] == 1:
                fp += 1
                # print "desire score :", y_test[i], "obtained :", score[i], "value :", oscore[i], "sentences :", sents[i]
            elif score[i] == 0:
                tn += 1
    # print 'total test object determined :', tp + tn + fn + fp
    # print "correct predict :", tp + tn
    # print 'true positive :', tp
    # print 'flase positive :', fp
    # print 'false negative :', fn
    # print 'true negative :', tn
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    acc = (tp + tn) / (tp + fp + tn + fn)
    # print 'precision', precision
    # print 'recall', recall
    # print 'F score', 100 * f1
    # print 'accuracy', acc * 100


    fd.write('\ntotal test object : '+ str(len(sents))+\
                 '\ntrue positive : '+ str(tp)+\
                 '\nflase positive : '+ str(fp)+\
                 '\nfalse negative : '+ str(fn)+\
                 '\ntrue negative : '+ str(tn)+\
                 '\nprecision '+ str(precision)+\
                 '\nrecall : '+ str(recall)+\
                 '\nF score : '+ str(f1)+\
                 '\naccuracy : '+ str(acc))
    return acc, f1

if __name__ == "__main__":
    train = '/home/zero/Desktop/top/5new_deep_herar_w_test_v_200___10_1_max/train_vector_dataset.csv'
    test = '/home/zero/Desktop/top/5new_deep_herar_w_test_v_200___10_1_max/test_vector_dataset.csv'
    sent = '/home/zero/Desktop/top/5new_deep_herar_w_test_v_200___10_1_max/test_sent_dataset.csv'
    fd= open('/home/zero/Desktop/top/5new_deep_herar_w_test_v_200___10_1_max/nn_result.txt','wb')
    test_nn(test, train, sent, fd, out_dir='/home/zero/Desktop/top/5new_deep_herar_w_test_v_200___10_1_max/', pool_size=10)