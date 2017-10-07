
import Global, os
from training_dataset_maker1 import test_data_set_maker_by_wd
from NN_test4 import test_nn
from roc import ROC_plot

def main_fun(weight_list=None, ddir=None, pool_size=10, num_feat = 1, mtype=None, stp=None, pf=None, v_size=None):
    base_dir = './data/'
    if weight_list is None or mtype is None:
        print "Weight is not provided !!"

    if not os.path.isfile(ddir + 'nn_results.txt'):
        fd1 = open(ddir + 'nn_results.txt', 'w')
    else:
        fd1 = open(ddir + 'nn_results.txt', 'a+')


    for i in range(len(weight_list)):
        w = weight_list[i]
        wstp = stp[i]

        Global.init_wv_self(stopword=wstp, vsize=v_size)
        temp = os.path.basename(w).split('.')

        print "start for :", temp[0], 'setting : pool_size=' + str(pool_size) + ' num_feat=' + str(num_feat) +" stopword="+str(wstp)+' aggr=' + pf
        fd1.write("start for : "+temp[0]+' setting : pool_size='+str(pool_size)+' num_feat='+str(num_feat)+" stopword="+str(wstp)+' aggr='+pf+'\n')

        mdir = '.'.join(temp[:-1])
        out_dir = ddir + mdir + '_' + str(pool_size) + '_' + str(num_feat) + '_' + pf + '/'
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        # fd2 = open(out_dir + 'nn_results.txt', 'a+')
        # open(out_dir+os.path.basename(w),'wb').write(open(w,'rb').read())

        Global.wfname = w
        # for j in range(1):

        train_fname, _= test_data_set_maker_by_wd(flag='train',base_dir=base_dir, out_dir=out_dir, stp=wstp, pool_size=pool_size, num_feat=num_feat, mtype = mtype[i], pf=pf)
        fd = open(out_dir+'nn_result.txt','w')
        # fd = open(out_dir+'SVM_result.txt','w')
        # train(train_fname, 'NN_train.csv', out_dir=out_dir, pool_size=pool_size)

        test_fname, test_sent_f = test_data_set_maker_by_wd(flag='test', base_dir=base_dir, out_dir=out_dir, stp=wstp, pool_size=pool_size, num_feat=num_feat, mtype = mtype[i], pf=pf)

        # acc, f1 = test(test_fname, sent_file, fd=fd, base_dir=base_dir, out_dir=out_dir, pool_size=pool_size, num_feat=num_feat)

        acc,f1 = test_nn(test_fname, train_fname, test_sent_f, fd, out_dir=out_dir, pool_size=pool_size)
        print '\t::\tAccuracy :', acc, '\tF1 score :', f1

        fd1.write('\t::\tAccuracy : '+str(acc)+'\tF1 score : '+str(f1)+'\n')
        # fd2.write(str(j)+" start for : "+temp[0]+' setting : pool_size='+str(pool_size)+' num_feat='+str(num_feat)+' aggr='+pf+'\n\tAccuracy : '+str(acc)+ '\n\tF1 score : '+ str(f1)+'\n')
        fd.close()

        nhid = -1
        if mtype[i] == 'deep':
            nhid = 2
        elif mtype[i] == 'normal':
            nhid = 1

        ROC_plot(out_dir, stp= wstp, num_feat=num_feat, nhlayer=nhid, v_size= v_size)
        # exit()
    fd1.close()

if __name__ == '__main__':
    Global.init()
    v_size = 200
    weights = [
        './weights/h1_wost_2__0_.pickle',
        './weights/h1_wst_2__1_.pickle',
        './weights/h2_wost_2__0_.pickle',
        './weights/h2_wst_2__1_.pickle',
    ]
    mtype =['normal','normal','deep','deep']
    spwd = [0,1,0,1]
    ddir = './results/'
    plist = [15]
    nflist = [0,1]
    pool_fun = 'min'
    for num_feat in nflist:
        for pool in plist:
            main_fun(weights, ddir=ddir, pool_size=pool, num_feat=num_feat, mtype=mtype, stp=spwd, pf=pool_fun, v_size=v_size)


