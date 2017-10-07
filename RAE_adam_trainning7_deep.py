import os
import pickle
import random
import sys
from random import shuffle

import numpy as np

from Global import get_word_vect
from RAE_adam_herical_deep1 import rae_trainning_normal
from utility1 import get_words_id, get_parents, get_dep, pdep_2_deporder_dep, dep_2_hid_var, save2pickle, get_weight_matrices, init_weight, remove_dep ,get_zero_weight_matrices, zero_weight


import Global

def cleanwd(wds):
    rmwd = []
    for i in wds:
        if wds[i]['pid']==-1:
            rmwd.append(i)
    for j in rmwd:
        wds.pop(j)
    return wds

def wd_preprocess(word_data):
    twd = cleanwd(word_data)
    d = get_dep(twd)
    try:
        d, req_ = remove_dep(d)
    except MemoryError:
        print get_dep(twd)
        return -1,-1,-1,-1,-1
    idlist = [i for i in twd]
    if req_:
        for id in idlist:
            if (twd[id]['wid'],twd[id]['pid']) in req_:
                twd.pop(id)

    idlist = [i for i in twd if type(i) is not int]
    for wd in idlist:
        twd[int(wd)] = twd[wd]
        twd.pop(wd)
    p = get_parents(twd)
    Word_ids = get_words_id(twd)
    w_size = len(Word_ids)
    dep_order, d1 = pdep_2_deporder_dep(p, d)
    h_index, h_vect, wp , hh_index = dep_2_hid_var(p, dep_order, d1, Word_ids)
    return h_index, h_vect, wp, Word_ids, w_size, dep_order, hh_index, p

def word_to_extra(words_data, w1, i_size, h1_size, h2_size,neta):
    wd_extra = []
    wpresent = []
    for iword in range(len(words_data)):

        if words_data[iword] == -2:
            continue

        h_index, h_vect, wp, Word_ids, w_size, dep_order, hh_index, p = wd_preprocess(words_data[iword])
        if h_index == -1:
            continue

        cflag = 0
        for i in wp:
            if len(i) > abs(w1_range / 2):
                cflag = 1
                break
            for j in i:
                wpresent.append(j)
                if type(w1[j]) != np.ndarray:
                    w1[j] = init_weight(h1_size, h1_size)
                    Global.m[j] = zero_weight(h1_size, h1_size)
                    Global.v[j] = zero_weight(h1_size, h1_size)
                    Global.lr[j] = neta[0]

        if cflag == 1:
            continue
        wpresent = list(set(wpresent))
        Word_vects = []
        try:
            for i in sorted(words_data[iword]):
                Word_vects.append(get_word_vect(words_data[iword][i]['word'].lower(), Global.v_size))
        except KeyError:
            continue

        wd_extra.append({'w_size': w_size, 'h_index': h_index, 'h_vect': h_vect, 'Word_vects': Word_vects, 'wp': wp,
                         "dep_order": dep_order, "hh_index": hh_index, 'Word_ids': Word_ids, 'p': p})
    return wd_extra, wpresent

def log_data(text,fname):
    open(fname,'a+').write(text+'\n')
    return

from list_data import list_1perdir_files
def train(inflag, src, mode, w1, w2, neta, regu, v_size = 50, h1_size = 40, h2_size=30, w1_range = 51, ite=50, ep_size=1000, wfname = 'weight',debug = 1, log=None):

    ## errors of various type
    iter_err = {0:np.inf}
    epoc_err = []
    epoc_samp_err = []
    epoc_samp_branch_err = []

    # samp_err = []
    # branch_samp_err = []
    ## errors of various type end

    ecount = 1
    if w1 is None:
        w1 = get_weight_matrices(w1_range, h1_size, v_size)
    Global.m = get_zero_weight_matrices(w1_range, h1_size, v_size)
    Global.v = get_zero_weight_matrices(w1_range, h1_size, v_size)
    Global.lr = {0:neta[0]}
    if w2 is None:
        w2 = init_weight(h2_size, h1_size)
    Global.m2 = zero_weight(h2_size, h1_size)
    Global.v2 = zero_weight(h2_size, h1_size)

    iter_count=1
    of_flag = []
    log_text = ''

    if os.path.isfile(log['iter']):
        iter_count=pickle.load(open(log['iter'],'rb'))
    if os.path.isfile(log['iter_err']):
        iter_err= pickle.load(open(log['iter_err'],'rb'))

    if inflag == 'd':
        if debug == 1:
            print "Training Directory :", src
            log_text += "\nTraining Directory : "+ src
    elif inflag == 't':
        if debug == 1:
            print "Training file :", src,"\n\n"
            log_text += "\nTraining file : " + src

    while iter_count < ite:
        words_data = []
        if inflag == 'd':

            file_list = list_1perdir_files(src)
            if debug == 1:
                print iter_count, "iteration is running\n\tfiles read :"
                log_text += '\n' + str(iter_count) + " iteration is running\n\tfiles read : "
            for fname in file_list:
                if debug == 1:
                    print '\t', fname
                    log_text += '\n\t' + fname
                words_data += pickle.load(open(fname, 'rb'))
        elif inflag == 't':
            if debug == 1:
                print iter_count, "iteration is running"
                log_text += '\n' + str(iter_count) + " iteration is running"
            words_data = pickle.load(open(src, 'rb'))

        iter_err[iter_count]=0.0
        total_sample = 0

        wd_extra, wpresent = word_to_extra(words_data, w1, i_size=v_size, h1_size=h1_size, h2_size=h2_size,neta=neta)
        del words_data
        wd_len = len(wd_extra)

        if debug == 1:
            print "\n\tStart trainning "
            print '\tnumber of training sample :', wd_len
            log_text += "\n\n\tStart trainning\n\tnumber of training sample : "+str(wd_len)

        index=0;
        wsum1 = 0.0;
        kerr=0.0;
        wsum11 = 0.0;
        shuffle(wd_extra)
        while index < wd_len:

            wtrain = (index + ep_size) if index + ep_size < wd_len else wd_len

            terr = 0.0;
            tkerr=0.0
            # try:
            if mode == 'normal':
                w1, w2, _, _, tkerr, terr, tsamp_err, tbranch_samp_err, branchs= rae_trainning_normal(wd_extra[index:wtrain], w1=w1, w2=w2, neta=neta, regu=regu, wpresent= wpresent)
                # w1, dw1, epnum1, tkerr, terr, tsamp_err, tbranch_samp_err, branchs= rae_trainning_normal(wd_extra[index:wtrain], w1=w1, dw1=dw1,epnum1=epnum1, neta=neta, regu=regu)
                epoc_samp_err.append(terr/(wtrain-index))
                epoc_err.append(terr)
                epoc_samp_branch_err.append(terr/branchs)
            # print wtrain, " : finished"
                # samp_err += tsamp_err
                # branch_samp_err += tbranch_samp_err
            # except KeyError:
            #     tkerr += 1
            #     index += ep_size
            #     continue
            index += ep_size
            iter_err[iter_count] += terr
            kerr += tkerr
        # raw_input()

        for wi in w1:
            wsum1 += np.sum(abs(w1[wi]))
            wsum11 += np.sum(w1[wi])
        wsum2 = np.sum(abs(w2))
        wsum21 = np.sum(w2)

        save2pickle({'w1':w1,'w2':w2}, wfname)

        total_sample+=wd_len

        if debug == 1:
            print "\tWeight1 sum :", wsum1
            print "\tWeight1 sum1 :", wsum11
            print "\tWeight2 sum :", wsum2
            print "\tWeight2 sum1 :", wsum21
            print "\tKey error :", kerr
            print "\tTotal sample trained :", total_sample
            print "Prev iteration error :", iter_err[iter_count - 1]
            print "This iteration error :", iter_err[iter_count]
            print "break lower Threshold :", .001 * total_sample
            print "break value :", abs(iter_err[iter_count - 1] - iter_err[iter_count])

            log_text += "\n\tWeight sum : " + str(wsum1) + "\n\tWeight sum : " + str(wsum2) + "\n\tError sum : " + str(
                kerr) + "\n\tTotal sample trained : " + str(total_sample) \
                        + "\nPrev iteration error : " + str(
                iter_err[iter_count - 1]) + "\nThis iteration error : " + str(iter_err[iter_count]) + \
                        "\nbreak lower Threshold : " + str(.001 * total_sample) + "\nbreak value : " + str(
                abs(iter_err[iter_count - 1]) - abs(iter_err[iter_count]))

            pickle.dump(iter_err, open(log['iter_err'], 'wb'))

            pickle.dump(epoc_err, open(log['epoc_err'] + str(ecount), 'wb'))
            pickle.dump(epoc_samp_err, open(log['epoc_samp_err'] + str(ecount), 'wb'))
            pickle.dump(epoc_samp_branch_err, open(log['epoc_samp_branch_err'] + str(ecount), 'wb'))
            if len(epoc_err) > 10000:
                epoc_err = []
                epoc_samp_err = []
                epoc_samp_branch_err = []
                ecount += 1
        # if abs(iter_err[iter_count - 1]) - abs(iter_err[iter_count]) < .001 * total_sample:
        #     of_flag.append(1)
        #     if debug == 1:
        #         print "Number of Boundry cross :", len(of_flag)
        #         log_text += "\nNumber of Boundry cross : " + str(len(of_flag))
        #     if len(of_flag) > 2:
        #         t1 = wfname.split('/');
        #         wf = '/'.join(t1[:-1]) + '/' + str(iter_count) + t1[-1]
        #         save2pickle({"w1": w1, "w2": w2}, wf)
        #         pickle.dump(iter_count+1, open(log['iter'], 'wb'))
        #         break
        # else:
        #     if of_flag:
        #         of_flag.pop()


        if abs(iter_err[iter_count - 1] - iter_err[iter_count]) < .001 * total_sample:
            of_flag.append(1)
            if debug == 1:
                print "Number of Boundry cross :", len(of_flag)
                log_text += "\nNumber of Boundry cross : " + str(len(of_flag))
            if len(of_flag) > 0:
                break
        else:
            if of_flag:
                of_flag.pop()
        # if iter_count % 5 == 0:
        t1 = wfname.split('/');
        wf = '/'.join(t1[:-1]) + '/' + str(iter_count) + t1[-1]
        save2pickle({"w1":w1,"w2":w2}, wf)
        if debug == 1:
            print '\n\n'
            log_text += '\n\n\n'
            log_data(log_text, log['rae'])
            log_text = ''
            # if iter_count%4 == 0:
            #     pickle.dump(samp_err,open(log['samp_err'] + str(iter_count),'wb'))
            #     pickle.dump(branch_samp_err,open(log['branch_samp_err']+ str(iter_count),'wb'))
            #     samp_err=[]
            #     branch_samp_err=[]

        iter_count += 1
        pickle.dump(iter_count, open(log['iter'], 'wb'))
    for i in iter_err:
        if debug == 1:
            print i, "iteration error :", iter_err[i]
            log_text += '\n' + str(i) + " iteration error : " + str(iter_err[i])
    if debug == 1:
        if epoc_err:
            pickle.dump(epoc_err, open(log['epoc_err'] + str(ecount), 'wb'))
            pickle.dump(epoc_samp_err, open(log['epoc_samp_err'] + str(ecount), 'wb'))
            pickle.dump(epoc_samp_branch_err, open(log['epoc_samp_branch_err'] + str(ecount), 'wb'))
        log_data(log_text, log['rae'])
    return


import time
if __name__=='__main__':
    Global.init()
    flag=None
    w = {}
    neta = None; regu = 0.01; v_size = None; h1_size = None; h2_size=None; w1_range = 201; ite = 5; epoch = 1000; wname = 'weight(w)';src=None;wload=None;spwd=0
    log = {}

    if sys.argv[1] == '--help':
        print "Usage : python RAE_trainning.py [options] src"
        print "options:"
        print "\t -dir directory_name: directory has the files contain pickle data for trainning"
        print "\t -in infile: infile has location of pickle file for trainning"
        print "\t -neta value: learning rate(default=0.001)"
        print "\t -hlayer value: Number of nodes in hidden layer"
        print "\t -insize value: word vector size(50 or 200)"
        print "\t -regu value: reguleraziation (default=0.01)"
        print "\t -w weight_file_name: filename contain weights parameter for machine"
        print "\t -wload weight_file_name: filename contain weights parameter for machine(when weight of file as initiate weight)"
        print '\t -iter iterations'
        print '\t -stopwrd include stopword(by default 0)'
        print '\t -epoch epoch size\n\n'
        exit()
    elif '-dir' in sys.argv:
        ind=sys.argv.index('-dir')
        if os.path.isdir(sys.argv[ind+1]):
            flag ='d'
            src = sys.argv[ind+1]
    elif '-in' in sys.argv:
        ind = sys.argv.index('-in')
        if os.path.isfile(sys.argv[ind+1]):
            flag='t'
            src = sys.argv[ind + 1]
        else:
            print "Input file not Present !"
            exit()
    else:
        exit()
    if '-neta' in sys.argv:
        ind=sys.argv.index('-neta')
        t1 = sys.argv[ind + 1].strip(']|[').split(',')
        neta = [float(t1[0]),float(t1[1])]
    if '-regu' in sys.argv:
        ind=sys.argv.index('-regu')
        t1 = sys.argv[ind + 1].strip(']|[').split(',')
        regu = [float(t1[0]),float(t1[1])]
    if '-w' in sys.argv:
        ind=sys.argv.index('-w')
        wname = sys.argv[ind+1]
    if '-iter' in sys.argv:
        ind=sys.argv.index('-iter')
        ite = int(sys.argv[ind+1])
    if '-stopwrd' in sys.argv:
        ind=sys.argv.index('-stopwrd')
        spwd = int(sys.argv[ind+1])
    if '-epoch' in sys.argv:
        ind=sys.argv.index('-epoch')
        epoch = int(sys.argv[ind+1])
    if '-wload' in sys.argv:
        ind=sys.argv.index('-wload')
        wload = sys.argv[ind+1]
    if '-hlayer' in sys.argv:
        ind = sys.argv.index('-hlayer')
        t1 = sys.argv[ind + 1].strip(']|[').split(',')
        h1_size = int(t1[0])
        h2_size = int(t1[1])
    if '-insize' in sys.argv:
        ind = sys.argv.index('-insize')
        v_size = int(sys.argv[ind + 1])
        # if v_size != 50 or v_size != 200:
        #     print "Enter : -insize (50 or 200)"
        #     exit()
    wt = wname.split('/')[-1]

    wfile = '_'
    ddir = wname + wfile
    if not os.path.isdir(ddir):
        os.mkdir(ddir)
    if not os.path.isdir(ddir+'/error/'):
        os.mkdir(ddir+'/error/')

    wfname = ddir + '/' + wt + wfile + '_'+str(spwd)+'_.pickle'
    log['iter'] = ddir + '/iter_count.pickle'
    log['iter_err'] = ddir + '/iter_err_count.pickle'
    log['rae'] = ddir + '/log.txt'
    log['epoc_err'] = ddir + '/error/epoc_err.pickle'
    log['epoc_samp_err'] = ddir + '/error/epoc_samp_err.pickle'
    log['epoc_samp_branch_err'] = ddir + '/error/epoc_samp_branch_err.pickle'

    log['samp_err'] = ddir + '/error/samp_err.pickle'
    log['branch_samp_err'] = ddir + '/error/branch_samp_err.pickle'


    print "\n\nTrainning Start with:"
    print " neta =",neta
    print " regu=",regu
    print " word vector size =",v_size
    print " hidden layers size =" + str(h1_size)+" , "+str(h2_size)
    print " pharse vector size =",h2_size
    print " iteration :",ite
    print " Stopword :",spwd
    print " epoch size :",epoch
    print " Input flag :",flag
    print " Input file :",src
    print " weight file :",wfname

    open(ddir +'/setting.txt','w').write(\
        "\nneta ="+str(neta)\
        +"\nregu="+str(regu)\
        +"\nword vector size ="+str(v_size) \
        + "\nhidden layers size =" + str(h1_size)+" , "+str(h2_size) \
        +"\npharse vector size ="+str(h2_size)\
        +"\niteration :"+str(ite)\
        +"\nStopword :"+str(spwd)\
        +"\nepoch size :"+str(epoch)\
        +"\nInput flag :"+str(flag)\
        +"\nInput file :"+src\
        +"\nweight file :"+wfname
    )

    if os.path.isfile(wfname):
        print " load weight file :", wfname
    else:
        print " load weight file :",wload
    print

    # a=raw_input("Want to continue (y\Y):")
    # a=str(a)
    # if a == 'y' or a == 'Y':
    #     pass
    # else:
    #     exit()

    if wload:
        if os.path.isfile(wload):
            w = pickle.load(open(wload, 'rb'))
        else:
            print "Input weight file not exist"
    else:
        if os.path.isfile(wfname):
            w = pickle.load(open(wfname, 'rb'))
        else:
            w['w1']=None
            w['w2']=None
    Global.init_wv_self(stopword=spwd, vsize=v_size)
    Global.adam_parm_init()
    Global.lr = neta[0]
    Global.lr2 = neta[1]
    train(flag, src=src, mode='normal', w1=w['w1'], w2=w['w2'], neta=neta, regu=regu, v_size=Global.v_size, h1_size=h1_size, h2_size=h2_size, w1_range=w1_range, ite=ite, ep_size=epoch, wfname = wfname, log=log, debug=1)