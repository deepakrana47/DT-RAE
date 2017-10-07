import os
from utility1 import extract_batchfeature_using_senna, get_parents, get_dep, get_words_id, pdep_2_deporder_dep, dep_2_hid_var
from score_calc import get_rae_score_by_wd, get_rae_score_by_wd_dep
from dynamic_pooling import get_dynamic_pool
from text_process1 import line_processing, get_n_feature
import numpy as np
import pickle


def get_vect_data_by_dep(fname, pool_size, mtype, pf):
    # # # print fname, 'file processing'
    sent_fd = open(fname)
    sents = sent_fd.read().split('\n')
    slen = len(sents)
    # slen = 3
    schk = 20
    scount = 0
    asents_vect = []
    asent_score = []
    asents = []
    nscore = []
    nfeat = []

    while scount < slen:
        lscount = scount + schk if slen - scount > schk else slen
        all_sents = []
        score = []
        try:
            for sent in sents[scount:lscount]:
                a = sent.split('\t')
                if len(a) < 3:
                    continue
                score.append(float(a[0]))
                line1 = line_processing(a[1])
                line2 = line_processing(a[2])
                all_sents += [line1, line2]
                nfeat.append(get_n_feature(line1, line2))
        except IndexError:
            pass
        wds = extract_batchfeature_using_senna(all_sents)
        wd_len = len(wds)
        wd_count = 0
        scor_count = 0

        while wd_count < wd_len:
            try:
                temp_score, _, s_matrix = get_rae_score_by_wd_dep(wds[wd_count], wds[wd_count + 1],mtype=mtype)
            except KeyError:
                scor_count += 1
                wd_count += 2
                continue
            except IndexError:
                pass
            sents_vect = get_dynamic_pool(s_matrix, pool_size=pool_size, pf=pf)
            if not np.any(sents_vect):
                scor_count += 1
                wd_count += 2
                continue
            asents_vect.append(sents_vect)
            asent_score.append(score[scor_count])
            nscore.append(temp_score)
            asents.append(all_sents[wd_count]+'\t'+all_sents[wd_count+1])
            scor_count += 1
            wd_count += 2
        scount = lscount
        # print scount,
    # print
    sent_fd.close()
    return asents, asents_vect, asent_score, nscore, nfeat

def test_get_vect_data_by_dep(fsent, fpickle, fscore, fnfeat, pool_size, mtype, pf):
    # print pf
    # # # print fname, 'file processing'
    asents_vect = []
    asent_score = []
    asents = []
    nscore = []
    nfeat = []
    wds = pickle.load(open(fpickle,'rb'))
    sents = pickle.load(open(fsent,'rb'))
    scores = pickle.load(open(fscore,'rb'))
    nfeats = pickle.load(open(fnfeat,'rb'))


    for i in range(len(sents)):
        temp_score, _, s_matrix = get_rae_score_by_wd_dep(wds[i][0], wds[i][1],mtype=mtype)
        sents_vect = get_dynamic_pool(s_matrix, pool_size=pool_size, pf=pf)
        if not np.any(sents_vect):
            continue
        asents_vect.append(sents_vect)
        asent_score.append(scores[i])
        nscore.append(temp_score)
        asents.append(sents[i][0]+'\t'+sents[i][1])
        nfeat.append(nfeats[i])
    return asents, asents_vect, asent_score, nscore, nfeat


def get_vect_data(fname, pool_size, pf):
    # # print fname, 'th sentence processing'
    sent_fd = open(fname)
    sents = sent_fd.read().split('\n')
    slen = len(sents)
    # slen = 3000
    schk = 10
    scount = 0
    asents_vect = []
    asent_score = []
    nscore = []
    asents = []

    while scount < slen:
        lscount = scount + schk if slen - scount > schk else slen
        all_sents = []
        score = []
        try:
            for sent in sents[scount:lscount]:
                a = sent.split('\t')
                if len(a) < 3:
                    continue
                score.append(float(a[0]))
                all_sents += [line_processing(a[1]), line_processing(a[2])]
        except IndexError:
            pass
        wds = extract_batchfeature_using_senna(all_sents)
        wd_len = len(wds)
        wd_count = 0
        scor_count = 0

        while wd_count < wd_len:
            try:
                temp_score, _, s_matrix = get_rae_score_by_wd(wds[wd_count], wds[wd_count + 1], mtype='deep')
            except KeyError:
                scor_count += 1
                wd_count += 2
                continue
            except IndexError:
                pass
            sents_vect = get_dynamic_pool(s_matrix, pool_size=pool_size, pf=pf)
            if not np.any(sents_vect):
                scor_count += 1
                wd_count += 2
                continue
            asents_vect.append(sents_vect)
            asent_score.append(score[scor_count])
            nscore.append(temp_score)
            asents.append(all_sents[wd_count]+'\t'+all_sents[wd_count+1])
            scor_count += 1
            wd_count += 2
        scount = lscount
        # # print scount,
    # # print
    sent_fd.close()
    return asents, asents_vect, asent_score, nscore

def test_data_set_maker_by_wd(flag=None, base_dir = None, out_dir=None, pool_size=10, num_feat=1, stp=None, mtype='Normal', pf=None):

    if flag == 'train':
        v_csv_file = out_dir + 'train_vector_dataset.csv'
        sent_file = out_dir + 'train_sent_dataset.txt'
        nscore_txt = out_dir + 'training_orig_score.pickle'
        src_dir = base_dir + 'train/'
        tpickle = src_dir + 'msr_paraphrase_train' + str(stp) + '.pickle'
        tsent= src_dir+ 'msr_paraphrase_trainsent'+ str(stp) + '.pickle'
        tscore= src_dir+ 'msr_paraphrase_trainscore'+ str(stp) + '.pickle'
        tnfeat= src_dir+ 'msr_paraphrase_trainnfeat'+ str(stp) + '.pickle'
    elif flag == 'test':
        v_csv_file = out_dir + 'test_vector_dataset.csv'
        sent_file = out_dir + 'test_sent_dataset.csv'
        nscore_txt = out_dir + 'test_orig_score.pickle'
        src_dir = base_dir + 'test/'
        tpickle = src_dir+'msr_paraphrase_test'+str(stp)+'.pickle'
        tsent = src_dir + 'msr_paraphrase_testsent' + str(stp) + '.pickle'
        tscore = src_dir + 'msr_paraphrase_testscore' + str(stp) + '.pickle'
        tnfeat = src_dir + 'msr_paraphrase_testnfeat' + str(stp) + '.pickle'

    if os.path.isfile(v_csv_file):
        if open(v_csv_file,'r').readline():
            # print "Already present :"
            return v_csv_file, sent_file

    data_csv_fd = open(v_csv_file,'w')
    sents_fd = open(sent_file,'w')
    all_nscore = []
    all_nfeat = []


    sents, sents_vect, score, nscore, nfeat = test_get_vect_data_by_dep(fpickle = tpickle,fsent=tsent, fscore=tscore, fnfeat=tnfeat, pool_size=pool_size, mtype=mtype, pf=pf)
    all_nscore += score
    all_nfeat += nfeat
    csv_txt = ''
    sent_txt = ''
    for i in range(len(sents_vect)):
        csv_txt += str(score[i])
        sent_txt += sents[i]+'\n'
        for j in sents_vect[i].reshape(pool_size*pool_size):
            csv_txt += ','+str(j)
        if num_feat == 1:
            for j in nfeat[i]:
                csv_txt += ',' + str(j)
        csv_txt += '\n'
    data_csv_fd.write(csv_txt)
    sents_fd.write(sent_txt)
    pickle.dump(all_nscore,open(nscore_txt, 'wb'))
    data_csv_fd.close()
    sents_fd.close()
    return v_csv_file, sent_file

def data_set_maker_by_wd(flag=None, base_dir = None, out_dir=None, pool_size=10, num_feat=1, mtype='Normal', pf=None):

    if flag == 'train':
        v_csv_file = out_dir + 'train_vector_dataset.csv'
        sent_file = out_dir + 'train_sent_dataset.txt'
        nscore_txt = out_dir + 'training_orig_score.pickle'
        src_dir = base_dir + 'train/'
        file_list = os.listdir(src_dir)
    elif flag == 'test':
        v_csv_file = out_dir + 'test_vector_dataset.csv'
        sent_file = out_dir + 'test_sent_dataset.csv'
        nscore_txt = out_dir + 'test_orig_score.pickle'
        src_dir = base_dir + 'test/'
        file_list = os.listdir(src_dir)

    if os.path.isfile(v_csv_file):
        if open(v_csv_file,'r').readline():
            # print "Already present :"
            return v_csv_file, sent_file

    data_csv_fd = open(v_csv_file,'w')
    sents_fd = open(sent_file,'w')
    all_nscore = []
    all_nfeat = []

    for i in range(len(file_list)):
        sents, sents_vect, score, nscore, nfeat = get_vect_data_by_dep(src_dir +file_list[i], pool_size=pool_size, mtype=mtype, pf=pf)
        all_nscore += score
        all_nfeat += nfeat
        csv_txt = ''
        sent_txt = ''
        for i in range(len(sents_vect)):
            csv_txt += str(score[i])
            sent_txt += sents[i]+'\n'
            for j in sents_vect[i].reshape(pool_size*pool_size):
                csv_txt += ','+str(j)
            if num_feat == 1:
                for j in nfeat[i]:
                    csv_txt += ',' + str(j)
            csv_txt += '\n'
        data_csv_fd.write(csv_txt)
        sents_fd.write(sent_txt)
    pickle.dump(all_nscore,open(nscore_txt, 'wb'))
    data_csv_fd.close()
    sents_fd.close()
    return v_csv_file, sent_file

def wd_making(fname,stp):
    sent_fd = open(fname)
    sents = sent_fd.read().rstrip(' |\n').split('\n')
    count = 1
    nfeat = []
    wds = []
    all_sents = []
    score = []
    for sent in sents:
        a = sent.split('\t')
        if len(a) < 3:
            continue
        score.append(float(a[0]))
        line1 = line_processing(a[1])
        line2 = line_processing(a[2])
        all_sents.append([line1, line2])
        nfeat.append(get_n_feature(line1, line2))
        temp = extract_batchfeature_using_senna([line1, line2])
        print count,
        if len(temp) !=2:
            print "not all sentences parsed !!"
            pass
        wds.append(temp)
        count +=1
    pickle.dump(wds,open(fname.split('.')[0]+str(stp)+'.pickle','wb'))
    pickle.dump(all_sents,open(fname.split('.')[0]+'sent'+str(stp)+'.pickle','wb'))
    pickle.dump(score,open(fname.split('.')[0]+'score'+str(stp)+'.pickle','wb'))
    pickle.dump(nfeat,open(fname.split('.')[0]+'nfeat'+str(stp)+'.pickle','wb'))
    return

import Global
if __name__=='__main__':

    testf = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/NN-dataset/MSRParaphraseCorpus/test/msr_paraphrase_test.txt'
    trainf = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/NN-dataset/MSRParaphraseCorpus/train/msr_paraphrase_train.txt'
    Global.init()
    # Global.init_wv_self(0,50)
    # wd_making(testf,0)
    # wd_making(trainf,0)
    Global.init_wv_self(1,50)
    wd_making(testf,1)
    wd_making(trainf,1)