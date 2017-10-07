from onlinedata_process import get_chk_vect,get_vect_by_wd_dep, get_chk_vect_by_wd#, get_sent_vect
import numpy as np
import Global

def cosine_sim(a,b):
    return np.sum(a * b) / ((np.sum(a ** 2) ** .5) * (np.sum(b ** 2) ** .5))

def ecludian_dist(a,b):
    return np.sqrt(np.sum(np.square(a-b)))

# def get_sent_score(sent1, sent2):
#     vect1= get_sent_vect(sent1)
#     vect2= get_sent_vect(sent2)
#     return ecludian_dist(vect1, vect2)

def get_rae_score_by_wd(wd_1, wd_2, mtype='normal'):
    # wd_ = {1:{wid :1,pid:3,pos:'af',chk:,nre:},2:{}, ..}
    chk1, ch_vect1 = get_chk_vect_by_wd('t', wd_1, mtype)
    chk2, ch_vect2 = get_chk_vect_by_wd('t', wd_2, mtype)

    s_matrix = np.zeros((len(chk1), len(chk2)))
    for i in ch_vect1:
        for j in ch_vect2:
            # if len(chk1[i].split(' ')) == 1 and len(chk2[j].split(' ')) == 1:
            #     if chk1[i] in Global.word_vec and chk2[j] in Global.word_vec:
            #         s_matrix[i, j] = ecludian_dist(Global.word_vec[chk1[i]], Global.word_vec[chk2[j]])
            #     else:
            #         s_matrix[i, j] = ecludian_dist(ch_vect1[i], ch_vect2[j])
            # else:
            #     s_matrix[i, j] = ecludian_dist(ch_vect1[i], ch_vect2[j])

            s_matrix[i, j] = ecludian_dist(ch_vect1[i], ch_vect2[j])

    s_min = {}
    for i in ch_vect1:
        s_min[(chk1[i], chk2[np.argmin(s_matrix[i,])])] = np.amin(s_matrix[i,])

    s = 0.0
    for i in s_min:
        s += s_min[i]
    score = s / len(s_min)
    return score, s_min, s_matrix


def get_rae_score_by_wd_dep(wd_1, wd_2, mtype='normal'):
    # wd_ = {1:{wid :1,pid:3,pos:'af',chk:,nre:},2:{}, ..}
    chk1, ch_vect1 = get_vect_by_wd_dep('t', wd_1, mtype)
    chk2, ch_vect2 = get_vect_by_wd_dep('t', wd_2, mtype)

    s_matrix = np.zeros((len(chk1), len(chk2)))
    for i in ch_vect1:
        for j in ch_vect2:
            # if len(chk1[i].split(' ')) == 1 and len(chk2[j].split(' ')) == 1:
            #     if chk1[i] in Global.word_vec and chk2[j] in Global.word_vec:
            #         s_matrix[i, j] = ecludian_dist(Global.word_vec[chk1[i]], Global.word_vec[chk2[j]])
            #     else:
            #         s_matrix[i, j] = ecludian_dist(ch_vect1[i], ch_vect2[j])
            # else:
            #     s_matrix[i, j] = ecludian_dist(ch_vect1[i], ch_vect2[j])

            s_matrix[i, j] = ecludian_dist(ch_vect1[i], ch_vect2[j])

    s_min = {}
    for i in ch_vect1:
        s_min[(chk1[i], chk2[np.argmin(s_matrix[i,])])] = np.amin(s_matrix[i,])

    s = 0.0
    for i in s_min:
        s += s_min[i]
    score = s / len(s_min)
    return score, s_min, s_matrix

def get_rae_score(sent1, sent2, chk1=None, ch_vect1=None, chk2=None, ch_vect2=None,  mtype='normal'):
    if chk1 == None or ch_vect1 == None:
        chk1, ch_vect1 = get_chk_vect('t', sent1, mtype)
    if chk2 == None or ch_vect2 == None:
        chk2, ch_vect2 = get_chk_vect('t', sent2, mtype)

    s_matrix = np.zeros((len(chk1),len(chk2)))
    for i in ch_vect1:
        for j in ch_vect2:
            if len(chk1[i].split(' ')) == 1 and len(chk2[j].split(' ')) == 1:
                # t1 = ecludian_dist(ch_vect1[i],ch_vect2[j])
                # t2 = np.inf
            #     if chk1[i] in Global.word_vec and chk2[j] in Global.word_vec:
            #         s_matrix[i, j] = ecludian_dist(Global.word_vec[chk1[i]], Global.word_vec[chk2[j]])
            #     else:
            #         s_matrix[i, j] = ecludian_dist(ch_vect1[i],ch_vect2[j])
            # else:
                s_matrix[i,j] = ecludian_dist(ch_vect1[i],ch_vect2[j])

    s_min = {}
    for i in ch_vect1:
        s_min[(chk1[i],chk2[np.argmin(s_matrix[i,])])] = np.amin(s_matrix[i,])

    s=0.0

    for i in s_min:
        s+=s_min[i]
    score = s/len(s_min)
    return score, s_min, s_matrix

import Global
if __name__=='__main__':
    Global.init()
    Global.init_wv_self(0,200)
    Global.wfname = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/test/new_deep_herar_w_st_test7_v_200_/new_deep_herar_w_st_test7_v_200__1_.pickle'

    print "score by weight :",Global.wfname
    # sent1 = 'bodies 18 illegal mexican immigrants died suffocation heat exhaustion discovered wednesday packed tractor trailer abandoned rest stop'
    # sent2 = 'bodies 17 undocumented immigrants suffocated stifling trailer discovered yesterday south texas truck stop smugglers abandoned'
    sent1='national city shares down 15 cents 32.43 new york stock exchange'
    sent2 = 'national city stock ended day 32.58 up 9 cents trading new york stock exchange'
    score, s_min, m = get_rae_score(sent1,sent2,mtype='deep')
    print m
    print score
    for i in s_min:
        print i,':',s_min[i]
