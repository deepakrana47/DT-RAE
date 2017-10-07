import numpy as np
from utility1 import sigmoid,dsigmoid,preprocess, get_parent_detail

def weight_update(w, dw, epnum, neta, regu, wpresent):
    for i in wpresent:
        if epnum[i] != 0:
            dw[i]=dw[i].transpose()
            w[i] = w[i] - neta*(dw[i]/epnum[i] + regu*(w[i]))
    return w

def rae_encoding(vect, w, w_size, h_vect, wp):
    for i in range(len(h_vect)):
        temp1=0.0
        # print w_size+i
        for j in range(len(h_vect[i])):
            # print '\t',h_vect[i][j],
            temp1 += np.dot(w[wp[i][j]],vect[h_vect[i][j]])
        # print
        vect[w_size + i] = sigmoid(temp1)
    return

def rae_decoding(vect, o, w, w_size, h_vect, wp):
    vect_ = {};
    vect_[2*w_size-1] = vect[2*w_size-1]
    o[2 * w_size - 1] = np.nan
    for i in reversed(range(len(h_vect))):
        # print w_size+i
        for j in range(len(h_vect[i])):
            # print '\t',h_vect[i][j],
            o[h_vect[i][j]] = np.dot(w[wp[i][j]].transpose(), vect_[w_size + i])
            vect_[h_vect[i][j]] = sigmoid(o[h_vect[i][j]])
        # print
    return vect_

def pre_gradiant(vect, vect_, o, h_vect, w, wp, w_size):
    grad={}
    cost = []
    for i in range(len(h_vect)):
        # print w_size+i
        grad[w_size + i] = 0.0
        for j in range(len(h_vect[i])):
            if h_vect[i][j] < w_size:
                # print '\t', h_vect[i][j],
                tcost = (np.power((vect[h_vect[i][j]] - vect_[h_vect[i][j]]),2))
                cost.append(tcost)
                grad[h_vect[i][j]] =  tcost * dsigmoid(o[h_vect[i][j]])
                grad[w_size + i] +=  np.dot(w[wp[i][j]], grad[h_vect[i][j]]) * dsigmoid(o[w_size + i])
            else:
                # print '\t', h_vect[i][j],
                grad[w_size + i] += np.dot(w[wp[i][j]], grad[h_vect[i][j]]) * dsigmoid(o[w_size + i])
        # print
    return grad, cost

# def gradiant(w, grad, n_parent, npa_sibling, self_wp, npa_wp, npas_wp):
#     herir_grad={}
#     for i in grad:
#         t1=0.0
#         if npa_sibling[i]:
#             for j in range(len(npa_sibling[i])):
#                 t1 = t1+np.dot(w[npas_wp[i][j]],grad[npa_sibling[i][j]])
#             # t1 = np.sum([np.dot(w[npas_wp[i][j]],grad[j]) for j in npa_sibling[i]])
#             t3 = npa_wp[i]
#             t2=w[t3].transpose()
#             temp1 = np.dot(t2, t1)
#             temp1 += grad[n_parent[i]]
#             t4 = np.dot(w[self_wp[i]].transpose(), temp1)
#         else:
#             t4 = 0
#         herir_grad[i] = grad[i] + t4
#         pass
#     return herir_grad

# def gradiant(w, grad, n_parent, npa_sibling, self_wp, npa_wp, npas_wp, neta):
#     herir_grad={}
#     for i in grad:
#         t1=0.0
#         if npa_sibling[i]:
#             for j in range(len(npa_sibling[i])):
#                 t1 += np.dot(w[npas_wp[i][j]],grad[npa_sibling[i][j]])
#             t4 = np.dot(w[self_wp[i]].transpose(), grad[n_parent[i]] + np.dot(w[npa_wp[i]].transpose(), t1))
#         else:
#             t4 = 0
#         herir_grad[i] = grad[i] + t4
#     return herir_grad

# def gradiant(w, grad, w_size, n_parent, npa_sibling, self_wp, npa_wp, npas_wp, neta):
#     herir_grad={}
#     for i in grad:
#         if npa_sibling[i]:
#             t4 = np.dot(w[self_wp[i]].transpose(), grad[n_parent[i]])
#         else:
#             t4 = 0
#         herir_grad[i] = grad[i] + t4
#     return herir_grad

def rae_operation(vect, w, w_size, h_vect, h_index, dep_order, Word_ids,hh_index, wp, p, neta, regu, wpresent):
    o = {};
    dw={};
    epnum={};
    for i in wpresent:
        epnum[i] = 0
        dw[i] = 0.0
    rae_encoding(vect, w, w_size, h_vect, wp)
    vect_ = rae_decoding(vect, o, w, w_size, h_vect, wp)
    grad , _= pre_gradiant(vect, vect_, o, h_vect, w, wp, w_size)
    # for i in range(len(h_vect)):
    #     temp1=0.0
    #     for j in range(len(h_vect[i])):
    #         temp1 += np.dot(w[wp[i][j]],vect[h_vect[i][j]])
    #     vect[w_size + i] = sigmoid(temp1)
    #
    #     for j in range(len(h_vect[i])):
    #         o[h_vect[i][j]] = np.dot(w[wp[i][j]].transpose(), vect[w_size + i])
    #         vect_[h_vect[i][j]] = sigmoid(o[h_vect[i][j]])
    #
    #     for j in range(len(h_vect[i])):
    #         cost = np.power((vect[h_vect[i][j]] - vect_[h_vect[i][j]]),2)+ regu * np.sum(w[h_vect[i][j]])
    #         # cost = np.power((vect[h_vect[i][j]] - vect_[h_vect[i][j]]),2)
    #         pre_grad[h_vect[i][j]] = cost * dsigmoid(o[h_vect[i][j]])
    #         epnum[wp[i][j]] += 1
    # n_parent, npa_sibling, self_wp, npa_wp, npas_wp = get_parent_detail(Word_ids=Word_ids, dep_order=dep_order, h_index=h_index, hh_index=hh_index, h_vect=h_vect, wp=wp, p=p)
    # grad = gradiant(w, pre_grad, n_parent=n_parent, npa_sibling= npa_sibling, npa_wp=npa_wp, npas_wp=npas_wp, self_wp=self_wp, neta=neta)

    for i in range(len(h_vect)):
        for j in range(len(h_vect[i])):
            dw[wp[i][j]] += np.dot(grad[h_vect[i][j]], vect_[w_size + i].transpose())
            epnum[wp[i][j]] += 1
    return vect_, dw, epnum



def rae_trainning_normal(wd_extra, neta = None, regu=0.9, w=None, wpresent=None):
    err = 0.0
    kerr = 0.0
    samp_err = []
    branch_samp_err = []
    branchs = 0
    t_dw = {};
    t_epnum = {};
    for i in wpresent:
        t_dw[i] = 0.0
        t_epnum[i] = 0
    for iword in wd_extra:
        try:
            # vect = [i for i in iword['Word_vects']] + [None for i in range(len(iword['h_vect']))]
            vect = [preprocess(i) for i in iword['Word_vects']] + [None for i in range(len(iword['h_vect']))]
            vect_, dw, epnum = rae_operation(vect=vect, w=w, w_size=iword['w_size'], h_vect=iword['h_vect'], Word_ids= iword['Word_ids'], wp=iword['wp'], h_index=iword['h_index'], dep_order = iword['dep_order'], hh_index=iword['hh_index'], p=iword['p'], neta=neta, regu=regu, wpresent=wpresent)
            terr = 0.0
            bc=0
            for iv in range(iword['w_size']):
                terr += np.sum(np.power((vect[iv] - vect_[iv]),2))
                bc+=1
            branchs += bc
            err += terr
            if bc:
                samp_err.append(terr)
                branch_samp_err.append(terr/bc)
        except RuntimeError:
            kerr += 1
            continue
        for wi in wpresent:
            t_dw[wi] += dw[wi]
            t_epnum[wi] += epnum[wi]
    weight_update(w, t_dw, neta=neta, regu=regu, epnum=t_epnum, wpresent=wpresent)
    return w, t_dw, t_epnum, kerr, err, samp_err, branch_samp_err, branchs



## testing the validity of code
if __name__=='__main__':
    import pickle, Global
    from RAE_trainning4 import wd_preprocess
    from utility1 import get_weight_matrices, init_weight
    from Global import get_word_vect
    w1 = {}
    w2 = {}
    w_range = 101
    v_size = 50
    h_size = 30
    neta = 0.0001
    regu = .1
    w = pickle.load(open('/media/zero/41FF48D81730BD9B/Final_Thesies/data/test/masr_test9_0.0005_0.1_201_50_30_100/masr_test9_0.0005_0.1_201_50_30_100_.pickle','rb'))
    test_file = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/MSRParaphraseCorpus/msr_phrase.pickle'
    words_data = pickle.load(open(test_file, 'rb'))[:20]
    wd_extra = []
    Global.init_wv(v_size)
    e = 1e-4
    for iword in range(len(words_data)):

        h_index, h_vect, wp, Word_ids, w_size, dep_order, hh_index, p = wd_preprocess(words_data[iword])
        if h_index == -1:
            continue

        cflag = 0
        # for i in wp:
        #     if len(i) > abs(w_range / 2):
        #         cflag = 1
        #         break
        #     for j in i:
        #         if type(w[j]) != np.ndarray:
        #             w[j] = init_weight(h_size)
        # if cflag == 1:
        #     continue

        Word_vects = []
        for i in sorted(words_data[iword]):
            Word_vects.append(get_word_vect(words_data[iword][i]['word'].lower(), Global.v_size))

        wd_extra.append({'w_size': w_size, 'h_index': h_index, 'h_vect': h_vect, 'Word_vects': Word_vects, 'wp': wp,
                         "dep_order": dep_order, "hh_index": hh_index, 'Word_ids': Word_ids, 'p': p})
    for i in w:
        if type(w[i]) == np.ndarray:
            w1[i] = w[i].copy() + e
            w2[i] = w[i].copy() - e
    epnum = {}
    for i in w:
        epnum[i] = 0
    for iword in wd_extra:
        vect1 = [preprocess(i) for i in iword['Word_vects']] + [None for i in range(len(iword['h_vect']))]
        vect2 = [preprocess(i) for i in iword['Word_vects']] + [None for i in range(len(iword['h_vect']))]
        o1 = {};
        o2 = {};
        vect_1 = {};
        vect_2 = {};

        rae_encoding(vect=vect1, w=w1, w_size= iword['w_size'], h_vect=iword['h_vect'], wp=iword['wp'])
        vect_1 = rae_decoding(vect=vect1, o=o1, w=w1, w_size= iword['w_size'], h_vect=iword['h_vect'], wp=iword['wp'])
        grad, cost1 = pre_gradiant(vect=vect1, vect_=vect_1, o=o1, h_vect=iword['h_vect'], wp=iword['wp'], epnum=epnum)

        rae_encoding(vect=vect2, w=w2, w_size=iword['w_size'], h_vect=iword['h_vect'], wp=iword['wp'])
        vect_2 = rae_decoding(vect=vect2, o=o2, w=w2, w_size=iword['w_size'], h_vect=iword['h_vect'], wp=iword['wp'])
        grad, cost2 = pre_gradiant(vect=vect2, vect_=vect_2, o=o2, h_vect=iword['h_vect'], wp=iword['wp'], epnum=epnum)

        for i in range(len(cost1)):
            print i, (np.array(cost1[i]) - np.array(cost2[i])).transpose()/(2 * e)
        raw_input()

