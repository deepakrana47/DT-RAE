import numpy as np
from utility1 import sigmoid,dsigmoid,preprocess, get_parent_detail, get_child_no

def weight_update(w1, w2, dw1, dw2, neta, regu, epnum1, epnum2, wpresent=None):
    for i in wpresent:
        if epnum1[i] != 0:
            w1[i] = w1[i] - neta[0]*(dw1[i].transpose()/epnum1[i] - regu[0]*(w1[i]))
    if epnum2 != 0:
        w2 = w2 - neta[1]*(dw2.transpose()/epnum2 - regu[1]*(w2))
    return w1, w2

def rae_encoding(vect, w, w_size, h_vect, wp):
    vect21={}
    vect22={}
    o21={}
    w1=w['w1']
    w2=w['w2']
    for i in range(len(h_vect)):
        temp1=0.0
        # print w_size + i
        for j in range(len(h_vect[i])):
            if h_vect[i][j] < w_size:
                # print '\tv->h1\t',str(h_vect[i][j])+'->'+str(w_size+i), wp[i][j]
                temp1 += np.dot(w1[wp[i][j]],vect[h_vect[i][j]])
            else:
                # print '\th2->h1\t',str(h_vect[i][j]),'w2`'
                temp11 = np.dot(w2.transpose(),vect[h_vect[i][j]])
                o21[h_vect[i][j]] = temp11
                temp12 = sigmoid(temp11)
                vect22[h_vect[i][j]] = temp12
                # print '\th1->h1\t', str(h_vect[i][j]) + '->' + str(w_size + i), wp[i][j]
                temp1 += np.dot(w1[wp[i][j]],temp12)
        temp12 = sigmoid(temp1)
        vect21[w_size+i] = temp12
        temp2 = np.dot(w2,temp12)
        # print '\th1->h2\t','w2'
        vect[w_size+i] = sigmoid(temp2)
        # print
    return vect21, vect22, o21

def rae_decoding(vect1, w1, w2, w_size, h_vect, wp):
    vect1_ = {}
    vect21_ = {}
    vect22_ = {}
    o_ = {2 * w_size - 1:np.nan}
    o21_ = {}
    vect1_[2 * w_size - 1] = vect1[2 * w_size - 1]
    for i in reversed(range(len(h_vect))):
        to2 = np.dot(w2.transpose(),vect1_[w_size + i])
        # print w_size + i
        o21_[w_size+i] = to2
        # print '\th2->h1\t','w2`'
        t2 = sigmoid(to2)
        vect22_[w_size+i] = t2
        for j in range(len(h_vect[i])):
            if h_vect[i][j] < w_size:
                to1 = np.dot(w1[wp[i][j]].transpose(),t2)
                o_[h_vect[i][j]]=to1
                # print '\th1->v\t',str(w_size+i)+'->'+str(h_vect[i][j]),wp[i][j]
                vect1_[h_vect[i][j]] = sigmoid(to1)
            else:
                to1=np.dot(w1[wp[i][j]].transpose(), t2)
                o_[h_vect[i][j]] = to1
                # print '\th1->h1\t',str(w_size+i)+'->'+str( h_vect[i][j]),wp[i][j]
                t1 = sigmoid(to1)
                vect21_[h_vect[i][j]] = t1
                to2 = np.dot(w2, t1)
                # print '\th1->h2\t', h_vect[i][j],'w2'
                vect1_[h_vect[i][j]] = sigmoid(to2)
    return vect1_, o_, vect21_, vect22_, o21_

# def gradiant1(vect1, vect1_, o_, vect21_, vect22, h_vect,w_size):
#     grad1={}
#     cost = {}
#     for i in range(len(h_vect)):
#         for j in range(len(h_vect[i])):
#             if h_vect[i][j] < w_size:
#                 cost[h_vect[i][j]] = np.power(vect1-vect1_,2)
#                 grad1[h_vect[i][j]] = cost[h_vect[i][j]]*o_[h_vect[i][j]]
#             else:
#                 cost[h_vect[i][j]] = np.power(vect22 - vect21_,2)
#                 grad1[h_vect[i][j]] = cost[h_vect[i][j]] * o_[h_vect[i][j]]
#     return grad1, cost

def gradiant1(vect1, vect1_, o_, h_vect, w1, wp, w_size):
    grad1={}
    cost1={}
    for i in range(len(h_vect)):
        # print w_size+i
        grad1[w_size + i] = 0.0
        for j in range(len(h_vect[i])):
            if h_vect[i][j] < w_size:
                # print '\tv\t', h_vect[i][j]
                tcost = np.power((vect1[h_vect[i][j]] - vect1_[h_vect[i][j]]), 2)
                cost1[h_vect[i][j]]=tcost
                grad1[h_vect[i][j]] = tcost * dsigmoid(o_[h_vect[i][j]])
                # print '\th1\t',h_vect[i][j], wp[i][j], w_size+i
                grad1[w_size + i] += np.dot(w1[wp[i][j]], grad1[h_vect[i][j]]) * dsigmoid(o_[w_size+i])
            else:
                # print '\th1\t', h_vect[i][j],wp[i][j], w_size + i
                grad1[w_size + i] += np.dot(w1[wp[i][j]], grad1[h_vect[i][j]]) * dsigmoid(o_[h_vect[i][j]])
        # print
    return grad1, cost1

def gradiant2(vect21, vect22, o21, vect21_, vect22_, o21_, w_size, vect1_):
    grad2 = []
    cost2 = []
    v=[]
    vect21_[w_size*2-1]=vect21[w_size*2-1]
    vect22[w_size * 2 - 1] = vect22_[w_size * 2 - 1]
    o21[w_size*2-1] = o21_[w_size*2-1]
    # print "encoding grad"
    for i in vect22:
        # print '\t',i
        tcost = np.power(vect21[i] - vect22[i],2)
        cost2.append(tcost)
        grad2.append(tcost*sigmoid(o21[i]))
        v.append(vect1_[i])
    # print "decoding grad"
    for i in vect22_:
        # print '\t',i
        tcost = np.power(vect21_[i] - vect22_[i],2)
        cost2.append(tcost)
        grad2.append(tcost * sigmoid(o21_[i]))
        v.append(vect1_[i])
    grad2=grad2[:-1]
    cost2=cost2[:-1]
    return grad2, cost2, v


def rae_operation(vect1, w1, w2, w_size, h_vect, h_index, dep_order, Word_ids,hh_index, wp, p, regu, wpresent):
    epnum1 = {}
    dw1 = {}
    for i in wpresent:
        epnum1[i] = 0
        dw1[i] = 0.0
    epnum2=0;

    vect21, vect22, o21=rae_encoding(vect1,{'w1':w1,'w2':w2}, w_size, h_vect, wp)
    vect1_, o_, vect21_, vect22_, o21_=rae_decoding(vect1, w1, w2, w_size, h_vect, wp)
    grad1, _ = gradiant1(vect1,vect1_, o_, h_vect, w1, wp, w_size)
    grad2, _, v = gradiant2(vect21, vect22, o21, vect21_, vect22_, o21_, w_size, vect1_)
    dw2 = 0.0
    # ch_no = get_child_no(h_vect, w_size)
    for i in range(len(h_vect)):
        # ch_sum = sum(ch_no[i])
        for j in range(len(h_vect[i])):
            dw1[wp[i][j]] += np.dot(grad1[h_vect[i][j]], vect21_[w_size + i].transpose())#*ch_no[i][j]/ch_sum
            epnum1[wp[i][j]]+=1
    for i in range(len(grad2)):
        dw2 += np.dot(grad2[i], v[i].transpose())
        epnum2+=1
    return vect1_, dw1, dw2, epnum1, epnum2

def rae_trainning_normal(wd_extra, w1, w2, neta, regu=None, wpresent=None):
    err = 0.0
    kerr = 0.0
    samp_err = []
    branch_samp_err = []
    branchs = 0
    t_dw1 = {}
    t_dw2 = 0.0
    t_epnum1 = {}
    for i in wpresent:
        t_epnum1[i] = 0
        t_dw1[i] = 0.0
    t_epnum2 = 0.0
    for iword in wd_extra:
        try:
            vect = [preprocess(i) for i in iword['Word_vects']] + [None for i in range(len(iword['h_vect']))]
            vect_, dw1, dw2, epnum1, epnum2 = rae_operation(vect1=vect, w1=w1, w2=w2, w_size=iword['w_size'], h_vect=iword['h_vect'], Word_ids= iword['Word_ids'], wp=iword['wp'], h_index=iword['h_index'], dep_order = iword['dep_order'], hh_index=iword['hh_index'], p=iword['p'], regu = regu, wpresent=wpresent)
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
            t_dw2 += dw2
            t_epnum2 += epnum2
        except KeyError:
            kerr += 1
            continue
        for wi in wpresent:
            t_dw1[wi] += dw1[wi]
            t_epnum1[wi] += epnum1[wi]
    w1, w2 = weight_update(w1=w1, w2=w2, dw1 = t_dw1, dw2=t_dw2, neta=neta, regu=regu, epnum1=t_epnum1, epnum2=t_epnum2, wpresent=wpresent)
    return w1, w2, None, None, kerr, err, samp_err, branch_samp_err, branchs