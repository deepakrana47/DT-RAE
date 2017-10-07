import pickle

import numpy as np

from Global import get_word_vect
import RAE_adam_herical
import RAE_adam_herical_deep1
from utility1 import extract_feature_using_senna,get_words_id,get_parents,get_dep,pdep_2_deporder_dep,dep_2_hid_var,get_words_vect,preprocess,get_chunk_main,get_chunks,get_chunks_by_dep,get_chunks_by_dep1, get_order
from text_process1 import line_processing


def remove_extra(wds):
    t1=[]
    for wd in wds:
        if type(wd) == float:
            t1.append(wd)
    for i in t1:
        wds.pop(i)
def get_chk_vect(flag, line, mtype='normal'):
    if flag == 't':
        line = line_processing(line)
        # print line
        words_data= extract_feature_using_senna(line)
        p=get_parents(words_data)
        d=get_dep(words_data)
        Word_ids = get_words_id(words_data)
        for i in Word_ids:
            words_data[i]['vect']=preprocess(get_word_vect(words_data[i]['word'].lower(), Global.v_size))
        w_size = len(Word_ids)
        dep_order, d1 = pdep_2_deporder_dep(p, d)
        h_index, h_vect, wp ,_ = dep_2_hid_var(p, dep_order, d1, Word_ids)
        Word_vects = get_words_vect(words_data, Word_ids, Global.v_size)
        vect = Word_vects + [None for i in h_vect]
        del Word_vects
        w = pickle.load(open(Global.wfname, 'rb'))
        if mtype == 'normal':
            RAE_adam_herical.rae_encoding(vect=vect, w=w, w_size=w_size, h_vect=h_vect, wp=wp)
        elif mtype == 'deep':
            RAE_adam_herical_deep1.rae_encoding(vect=vect, w=w, w_size=w_size, h_vect=h_vect, wp=wp)
        chks = get_chunks_by_dep(Word_ids, h_index, h_vect)
        chunks = {}
        chunks_vect = {}
        count = 0
        order = get_order(d1, w_size)
        for m in order:
            chunks[count] = ' '.join([words_data[i]['word'] for i in chks[m]])
            chunks_vect[count] = vect[h_index[m]]
            count += 1
        return chunks, chunks_vect

def get_vect_by_wd_dep(flag, words_data, mtype='normal'):
    if flag == 't':
        Word_ids = get_words_id(words_data)
        Word_vects = []
        for i in sorted(Word_ids):
            Word_vects.append(preprocess(get_word_vect(words_data[i]['word'].lower(), Global.v_size)))
        w_size = len(Word_ids)
        p = get_parents(words_data)
        d = get_dep(words_data)
        dep_order, d1 = pdep_2_deporder_dep(p, d)
        h_index, h_vect, wp ,_ = dep_2_hid_var(p, dep_order, d1, Word_ids)
        # Word_vects = get_words_vect(words_data, Word_ids, Global.v_size)
        vect = Word_vects + [None for i in range(len(h_vect))]
        del Word_vects
        w = pickle.load(open(Global.wfname, 'rb'))
        if mtype == 'normal':
            RAE_adam_herical.rae_encoding(vect=vect, w=w, w_size=w_size, h_vect=h_vect, wp=wp)
        elif mtype == 'deep':
            RAE_adam_herical_deep1.rae_encoding(vect=vect, w=w, w_size=w_size, h_vect=h_vect, wp=wp)
        chunks = {}
        chunks_vect = {}
        for i in range(len(Word_ids)):
            chunks[i] = words_data[Word_ids[i]]['word']
            chunks_vect[i] = vect[i]
        rev_h_index = {v: k for k, v in h_index.items()}
        count = i + 1
        for i in h_vect:
            if len(i) > 1:
                chunks_vect[count] = vect[h_index[Word_ids[min(i)]]]
                chunks[count] = ' '.join([words_data[rev_h_index[j]]['word'] if j >= len(Word_ids) else words_data[Word_ids[j]]['word'] for j in i])
                count+=1
        return chunks, chunks_vect


        # chks {chk_main:[chk], ...}
        # chks = get_chunks_by_dep(Word_ids, h_index, h_vect)
        # chks = get_chunks_by_dep1(Word_ids, h_index, h_vect)
        #

        # count = 0
        # order = get_order(d1, w_size)
        # for m in order:
        #     chunks[count] = ' '.join([words_data[i]['word'] for i in chks[m]])
        #     chunks_vect[count] = vect[h_index[m]]
        #     count+=1
        # return chunks, chunks_vect
        # chks = get_chunks_by_dep1(Word_ids, h_index, h_vect)
        # for m in range(len(chks)):
        #     chunks[m] = ' '.join([words_data[i]['word'] for i in chks[m]])
        #     if len(chks[m]) == 1:
        #         chunks_vect[m] = vect[m]
        #     else:
        #         chunks_vect[m] = vect[h_index[Word_ids[min(chk[m])]]]
        # return chunks, chunks_vect

def get_chk_vect_by_wd(flag, words_data, mtype='normal'):
    if flag == 't':
        Word_ids = get_words_id(words_data)
        Word_vects = []
        for i in sorted(Word_ids):
            Word_vects.append(preprocess(get_word_vect(words_data[i]['word'].lower(), Global.v_size)))
        w_size = len(Word_ids)
        p=get_parents(words_data)
        d=get_dep(words_data)
        dep_order, d1 = pdep_2_deporder_dep(p, d)
        h_index, h_vect, wp ,_ = dep_2_hid_var(p, dep_order, d1, Word_ids)
        # Word_vects = get_words_vect(words_data, Word_ids, Global.v_size)
        vect = Word_vects + [None for i in range(len(h_vect))]
        del Word_vects
        w = pickle.load(open(Global.wfname, 'rb'))
        if mtype == 'normal':
            RAE_adam_herical.rae_encoding(vect=vect, w=w, w_size=w_size, h_vect=h_vect, wp=wp)
        elif mtype == 'deep':
            RAE_adam_herical_deep1.rae_encoding(vect=vect, w=w, w_size=w_size, h_vect=h_vect, wp=wp)
        chks = get_chunks(words_data)
        chks_main = get_chunk_main(chks, dep_order)
        chunks = {}
        chunks_vect = {}
        for c in range(len(chks)):
            chunks[c] = ' '.join([words_data[i]['word'] for i in chks[c]])
        del words_data
        for c in range(len(chks_main)):
            ind = h_index[chks_main[c]]
            chunks_vect[c] = vect[ind]
        return chunks, chunks_vect

# def get_all_vect(flag, var):
#     if flag == 't':
#         line = line_processing(var)
#         words_data = extract_feature_using_senna(line)
#         Word_ids = get_words_id(words_data)
#         for i in Word_ids:
#             words_data[i]['vect']=get_word_vect(words_data[i]['word'].lower(), Global.v_size)
#         w_size = len(Word_ids)
#         p = get_parents(words_data)
#         d = get_dep(words_data)
#         # remove_extra(words_data)
#         from dep_correction import depth
#         t1 = [id for id in d]
#         req_ = []
#         depth(t1, 0, req_)
#         for r in req_:
#             d.remove(r)
#         dep_order, d1 = pdep_2_deporder_dep(p, d)
#         h_index, h_vect, wp ,_ = dep_2_hid_var(p, dep_order, d1, Word_ids)
#         Word_vects = get_words_vect(words_data, Word_ids)
#         vect = [np.array([preprocess(i)]).transpose() for i in Word_vects] + [None for i in range(len(h_vect))]
#         del Word_vects
#         w = pickle.load(open(Global.wfname, 'rb'))
#         rae_encoding(vect=vect, w=w, w_size=w_size, h_vect=h_vect, wp=wp)
#         chunks = {}; chunks_vect = {}
#         svect = [i for i in Word_ids if i not in dep_order]
#         mvect = [i for i in dep_order]
#         iv=0
#         for iv in range(len(svect)):
#             chunks[iv] = words_data[svect[iv]]['word']
#             chunks_vect[iv] = vect[h_index[svect[iv]]]
#         iv+=1
#         for imv in range(len(mvect)):
#             temp = ''
#             for j in sorted(d1[mvect[imv]]+[mvect[imv]]):
#                 temp += words_data[j]['word']+' '
#             chunks[iv + imv] = temp
#             chunks_vect[iv+imv] = vect[h_index[mvect[imv]]]
#     return chunks, chunks_vect

# def get_sent_vect(sent):
#     line = line_processing(sent)
#     words_data = extract_feature_using_senna(line)
#     Word_ids = get_words_id(words_data)
#     for i in Word_ids:
#         try:
#             words_data[i]['vect'] = get_word_vect(words_data[i]['word'].lower(), Global.v_size)
#         except:
#             words_data[i]['vect'] = np.zeros((50))
#     w_size = len(Word_ids)
#     p = get_parents(words_data)
#     d = get_dep(words_data)
#     # remove_extra(words_data)
#     from dep_correction import depth
#     t1 = [id for id in d]
#     req_ = []
#     depth(t1, 0, req_)
#     for r in req_:
#         d.remove(r)
#     dep_order, d1 = pdep_2_deporder_dep(p, d)
#     h_index, h_vect, wp ,_ = dep_2_hid_var(p, dep_order, d1, Word_ids)
#     Word_vects = get_words_vect(words_data, Word_ids)
#     vect = [np.array([preprocess(i)]).transpose() for i in Word_vects] + [None for i in range(len(h_vect))]
#     del Word_vects
#     w = pickle.load(open(Global.wfname, 'rb'))
#     rae_encoding(vect=vect, w=w, w_size=w_size, h_vect=h_vect, wp=wp)
#     return vect[-1]
#
# def get_sent_data(sent):
#     line = line_processing(sent)
#     words_data = extract_feature_using_senna(line)
#     if not words_data:
#         return None,None
#     Word_ids = get_words_id(words_data)
#     for i in Word_ids:
#         try:
#             words_data[i]['vect'] = get_word_vect(words_data[i]['word'].lower(), Global.v_size)
#         except:
#             words_data[i]['vect'] = np.zeros((50))
#     w_size = len(Word_ids)
#     p = get_parents(words_data)
#     d = get_dep(words_data)
#     # remove_extra(words_data)
#     from dep_correction import depth
#     t1 = [id for id in d]
#     req_ = []
#     depth(t1, 0, req_)
#     for r in req_:
#         d.remove(r)
#     dep_order, d1 = pdep_2_deporder_dep(p, d)
#     h_index, h_vect, wp ,_ = dep_2_hid_var(p, dep_order, d1, Word_ids)
#     Word_vects = get_words_vect(words_data, Word_ids)
#     vect = [np.array([preprocess(i)]).transpose() for i in Word_vects] + [None for i in range(len(h_vect))]
#     del Word_vects
#     w = pickle.load(open(Global.wfname, 'rb'))
#     rae_encoding(vect=vect, w=w, w_size=w_size, h_vect=h_vect, wp=wp)
#     words_extra = {'ihvect':h_vect, 'hindex':h_index, 'dep': d1,'dorder':dep_order, 'vect':vect}
#     return words_data, words_extra

import Global
if __name__=='__main__':
    Global.init()
    Global.init_wv(50)
    Global.wfname = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/wiki/weights/wiki_weight_0.0005_0.7_201_50_40_1000_/15wiki_weight_0.0005_0.7_201_50_40_1000_.pickle'
    wd=extract_feature_using_senna('sub indexes measuring prices new orders inventories exports increased')
    chk, chk_vect = get_vect_by_wd_dep('t',wd)
    for i in chk_vect:
        print i, chk_vect[i].shape, chk[i]