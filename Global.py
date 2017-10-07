import pickle

def init():
    global rejected, accepted, wfname, v_size, fstopwds, fabbwds, r, c
    fstopwds = "./stopwords2.txt"
    fabbwds = "./abbrev1.txt"
    rejected = 0
    accepted = 0
    wfname = None

import numpy as np

def init_wv_self(stopword=None, vsize=200):
    global word_vec, word_vec2, word_vec1, wfname, v_size, fstopwds, extra_word1_200, extra_word1_50, extra_word1_100
    v_size = vsize
    extra_word1_200 = './word_embeddings/extra200.pickle'
    model_file = './word_embeddings/wiki_word200.vector.pickle'

    if stopword == 0:
        fstopwds = "./stopwords2.txt"
    else:
        fstopwds = "./stopwords3.txt"

    word_vec = pickle.load(open(model_file, 'rb'))
    word_vec1 = pickle.load(open(extra_word1_200, 'rb'))

def generate_word_vect(v_size):
    return np.random.normal(0.0,0.01,(v_size,1))

def add_word(vsize,extraword):
    global extra_word1_200, extra_word1_50
    pickle.dump(extraword, open(extra_word1_200, 'wb'))

def get_word_vect(word, vsize=None):
    global word_vec1, word_vec
    if word in word_vec:
        return word_vec[word].reshape((vsize,1))
    elif word in word_vec1:
        return word_vec1[word]
    else:
        try:
            int(word)
            return word_vec['number'].reshape((vsize,1))
        except ValueError:
            try:
                float(word)
                return word_vec['number'].reshape((vsize,1))
            except ValueError:
                print word,
                a=generate_word_vect(vsize)
                word_vec1[word] = a
                add_word(vsize, word_vec1)
                return a

def adam_parm_init():
    global lr, m, v, lr2, m2, v2
    lr = 0.0001
    lr2 = 0.00001
    m = {}
    v = {}
    m2=0.0
    v2=0.0