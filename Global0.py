import pickle
from gensim.models import Word2Vec
import numpy as np

# embed_text='/media/zero/41FF48D81730BD9B/Final_Thesies/Coding/glove_6B/glove.6B.200d.txt'
model_name200 = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/cbow_word2vect_200d/wiki.en.word2vec.model'
model_name50 = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/word2vect_50d/wiki.en.text_50d.model'

# extra_word200 = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/RAE_training_data/extra_word(num)_200d/extra.word2vec.model'
#
# extra_word50 = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/extra_word_50d/extra.word2vec.model'
#
# extra_word1_200 = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/RAE_training_data/raw/extra.pickle'
# extra_word1_50 = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/word2vect_50d/extra.pickle'

# glove_200d = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/Glove_200d/glove.6B.200d.txt'
# def get_glove_vect(text):
#     word_vec = {}
#     for i in open(text):
#         t1 = i.split(' ')
#         word_vec[t1[0]] = np.array([float(j) for j in t1[1:]])
#     return word_vec
# def init_wv(vsize):
#     global word_vec, wfname, model, v_size, word_vec2
#     v_size = vsize
#     word_vec = get_glove_vect(glove_200d)
#     word_vec2 = pickle.load(open(extra_word200, 'rb'))

def init():
    global rejected, accepted, wfname, v_size, fstopwds, fabbwds, r, c
    fstopwds = "/media/zero/41FF48D81730BD9B/Final_Thesies/Coding/final_STS_machine_2/extra/stopwords2.txt"
    fabbwds = "/media/zero/41FF48D81730BD9B/Final_Thesies/Coding/final_STS_machine_2/extra/abbrev1.txt"
    rejected = 0
    accepted = 0
    wfname = None

#
# def init_h12(hi1, hi2):
#     global h1, h2
#     h1=hi1
#     h2=hi2
#
# def _init_wv(vsize=50):
#     global word_vec, word_vec1, wfname, v_size, word_vec2
#     if vsize==200:
#         model = Word2Vec.load(model_name200)
#         emodel = Word2Vec.load(extra_word200)
#         word_vec2=pickle.load(open(extra_word1_200,'rb'))
#         v_size=200
#     else:
#         model=Word2Vec.load(model_name50)
#         emodel=Word2Vec.load(extra_word50)
#         word_vec2=pickle.load(open(extra_word1_50,'rb'))
#         v_size=50
#     word_vec = model.wv
#     word_vec1 = emodel.wv

import numpy as np

def init_real_time():
    global word_vec, word_vec1, wfname, v_size, fstopwds, word_vec2
    word_vec = Word2Vec.load(model_name200).wv
    word_vec1 = pickle.load(open('/media/zero/41FF48D81730BD9B/Final_Thesies/data/RAE_training_data/data_without_stopwrd/words_vect/wiki_word200.vector.pickle','rb'))
    word_vec2=pickle.load(open('/media/zero/41FF48D81730BD9B/Final_Thesies/data/RAE_training_data/raw/extra200.pickle','rb'))
    fstopwds = "/media/zero/41FF48D81730BD9B/Final_Thesies/Coding/final_STS_machine_2/extra/stopwords2.txt"
    v_size=200

# def init_wiki(stopword = None, vsize=200):
#     global word_vec, word_vec2, word_vec1, word_vec2, wfname, v_size, fstopwds, extra_word1_200, extra_word1_50, extra_word1_100
#     v_size = vsize
#     model_file = ''
#     extra_word1_200 = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/cbow_word2vect_200d/extra_word.pickle'
#     extra_word1_50 = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/cbow_word2vect_50d/extra_word.pickle'
#
#     if v_size == 200:
#         model_file = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/cbow_word2vect_200d/wiki.en.word2vec.model'
#     elif v_size == 50:
#         model_file = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/cbow_word2vect_50d/wiki.en.word2vec.model'
#
#     word_vec = Word2Vec.load(model_file).wv
#
#     if stopword == 0:
#         fstopwds = "/media/zero/41FF48D81730BD9B/Final_Thesies/Coding/final_STS_machine_2/extra/stopwords2.txt"
#         if v_size == 200:
#             model_file1 = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/RAE_training_data/data_without_stopwrd/words_vect/wiki_word200.vector.pickle'
#         elif v_size == 50:
#             model_file1 = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/RAE_training_data/data_without_stopwrd/words_vect/wiki_word50.vector.pickle'
#         else:
#             print "Not a valid word vector size"
#     elif stopword == 1:
#         fstopwds = "/media/zero/41FF48D81730BD9B/Final_Thesies/Coding/final_STS_machine_2/extra/stopwords3.txt"
#         if v_size == 200:
#             model_file1 = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/RAE_training_data/data_with_stopwrd/words_vect/wiki_word200.vector.pickle'
#         elif v_size == 50:
#             model_file1 = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/RAE_training_data/data_with_stopwrd/words_vect/wiki_word50.vector.pickle'
#         else:
#             print "Not a valid word vector size"
#     word_vec1 = pickle.load(open(model_file1, 'rb'))
#     if v_size == 200:
#         word_vec2 = pickle.load(open(extra_word1_200, 'rb'))
#     elif v_size == 50:
#         word_vec2 = pickle.load(open(extra_word1_50, 'rb'))

def init_wv_self(stopword=None, vsize=200):
    global word_vec, word_vec2, word_vec1, wfname, v_size, fstopwds, extra_word1_200, extra_word1_50, extra_word1_100
    v_size = vsize
    model_file=''
    extra_word1_200 = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/RAE_training_data/raw/extra200.pickle'
    extra_word1_50 = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/RAE_training_data/raw/extra50.pickle'
    extra_word1_100 = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/RAE_training_data/raw/extra100.pickle'

    if stopword == 0:
        fstopwds = "/media/zero/41FF48D81730BD9B/Final_Thesies/Coding/final_STS_machine_2/extra/stopwords2.txt"
        if v_size == 200:
            model_file = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/RAE_training_data/data_without_stopwrd/words_vect/wiki_word200.vector.pickle'
        elif v_size == 50:
            model_file = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/RAE_training_data/data_without_stopwrd/words_vect/wiki_word50.vector.pickle'
        elif v_size == 100:
            model_file = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/RAE_training_data/raw/words_vect/word100.vector.pickle'
        else:
            print "Not a valid word vector size"
    elif stopword == 1:
        fstopwds = "/media/zero/41FF48D81730BD9B/Final_Thesies/Coding/final_STS_machine_2/extra/stopwords3.txt"
        if v_size == 200:
            model_file = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/RAE_training_data/data_with_stopwrd/words_vect/wiki_word200.vector.pickle'
        elif v_size == 50:
            model_file = '/media/zero/41FF48D81730BD9B/Final_Thesies/data/RAE_training_data/data_with_stopwrd/words_vect/wiki_word50.vector.pickle'
        else:
            print "Not a valid word vector size"
    word_vec = pickle.load(open(model_file, 'rb'))
    if v_size == 200:
        word_vec1 = pickle.load(open(extra_word1_200, 'rb'))
    elif v_size == 50:
        word_vec1 = pickle.load(open(extra_word1_50, 'rb'))
    if v_size == 100:
        word_vec1 = pickle.load(open(extra_word1_100, 'rb'))


def generate_word_vect(v_size):
    return np.random.normal(1,0.5,(v_size,1))


def add_word(vsize,extraword):
    global extra_word1_200, extra_word1_50
    if vsize == 200:
        pickle.dump(extraword, open(extra_word1_200, 'wb'))
    elif vsize == 50:
        pickle.dump(extraword, open(extra_word1_50, 'wb'))
    elif vsize == 100:
        pickle.dump(extraword, open(extra_word1_100, 'wb'))

# def get_word_vect(word, v_size=None):
#     if word in word_vec:
#         return word_vec[word].reshape((v_size,1))
#     # elif word in word_vec1:
#     #     return word_vec1[word].reshape((v_size,1))
#     elif word in word_vec2:
#         return word_vec2[word]
#     else:
#         try:
#             int(word)
#             return word_vec['number'].reshape((v_size,1))
#         except ValueError:
#             try:
#                 float(word)
#                 return word_vec['number'].reshape((v_size,1))
#             except ValueError:
#                 print word,
#                 a=generate_word_vect(v_size)
#                 word_vec2[word] = a
#                 add_word(v_size, word_vec2)
#                 return a

def get_word_vect(word, vsize=None):
    global word_vec1, word_vec
    if word in word_vec:
        return word_vec[word].reshape((vsize,1))
    elif word in word_vec1:
        return word_vec1[word]
    elif word in word_vec2:
        return word_vec2[word]
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
                word_vec2[word] = a
                add_word(vsize, word_vec2)
                return a

def adam_parm_init():
    global lr, m, v, lr2, m2, v2
    lr = 0.0001
    lr2 = 0.00001
    m = {}
    v = {}
    m2=0.0
    v2=0.0