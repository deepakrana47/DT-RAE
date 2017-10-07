import os, random

def list_dir_files(t_dir):
    f_list = []
    if t_dir[-1] != '/':
        t_dir+='/'
    for dir in os.listdir(t_dir):
        f_list += [t_dir+dir+'/'+i for i in os.listdir(t_dir+dir)]
    return f_list

def list_1perdir_files(t_dir):
    f_list = []
    if t_dir[-1] != '/':
        t_dir += '/'
    for dir in os.listdir(t_dir):
        f_list += [random.choice([t_dir + dir + '/' + i for i in os.listdir(t_dir + dir)])]
    # f_list=['/media/zero/41FF48D81730BD9B/Final_Thesies/data/New_test_set/msr_paraphrase_text.pickle']
    return f_list

# pickle.dump(f_list,open('/media/zero/41FF48D81730BD9B/Final_Thesies/data/wiki/pickle_list.pickle','w'))
if __name__ == '__main__':
    for i in list_1perdir_files('/media/zero/41FF48D81730BD9B/Final_Thesies/data/wiki/pickles'):
        print i
