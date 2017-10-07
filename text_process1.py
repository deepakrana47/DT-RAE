# Contraction replacement
#   For example, "wouldn't" is replaced with "would not"
# URLs, email addresses, parenthetical expressions removal
#   All the URLs, email addresses, and parenthetical expressions (in the cases for abbreviation definition) are removed
# Abbreviation normalization
#   WordNet 2 , (for example, "USA" is replaced with "United States of America")
#   for example, "govt" is replaced with "government", "1.6lb" and "5m" are replaced with "1.6 pound" and "5 million"
# Negation replacement
#   such as 'not present' with 'absent' using WordNet antonyms
# Misspelling correction
#   correct misspelled words using an English dictionary provided by a dictionary module pyenchant 3 in Python
# Stopwords	removal
#   Stopwords are removed for datasets headlines, plagiarism and postediting, according to the Stanford stopwords list
#   For the datasets of answer-answer and question-question,
#       stopwords are not removed since many pairs only contain stopwords,
#       such as sentence pair "Can you do this?" "You can do it, too", etc.
# Lemmatization
#   the remaining words in each sentence are lemmatized to their base forms
#   using the lemmatizer provided by natural language toolkit (NLTK) in Python

import re, Global
# from nltk.stem.wordnet import WordNetLemmatizer

# lmtzr = WordNetLemmatizer()
# abbwds = dict((i.strip('\n').split(', ') for i in open(abb_file,'r')))
# stopwds = (i.strip('\n') for i in open(stopwds_file,'r'))

def division(match):
    return str(float(match.group(1))/float(match.group(2)))

def _line_processing(line, fstopwds= None):
    '''
    Line pre-processing is performed
    :param line: line input
    :return:
        ptext: processed text
    '''
    fstopwds =  Global.fstopwds if fstopwds is None else fstopwds
    # fstopwds = "/media/zero/41FF48D81730BD9B/Final_Thesies/Coding/final_STS_machine_2/extra/stopwords2.txt"
    fabbwds = "/media/zero/41FF48D81730BD9B/Final_Thesies/Coding/final_STS_machine_2/extra/abbrev1.txt"
    ptext = ''

    # stop word removal and abbreviation formatting
    ptext = line.strip('\n|\r| ')
    ptext = ptext.strip('.')
    ptext = re.sub(r'(\.{2,}|\.[ ]*"|\.[ ]*\')$','',ptext)
    ptext = re.sub(r'(\.{2,}|\."|\.\')',' ',ptext)
    ptext = ptext.lower()

    # pre symbol removel
    ptext=re.sub(r'\-|,"|\;|\'s |\(|\)|\[|\]|\{|\}',r' ',ptext)
    ptext=re.sub(r'\xe2\x80\x99s|\xc2\xb4s|\'|"|,|\~|\+',r'',ptext)
    ptext = re.sub(r'( |^)([0-9]+)(st|th|ed|s)( |$)',r'\1\2\4',ptext)

    # wor processing
    ptext = re.sub(r'( |^)\<[^\>]+\>([ ]*)',r' ',ptext)
    # ptext = re.sub(r'\([^\)]+\)',r' ',ptext)

    ptext = re.sub(r'(\$|a\$|\$a|\$us|us\$)([0-9]+[\.]{0,1}[0-9]*)bn', r' \2 billion dollar ', ptext)
    ptext = re.sub(r'(\$|a\$|\$a|\$us|us\$)([0-9]+[\.]{0,1}[0-9]*)m', r' \2 million dollar ', ptext)
    ptext = re.sub(r'(\$|a\$|\$a|\$us|us\$)([0-9]+[\.]{0,1}[0-9]*)', r' \2 dollar ', ptext)

    ptext = re.sub(r'(\xc2\xa3|\xe2\x82\xac)([0-9]+[\.]{0,1}[0-9]*)bn', r' \2 billion pound ', ptext)
    ptext = re.sub(r'(\xc2\xa3|\xe2\x82\xac)([0-9]+[\.]{0,1}[0-9]*)m', r' \2 million pound ', ptext)
    ptext = re.sub(r'(\xc2\xa3|\xe2\x82\xac)([0-9]+[\.]{0,1}[0-9]*)', r' \2 pound ', ptext)

    # ptext = re.sub(r'( |^)([0-9]+)-([0-9]+)/([0-9]+)p( |$)',r'\1\2 \3 \4 ',ptext)
    ptext = re.sub(r'( |^)([0-9]+[\.]{0,1}[0-9]*)%( |$)', r'\1\2 percent\3', ptext)

    ptext = re.sub(r'( |^)([0-9]+[\.]{0,1}[0-9]*)(km/h|kmph|km/hr|kmphr)', r'\1\2 kilometer per hour', ptext)

    ptext = re.sub(r'https\://|http\://', r'', ptext)
    ptext = re.sub(r'( |^)([a-z]{2,})\.([^\. ]{2,})[ ]*(\.[a-z]{2,})', r'\1\3\4', ptext)


    # ptext = re.sub(r'( |^)v([0-9]+[\.]{0,1}[0-9]*) ',r'\1v \2 ',ptext)

    ptext = re.sub(r'( |^|\()([0-9]+)([a-z]+)( |$|\))', r'\1\2 \3\4', ptext)
    ptext = re.sub(r'( |^)([0-9]+\.[0-9]*)([a-z]+)( |$)', r'\1\2 \3\4', ptext)

    # ptext = re.sub(r' (st|th|ed) ', r' ', ptext)


    # post symbol removel
    ptext = re.sub(r'[^\x00-\x7F]+', r' ', ptext)
    regexp1 = re.compile(r'\!\!|\?\!|\?\?|\!\?|`|``|\=|\'\'|\-lrb\-|\-rrb\-|\-lsb\-|\-rsb\-|\'|\:|\?|\<|\>|\%|\$|\@|\!|\^|\#|\*|/|_')
    ptext = re.sub(regexp1, r' ', ptext)
    # ptext=re.sub('([a-z]+\.) ',r'\1 ',ptext)
    # ptext = re.sub(r'( |^)([0-9]+[\.]{0,1}[0-9]*)([a-z]+)( |$)', r'\1\2 \3\4', ptext)


    ## stop word and abbriavation processing
    abbwds = dict([i.strip('\n').split(',') for i in open(fabbwds, 'r')])
    ptext = ' '.join([abbwds[i.lower()] if i.lower() in abbwds else i for i in ptext.split(' ')])
    stopwds = [i.strip('\n') for i in open(fstopwds, 'r')]
    ptext = ' '.join(['' if i in stopwds else i for i in ptext.split(' ')])


    ptext = re.sub('[ \t]{2,}',r' ',ptext)
    ptext = ptext.strip(' ')
    return ptext


def line_processing(line, fstopwds= None, fabbwds=None):
    '''
    Line pre-processing is performed
    :param line: line input
    :return:
        ptext: processed text
    '''
    fstopwds =  Global.fstopwds if fstopwds is None else fstopwds
    fabbwds = "/media/zero/41FF48D81730BD9B/Final_Thesies/Coding/final_STS_machine_2/extra/abbrev1.txt"
    # fstopwds = "/media/zero/41FF48D81730BD9B/Final_Thesies/Coding/final_STS_machine_2/extra/stopwords2.txt"
    # fabbwds = "/media/zero/41FF48D81730BD9B/Final_Thesies/Coding/final_STS_machine_2/extra/abbrev1.txt"
    ptext = ''

    # stop word removal and abbreviation formatting
    ptext = line.strip('\n|\r| ')
    ptext = ptext.strip('.')
    ptext = re.sub(r'(\.{2,}|\.[ ]*"|\.[ ]*\')$', '', ptext)
    ptext = re.sub(r'(\.{2,}|\."|\.\')', ' ', ptext)
    ptext = ptext.lower()

    # pre symbol removel
    ptext = re.sub(r'\-|,"|\;|\'s |\(|\)|\[|\]|\{|\}', r' ', ptext)
    ptext = re.sub(r'\xe2\x80\x99s|\xc2\xb4s|\'|"|,|\~|\+|\&', r'', ptext)
    ptext = re.sub(r'( |^)([0-9]+)(st|th|ed|s)( |$)', r'\1\2\4', ptext)

    # wor processing
    ptext = re.sub(r'( |^)\<[^\>]+\>([ ]*)', r' ', ptext)
    # ptext = re.sub(r'\([^\)]+\)',r' ',ptext)

    ptext = re.sub(r'(\$|a\$|\$a|\$us|us\$)([0-9]+[\.]{0,1}[0-9]*)bn', r' \2 billion dollar ', ptext)
    ptext = re.sub(r'(\$|a\$|\$a|\$us|us\$)([0-9]+[\.]{0,1}[0-9]*)m', r' \2 million dollar ', ptext)
    ptext = re.sub(r'(\$|a\$|\$a|\$us|us\$)([0-9]+[\.]{0,1}[0-9]*)', r' \2 dollar ', ptext)

    ptext = re.sub(r'(\xc2\xa3|\xe2\x82\xac)([0-9]+[\.]{0,1}[0-9]*)bn', r' \2 billion pound ', ptext)
    ptext = re.sub(r'(\xc2\xa3|\xe2\x82\xac)([0-9]+[\.]{0,1}[0-9]*)m', r' \2 million pound ', ptext)
    ptext = re.sub(r'(\xc2\xa3|\xe2\x82\xac)([0-9]+[\.]{0,1}[0-9]*)', r' \2 pound ', ptext)

    # ptext = re.sub(r'( |^)([0-9]+)-([0-9]+)/([0-9]+)p( |$)',r'\1\2 \3 \4 ',ptext)
    ptext = re.sub(r'( |^)([0-9]+[\.]{0,1}[0-9]*)%( |$)', r'\1\2 percent\3', ptext)

    ptext = re.sub(r'( |^)([0-9]+[\.]{0,1}[0-9]*)(km/h|kmph|km/hr|kmphr)', r'\1\2 kilometer per hour', ptext)

    ptext = re.sub(r'https\://|http\://', r'', ptext)
    ptext = re.sub(r'( |^)([a-z]{2,})\.([^\. ]{2,})[ ]*\.([a-z]{2,})', r'\1\3 \4', ptext)
    ptext = re.sub(r'( |^)([^\. ]{2,})\.([a-z]+)', r'\1\2 \3', ptext)

    # ptext = re.sub(r'( |^)v([0-9]+[\.]{0,1}[0-9]*) ',r'\1v \2 ',ptext)

    ptext = re.sub(r'( |^|\()([0-9]+)([a-z]+)( |$|\))', r'\1\2 \3\4', ptext)
    ptext = re.sub(r'( |^)([0-9]+\.[0-9]*)([a-z]+)( |$)', r'\1\2 \3\4', ptext)

    # ptext = re.sub(r' (st|th|ed) ', r' ', ptext)


    # post symbol removel
    ptext = re.sub(r'[^\x00-\x7F]+', r' ', ptext)
    regexp1 = re.compile(
        r'\!\!|\?\!|\?\?|\!\?|`|``|\=|\'\'|\-lrb\-|\-rrb\-|\-lsb\-|\-rsb\-|\'|\:|\?|\<|\>|\%|\$|\@|\!|\^|\#|\*|/|_')
    ptext = re.sub(regexp1, r' ', ptext)
    ptext = re.sub(r'( |^)\.([^\.]+)', r'\1\2', ptext)
    ptext = re.sub(r'( |^)([^\. ])\.([^\. ])\.([^\. ])\.( |$)', r'\1\2\3\4\5', ptext)
    ptext = re.sub(r'( |^)([^\. ])\.([^\. ])\.([^\. ])\.([^\. ])\.( |$)', r'\1\2\3\4\5\6', ptext)
    # ptext=re.sub('([a-z]+\.) ',r'\1 ',ptext)
    # ptext = re.sub(r'( |^)([0-9]+[\.]{0,1}[0-9]*)([a-z]+)( |$)', r'\1\2 \3\4', ptext)


    ## stop word and abbriavation processing
    abbwds = dict([i.strip('\n').split(',') for i in open(fabbwds, 'r')])
    ptext = ' '.join([abbwds[i.lower()] if i.lower() in abbwds else i for i in ptext.split(' ')])
    stopwds = [i.strip('\n') for i in open(fstopwds, 'r')]
    ptext = ' '.join(['' if i in stopwds else i for i in ptext.split(' ')])

    ptext = re.sub('[ \t]{2,}', r' ', ptext)
    ptext = ptext.strip(' ')
    return ptext

def text_preprocessing(text):
    '''
        Text pre-processing is performed
        :param text: line data
        :return:
            ptext: processed line
        '''
    text = re.sub(r'\{\{[^\}\{]*\}\}', '', text)

    text = re.sub(r"'[']*([^']+)[']*'", r'\1', text)
    # text = re.sub(r"\([^\(\)]+\)", '', text)
    text = re.sub(r"(https\://|http\://)[^ ]+", '', text)
    text = re.sub(r'<ref[^>]*>[.]*</ref>|<ref[^>/]*/>', r'', text)

    text = re.sub(r"\<[^\>\<]+\>", '', text)
    text = re.sub(r"\=\=See also\=\=[\n\{\}a-zA-Z \|*'\[\]:\(\)0-9-;.\<\>\=/\?,\t_&\#\"\!\+]*", '', text)
    text = re.sub(r"\=[\=]+[^\=]+[\=]+\=", r'', text)
    # text = re.sub(r"\*[\:\-;\<\>\{\}\=/ '\[\]a-zA-Z\|\)\(0-9,\.\!_&\?]+",r'',text)
    text = re.sub(r"[^. ]+\.com", r'', text)
    text = re.sub(r"\[\[([^|\]]+)[^\]]*\]\]", r"\1", text)
    text = re.sub(r"\*[^*\n]+", r'', text)
    text = re.sub(r'[^\n\:]+\:', r"", text)
    text = re.sub(r'[ ]*[\n]+', '\n', text)

    text = re.sub(r'\n[ ]*[\|\:;\*\#\!\{\'][^\n]*', '', text)
    text = re.sub(r'\n[ ]*&nbsp[^\n]*', '', text)
    text = re.sub(r'\n[^\n]*\]\]', '', text)
    text = re.sub(r"\[[^\[\]]+\]", '', text)
    return text

def text2line_tokenize(text):
    text = re.sub(r"( [^ \.]+[^A-Z \.]{1})[ ]*\.[ ]+([A-Z0-9])", r"\1.\n\2", text)
    text = re.sub(r"( [^ \.]+[^A-Z \.]{1})[ ]*;[ ]+([A-Za-z0-9])", r"\1.\n\2", text)
    lines = text.split('\n')
    return lines

def text2line_tokenize2(text):
    text = re.sub(r"( [^\.]+)[ ]*\.[ ]+([A-Z0-9])", r"\1.\n\2", text)
    text = re.sub(r"( [^\.]+)[ ]*;[ ]+([A-Za-z0-9])", r"\1.\n\2", text)
    lines = text.split('\n')
    return lines

def text_process(text):
    ptext = text_preprocessing(text)
    if re.match(r'[^A-Z]{2}\.$',text.rstrip(' |\n|\t')[-2:]):
        left = ''
    else:
        ptext = re.sub(r"( [^ \.]+[^A-Z \.]{2}[ ]*\.)[ ]+([A-Z0-9])",r'\1',text)
        left = text[len(ptext):]
    del text
    ptext = text2line_tokenize(ptext)
    for i in range(len(ptext)):
        ptext[i] = line_processing(ptext[i])
    return ptext, left

def text_process_n(text):
    ptext = text.rstrip(' |\n').split('\n')
    for i in range(len(ptext)):
        ptext[i] = line_processing(ptext[i])
    return ptext, ''

def get_n_feature(line1, line2):
    nfeat = [0,0,0]
    p = re.compile(' [0-9]+ | [0-9]+\.[0-9]+ ')
    m1 = p.findall(line1)
    m2 = p.findall(line2)
    # if m1:
    #     print m1
    # if m2:
    #     print m2

    if m1 and m2:
        nfeat[0] = 0
    elif not m1 and not m2:
        nfeat[0]=1
        return nfeat
    else:
        return nfeat

    if len(m1) == len(m2):
        nfeat[0] = 1
        flag=0
        tm1 = [i for i in m1]
        tm2 = [i for i in m2]
        for i in m1:
            if i in tm2:
                tm2.remove(i)
                tm1.pop(0)
        if not tm2 and not tm1:
            nfeat[1]=1
    else:
        nfeat[0] = 0
        nfeat[0] = 0
        tm = [m1, m2] if len(m1)<len(m2) else [m2, m1]
        for i in tm[1]:
            if i in tm[0]:
                tm[0].remove(i)
        if not tm[0]:
            nfeat[2] = 1
    return nfeat






if __name__ == '__main__':
    lines = ['s.& p. 500 slipped 12.27 points or 1.2 percent to 981.73',
'suppose confirmed one thing british public consistently dull &lt no offence robbie pleeease thousand better songs formulated cheesy pop song kids',
'pratt &whitney said 75 per cent of engine equipment outsourced to europe final assembly in germany',
'pg&e corporation shares up 39 cents or 2.6 percent 15.59 dollar on new york stock exchange on tuesday',
'feature approval of book publishers puts 33 million pages of searchable text disposal of amazon.com shoppers',
'mcarthur told internetnews.com price declines moderated and remained below 30 percent from previous year last two quarters',
'personally id like to see cartoons transformers thundercats and m.a.s.k. get full hollywood remakes']
    for line in lines:
        print line
        print line_processing(line,fstopwds = "/media/zero/41FF48D81730BD9B/Final_Thesies/Coding/final_STS_machine_2/extra/stopwords2.txt")
        raw_input()