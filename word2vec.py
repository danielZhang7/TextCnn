#encoding:utf-8
import logging
import time
import codecs
import sys
import re
import jieba
from gensim.models import word2vec
from .model import TextConfig


class Get_Sentences(object):
    '''

    Args:
         filenames: a list of train_filename,test_filename,val_filename
    Yield:
        word:a list of word cut by jieba

    '''

    def __init__(self,filenames):
        self.filenames= filenames
        # jieba.load_userdict(file_userdict)

    def __iter__(self):
        for filename in self.filenames:
            with codecs.open(filename, 'r', encoding='utf-8') as f:
                for _,line in enumerate(f):
                    try:
                        line=line.strip()
                        line=line.split('\t')
                        word=[]
                        assert len(line)==2
                        word.extend(jieba.lcut(line[1]))
                        yield word
                    except:
                        pass

def train_word2vec(vector_word_filename,filenames):
    '''
    use word2vec train word vector
    argv:
        filenames: a list of train_filename,test_filename,val_filename
    return: 
        save word vector to config.vector_word_filename

    '''
    t1 = time.time()
    sentences = Get_Sentences(filenames)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=1, workers=6)
    model.wv.save_word2vec_format(vector_word_filename, binary=False)
    print('-------------------------------------------')
    print("Training word2vec model cost %.3f seconds...\n" % (time.time() - t1))


