# encoding:utf-8
from .model import *
from .loader import *
import tensorflow as tf
import tensorflow.contrib.keras as kr
import os
import jieba
import re
import heapq


class Predict:

    def predict(self, sentences, project_name):
        base_path = os.path.dirname(os.path.realpath(__file__))
        config = TextConfig(project_name)
        config.pre_training = get_training_word2vec_vectors(config.vector_word_npz)
        model = TextCNN(config)
        save_dir = base_path + '/checkpoints/' + project_name
        save_path = os.path.join(save_dir, 'best_validation')

        categories, cat_to_id = read_category(project_name)
        _, word_to_id = read_vocab(config.vocab_filename)
        input_x = self.process_file(sentences, word_to_id, max_length=config.seq_length)

        labels = {}
        idx = 0
        for x in categories:
            labels[idx] = x
            idx += 1

        feed_dict = {
            model.input_x: input_x,
            model.keep_prob: 1,
        }
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path=save_path)
        y_prob = session.run(model.prob, feed_dict=feed_dict)
        y_prob = y_prob.tolist()
        cat = []
        for prob in y_prob:
            top2 = list(map(prob.index, heapq.nlargest(1, prob)))
            cat.append(labels[top2[0]])
        tf.reset_default_graph()
        session.close()

        return cat

    def sentence_cut(self, sentences):
        """
        Args:
            sentences: a list of text need to segment
        Returns:
            seglist:  a list of sentence cut by jieba

        """
        re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")  # the method of cutting text by punctuation
        seglist = []
        for sentence in sentences:
            words = []
            blocks = re_han.split(sentence)
            for blk in blocks:
                if re_han.match(blk):
                    words.extend(jieba.lcut(blk))
            seglist.append(words)
        return seglist

    def process_file(self, sentences, word_to_id, max_length=600):
        """
        Args:
            sentences: a text need to predict
            word_to_id:get from def read_vocab()
            max_length:allow max length of sentence
        Returns:
            x_pad: sequence data from  preprocessing sentence

        """
        data_id = []
        seglist = self.sentence_cut(sentences)
        for i in range(len(seglist)):
            data_id.append([word_to_id[x] for x in seglist[i] if x in word_to_id])
        x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
        return x_pad

    def main(self, sentence, project_name):
        sentences = [sentence]
        if not project_name:
            return False
        cat = self.predict(sentences, project_name)
        for i, sentence in enumerate(sentences, 0):
            print('----------------------the text-------------------------')
            print(sentence[:50] + '....')
            print('the predict label:%s' % cat[i])
            return cat[i]
