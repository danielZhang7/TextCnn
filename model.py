# encoding:utf-8
import tensorflow as tf
import os

base_dir = os.path.dirname(os.path.realpath(__file__))
# 文章分类数
cats_num = {
    'tunews': 17,  # 文章分类为17个
    'spam': 2,  # spam判断只为两类：普通 和 广告
    'fin': 5,
    'stockEmotion': 3,
}

vocab_size = {
    'tunews': 80000,
    'spam': 80000,
    'fin': 8000,
    'stockEmotion': 80000,
}


class TextConfig():
    embedding_size = 100  # dimension of word embedding
    vocab_size = 8000  # number of vocabulary
    pre_training = None  # use vector_char trained by word2vec

    seq_length = 5000  # max length of sentence

    num_filters = 256  # number of convolution kernel
    kernel_size = 5  # size of convolution kernel
    hidden_dim = 128  # number of fully_connected layer units

    keep_prob = 0.5  # droppout
    lr = 1e-3  # learning rate
    lr_decay = 0.9  # learning rate decay
    clip = 5.0  # gradient clipping threshold

    num_epochs = 10  # epochs
    batch_size = 64  # batch_size
    print_per_batch = 100  # print result

    def __init__(self, project_name):
        self.project_name = project_name
        self.num_classes = cats_num[project_name]  # number of labels
        self.vocab_size = vocab_size[project_name]

        self.train_filename = base_dir + '/data/' + project_name + '/train.txt'  # train data
        self.test_filename = base_dir + '/data/' + project_name + '/test.txt'  # test data
        self.val_filename = base_dir + '/data/' + project_name + '/val.txt'  # validation data
        self.vocab_filename = base_dir + '/data/' + project_name + '/vocab.txt'  # vocabulary
        self.vector_word_filename = base_dir + '/data/' + project_name + '/vector_word.txt'  # vector_word trained by word2vec
        self.vector_word_npz = base_dir + '/data/' + project_name + '/vector_word.npz'  # save vector_word to numpy file


class TextCNN(object):

    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32, shape=[None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.cnn()

    def cnn(self):
        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_size],
                                             initializer=tf.constant_initializer(self.config.pre_training))
            embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope('cnn'):
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            outputs = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope('fc'):
            fc = tf.layers.dense(outputs, self.config.hidden_dim, name='fc1')
            fc = tf.nn.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

        with tf.name_scope('logits'):
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='logits')
            self.prob = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
