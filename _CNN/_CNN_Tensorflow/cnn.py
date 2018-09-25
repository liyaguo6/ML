import tensorflow as tf


class TextCnn:
    def __init__(self, sequence_length, num_classes, embedding_size, vocab_size, filter_sizes, num_filters,
                 l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")

        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, -1.0, name='W'))
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expended = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []
        for i, filter_sizes in enumerate(filter_sizes):  # filter_size：[2,3,4]
            with tf.name_scope('conv_maxpool-%s' % filter_sizes):
                filter_shape = [filter_sizes, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters], name='b'))
                conv = tf.nn.conv2d(self.embedded_chars_expended,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='conv'
                                    )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, sequence_length - filter_sizes + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name='pool'
                                        )
                pooled_outputs.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)  # 3*128
        self.h_pool = tf.concat(pooled_outputs,3)  #????
        self.h_pool_flat = tf.reshape(self.h_pool,[-1,num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout( self.h_pool_flat,self.dropout_keep_prob)

        with tf.name_scope('output'):
            pass


