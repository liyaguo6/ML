import tensorflow as tf


class TextCnn:
    def __init__(self, sequence_length, num_classes, embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda=0.0):
        # Placeholders for input, output, dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        ## Embedding layer
        # with tf.device('/cpu:0'), tf.name_scope('embedding'):
            # self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0, name='W'))  # ???
        self.embedded_chars = self.input_x  # ???
        self.embedded_chars_expended = tf.expand_dims(self.embedded_chars,
                                                          -1)  # shape : [ batch , height , width , 1 ]
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):  # filter_size：[2,3,4]
            with tf.name_scope('conv_maxpool-%s' % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                                name='W')  # shape : [filter_sizes, embedding_size, 1, num_filters]
                b = tf.Variable(tf.constant(0.1, shape=[num_filters], name='b'))
                conv = tf.nn.conv2d(self.embedded_chars_expended,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='conv'
                                    )  # shape :  [ batch , sequence_length - filter_sizes + 1 , 1 , num_filters ]
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name='pool'
                                        )  # shape: [batch , 1 , 1 , num_filters]
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)  # 3*128
        self.h_pool = tf.concat(pooled_outputs, 3)  # ???? [batch,1,1,num_filters_total]
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])  # shape: [batch,num_filters_total] =
        print("################conv_maxpool#######################")
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnomalized) scores and predictions
        with tf.name_scope('output'):
            W = tf.get_variable('W',
                                shape=[num_filters_total, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer()
                                )
            b = tf.Variable(
                tf.constant(0.1, shape=[num_classes]),
                name='b'
            )
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='score')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')
        print("#######################output##########################")
        # Calculate Mean cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.losses = tf.reduce_mean(losses) + l2_loss*l2_reg_lambda
        print("#######################loss##########################")
        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions =tf.equal(self.predictions,tf.argmax(self.input_y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
