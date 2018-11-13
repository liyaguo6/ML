import tensorflow as tf

class TextCnn:
    def __init__(self,sequence_length,num_classes,embedding_size,filter_sizes,num_filters,l2_reg_lambda):
        # Placeholders for input, output, dropout
        self.input_x= tf.placeholder(tf.float32,[None,sequence_length,embedding_size],name='input_x')
        self.input_y= tf.placeholder(tf.float32,[None,num_classes],name='input_x')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        with tf.device('/cpu:0'),tf.name_scope('embedding'):
            self.embedding_chars = self.input_x
            self.embedding_chars_expanded=tf.expand_dims(self.embedding_chars,-1)  # shape : [ batch , height , width , 1 ]
        pooled_outputs = []
        for i,filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv_maxpool-%s' % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W1 = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='W1')
                b1 = tf.Variable(tf.constant(0.1,shape=[num_filters]),name='b1')
                conv = tf.nn.conv2d(self.embedding_chars_expanded,W1,strides=[1,1,1,1],padding='VALID',name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv,b1),name='relu') #[ batch , sequence_length - filter_sizes + 1 , 1 , num_filters ]
                pool = tf.nn.max_pool(h,ksize=[1,sequence_length-filter_size+1,1,1],strides=[1,1,1,1],padding='VALID',name='pool')
                pooled_outputs.append(pool)

        num_filters_size = num_filters*len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool,shape=[-1,num_filters_size])
        with tf.name_scope('dropout'):
            self.dropout = tf.nn.dropout(self.h_pool_flat,self.dropout_keep_prob)
        with tf.name_scope('fully_connected'):
            W2 =tf.Variable(tf.truncated_normal([num_filters_size,100],stddev=0.1),name='W2')
            b2 = tf.Variable(tf.constant(0.1,shape=[100,],name='b2'))
            self.connected = tf.nn.xw_plus_b(self.dropout,W2,b2)
        # Final (unnomalized) scores and predictions
        with tf.name_scope('output'):

            W3 = tf.Variable(tf.truncated_normal([100, num_classes],stddev=0.1),name='W3')
            b3 = tf.Variable(tf.constant(0.1, shape=[num_classes], name='b'))
            l2_loss += tf.nn.l2_loss(W3)
            l2_loss += tf.nn.l2_loss(b3)
            self.socres = tf.nn.xw_plus_b(self.connected, W3, b3,name='socres')
            self.predictions = tf.argmax(self.socres, 1, name='predictions')

        # Calculate Mean cross-entropy loss
        with tf.name_scope('loss'):
            loss  = tf.nn.softmax_cross_entropy_with_logits(logits=self.socres,labels=self.input_y)
            self.losses = tf.reduce_mean(loss)+l2_loss*l2_reg_lambda
        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions,tf.argmax(self.input_y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


