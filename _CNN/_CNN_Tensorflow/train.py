import tensorflow as tf
import numpy as np
from matplotlib.pyplot import plot as plt
import time
import datetime
import os
from tensorflow.contrib import learn
import data
import word2vec_helpers
from .cnn import TextCNN

# params


# data loading params
tf.flags.DEFINE_float('dev_sample_percentage', 0.1, 'peecentage of training data to use for validation')
tf.flags.DEFINE_string('positive_data_file', './rt-polaritydata/rt-polarity.pos', 'data source for the positive')
tf.flags.DEFINE_string('negative_data_file', './rt-polaritydata/rt-polarity.neg', 'data source for the negative')

# Model Hparams
tf.flags.DEFINE_integer('embedding_dim', 128, 'dimensionality of character')
tf.flags.DEFINE_string('filter_size', '3,4,5', 'filter size')
tf.flags.DEFINE_integer('num_filters', 128, 'Number of filters')
tf.flags.DEFINE_float('dropout', 0.5, 'Dropout')
tf.flags.DEFINE_float('L2_reg_lambda', 0.0, 'L2')

# train params
tf.flags.DEFINE_integer('batch_size', 64, 'Batch size')
tf.flags.DEFINE_integer('num_epochs', 200, 'number of epochs')
tf.flags.DEFINE_integer('evaluate_every', 100, 'evaluate_every')
tf.flags.DEFINE_integer('checkpoint_every', 100, 'saving...')
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")


# Misc parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")




FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('\nParameters:')
for attr, val in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(), val))

# load data
x_text, y = data.load_data_and_lables(FLAGS.positive_data_file, FLAGS.negative_data_file)

# Prepare output directory for models and summaries
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# Get embedding vector
sentences, max_document_length = data.padding_sentences(x_text, '<PADDING>')
x = np.array(word2vec_helpers.embedding_sentences(sentences, embedding_size=FLAGS.embedding_dim,
                                                  file_to_save=os.path.join(out_dir, 'trained_word2vec.model')))

# Shuffle data randomly
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]






# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# Training
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        # 设置tf.ConfigProto()中参数allow_soft_placement=True，允许tf自动选择一个存在并且可用的设备来运行操作。
        log_device_placement=FLAGS.log_device_placement)  # 设置tf.ConfigProto()中参数log_device_placement = True ,可以获取到 operations 和 Tensor 被指派到哪个设备(几号CPU或几号GPU)上运行,会在终端打印出各项操作是在哪个设备上运行的。
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda
        )

        # Define Training procedure
        global_step = tf.Variable(0,name='global_step')
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.losses)
        train_op = optimizer.apply_gradients(grads_and_vars,global_step)

saver = tf.train.Saver(tf.global_variables(),max_to_keep=FLAGS.num_checkpoints)
sess.run(tf.global_variables_initializer())