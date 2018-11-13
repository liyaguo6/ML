import tensorflow as tf
from data_process import load_data_files,padding_sentences,batch_iter
from wor2vec import embedding_sentences
import time,os
import numpy as np
from cnn import TextCnn
import datetime
# data loading params
tf.app.flags.DEFINE_integer('val_sample_percentage',0.1,'peecentage of training data to use for testing')
tf.app.flags.DEFINE_integer('test_sample_percentage',0.1,'peecentage of training data to use for validation')
tf.app.flags.DEFINE_string('training_data','./data/cnn_data.csv','training data filename')
tf.app.flags.DEFINE_string('word2vec_model', 'word2vec_model', 'embedding word2vec of chinese word')
tf.app.flags.DEFINE_string('trained_word2vec_model', 'word2vec_model', 'embedding word2vec of chinese word')
tf.flags.DEFINE_integer("num_labels", 2, "Number of labels for data. (default: 2)")


# Model Hparams
tf.flags.DEFINE_integer('embedding_dim', 128, 'dimensionality of character')
tf.flags.DEFINE_string('filter_size','3,4,5','filter size')
tf.flags.DEFINE_integer('number_filter',128,'number of filter')
tf.app.flags.DEFINE_float('dropout',0.7,'dropout')
tf.app.flags.DEFINE_float('L2_reg_lambda',0.01,'L2')
tf.app.flags.DEFINE_float('learning_rate',0.01,'learning rate')


#Model params
tf.app.flags.DEFINE_integer('epochs',20,'number of epochs')
tf.app.flags.DEFINE_integer('batch_size',128,'size of batch')
tf.flags.DEFINE_integer('evaluate_every', 500, 'evaluate_every')
tf.flags.DEFINE_integer('checkpoint_every', 1000, 'saving...')
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("padding_sentence_length", 15, "padding sentence length")

# Misc parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Parse parameters from commands
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('\nParameters:')
for attr, val in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(), val))

x_text ,y = load_data_files(FLAGS.training_data)

# Prepare output directory for models and summaries
timestamp = str(int(time.time()))
out_dir =os.path.dirname(os.path.join(os.path.curdir,'data/runs',timestamp))
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# Get embedding vector
sentences, max_document_length = padding_sentences(x_text, '<PADDING>',FLAGS.padding_sentence_length)

x =np.array(embedding_sentences(sentences=sentences,file_to_save=os.path.join(out_dir,FLAGS.word2vec_model),file_to_load=os.path.join(out_dir,FLAGS.trained_word2vec_model)))

# Shuffle data randomly
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/valid/test set
test_sample_index = -1 * int(FLAGS.test_sample_percentage * float(len(y)))

x_train, x_test = x_shuffled[:test_sample_index], x_shuffled[test_sample_index:]
y_train, y_test = y_shuffled[:test_sample_index], y_shuffled[test_sample_index:]
val_sample_index = -1 * int(FLAGS.val_sample_percentage * float(len(y_train)))
x_val, y_val = x_train[val_sample_index:], y_train[val_sample_index:]

print("Train/Test/val split: {:d}/{:d}/{:d}".format(len(y_train), len(y_test),len(y_val)))

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        #  设置tf.ConfigProto()中参数allow_soft_placement=True，允许tf自动选择一个存在并且可用的设备来运行操作。
        allow_soft_placement=FLAGS.allow_soft_placement,
        # 参数log_device_placement = True ,可以获取到 operations 和 Tensor 被指派到哪个设备(几号CPU或几号GPU)上运行,会在终端打印出各项操作是在哪个设备上运行的。
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        cnn =TextCnn(sequence_length=x_train.shape[1],
                     num_classes=y_train.shape[1],
                     embedding_size = x_train.shape[2],
                     filter_sizes=list(map(int,FLAGS.filter_size.split(','))),
                     num_filters =FLAGS.number_filter,
                     l2_reg_lambda = FLAGS.L2_reg_lambda,
        )

        # Define Training procedure
        global_step = tf.Variable(0,name='global_step',trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.losses)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        print("Writing to {}\n".format(out_dir))
        #
        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.losses)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
        #
        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        #
        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "test")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
        #
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        sess.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout
            }
            _, step, summeries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.losses, cnn.accuracy], feed_dict=feed_dict
            )
            time_str = datetime.datetime.now().isoformat()
            print('{}:step:{} , loss:{} , acc:{}'.format(time_str, step, loss, accuracy))


        def val_test_step(x_batch, y_batch,model):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summeries, loss, accuracy = sess.run(
                [global_step, train_summary_op, cnn.losses, cnn.accuracy], feed_dict=feed_dict
            )
            time_str = datetime.datetime.now().isoformat()
            if model == 'test':
                print('test  {}: step:{} , loss:{} , acc:{}'.format(time_str, step, loss, accuracy))
            else:
                print('val  {}: step:{} , loss:{} , acc:{}'.format(time_str, step, loss, accuracy))


        batchs =batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.epochs)

        for batch in batchs:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print('\n evaluate_every')
                val_test_step(x_test, y_test,'test')
                val_test_step(x_val, y_val,'val')
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, './model/', global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
