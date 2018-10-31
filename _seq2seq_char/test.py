import tensorflow as tf

import numpy as np

###################tf.contrib.layers.embed_sequence####################
# input_data = [[1,2,3],[4,5,6]]
# voca_size =10
#
# embed_dim =4
#
# ret = tf.contrib.layers.embed_sequence(input_data,voca_size,embed_dim)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(ret))



###################tf.strided_slice####################


# data = [[[1, 1, 1], [2, 2, 2]],
#
# [[3, 3, 3], [4, 4, 4]],
#
# [[5, 5, 5], [6, 6, 6]]]
#
# x = tf.strided_slice(data,[0,0,0],[1,1,1])
#
# with tf.Session() as sess:
#
#        print(sess.run(x))


# data = [[1,2,3,4,5,6,7,8],[11,12,13,14,15,16,17,18],[11,12,13,14,15,16,17,18]]
#         # [[21,22,23,24,25,26,27,28],[211,212,213,214,215,216,217,218]],
#         # [[31,32,33,34,35,36,37,38],[311,12,313,314,15,16,17,18]]]
#
# x = tf.strided_slice(data,[0,0],[3,-1],[1,1])
# decoder_input = tf.concat([tf.fill([3, 1], 15), x], 1)
# # y = tf.strided_slice(data,[0,0,1],[2,2,5])
# with tf.Session() as sess:
#     print(sess.run(x))
#     # print(sess.run(y))
#     print(sess.run(decoder_input))


#######################tf.sequence_mask############################
# test=tf.sequence_mask([1, 3, 2], 5,dtype=tf.float32,)
# with tf.Session() as sess:
#     print(sess.run(test))


#########################tf.contrib.seq2seq.sequence_loss###############

#
# g = tf.Graph()
#
# with g.as_default():
#     x = tf.Variable(1.0, name='x')
#
#     x_plus_1 = tf.assign_add(x, 1, name='x_plus')
#
#     with tf.control_dependencies([x_plus_1]):
#         y = x
#
#         z = tf.identity(x, name='z_added')
#
#     init = tf.global_variables_initializer()
#
#     with tf.Session() as sess:
#         sess.run(init)
#
#         for i in range(5):
#             print(sess.run(z))
#
#             # 输出 2,3,4,5,6
#
#         # 如果改为输出 print(sess.run(y)) ,则结果为 1,1,1,1,1

###################tf.nn.embedding_lookup####################

decoder_input = [[1,2,2,4,5,6,7,8,3,0,],[12,13,14,15,16,3,0,0,0,0],[22,13,14,15,16,27,3,0,0,0]]

decoder_embeddings = tf.Variable(tf.random_uniform([40, 20]))
decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(decoder_embed_input).shape)

encoder_state = (np.random.rand(3,8,30).astype('float32'),np.random.rand(3,8,30).astype('float32'))
# #####################tf.contrib.seq2seq.dynamic_decode#########################
target_sequence_length=tf.Variable([9,6,7])
from tensorflow.python.layers.core import Dense
output_layer = Dense(40,
                     kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

#
decoder_cell = tf.contrib.rnn.LSTMCell(30,
                                       initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))

#
cell = tf.contrib.rnn.MultiRNNCell([decoder_cell for _ in range(2)])

training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                    sequence_length=target_sequence_length,
                                                    time_major=False)
# # 构造decoder
training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                   training_helper,
                                                   encoder_state,
                                                   output_layer)


training_decoder_output,final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                               impute_finished=True,
                                                               maximum_iterations=9)


# training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')

# print(cell)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(target_sequence_length))


#########################tf.contrib.seq2seq.sequence_loss#######################
# training_logits=tf.random_uniform((3,6,20),1.0,2.0)
# targets = tf.random_uniform((3,6),1,5,dtype='int32')
# weigts = tf.random_uniform((3,6),1.0,1.5)
# cost = tf.contrib.seq2seq.sequence_loss(
#     training_logits,
#     targets,
#     weigts)
#
# with tf.Session() as sess:
#     print(sess.run(cost))
