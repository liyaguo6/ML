import numpy as np
import time
import tensorflow as tf
from seq2seq_model import get_encoder_layer,decoding_layer
from data_prepare import process_decoder_input,extract_character_vocab,get_batches


def get_data(file):
    with open(file,"r",encoding='utf-8') as f:
        return f.read()

source_data = get_data('./data/letters_source.txt')
target_data = get_data('./data/letters_target.txt')


# 构造映射表
source_int_to_letter, source_letter_to_int = extract_character_vocab(source_data)
target_int_to_letter, target_letter_to_int = extract_character_vocab(target_data)



# 对字母进行转换
source_int = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>'])
               for letter in line] for line in source_data.split('\n')]
target_int = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>'])
               for letter in line] + [target_letter_to_int['<EOS>']] for line in target_data.split('\n')]



def get_inputs():
    '''
    模型输入tensor
    '''
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length


# 超参数
# Number of Epochs
epochs = 60
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 50
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 15
decoding_embedding_size = 15
# Learning Rate
learning_rate = 0.001


def seq2seq_model(input_data, targets, lr, target_sequence_length,
                  max_target_sequence_length, source_sequence_length,
                  source_vocab_size, target_vocab_size,
                  encoder_embedding_size, decoder_embedding_size,
                  rnn_size, num_layers):
    # 获取encoder的状态输出
    _, encoder_state = get_encoder_layer(input_data,
                                         rnn_size,
                                         num_layers,
                                         source_sequence_length,
                                         source_vocab_size,
                                         encoding_embedding_size)

    # 预处理后的decoder输入
    decoder_input = process_decoder_input(targets, target_letter_to_int, batch_size)

    # 将状态向量与输入传递给decoder
    training_decoder_output, predicting_decoder_output = decoding_layer(target_letter_to_int,
                                                                        decoding_embedding_size,
                                                                        num_layers,
                                                                        rnn_size,
                                                                        target_sequence_length,
                                                                        max_target_sequence_length,
                                                                        encoder_state,
                                                                        decoder_input,batch_size)

    return training_decoder_output, predicting_decoder_output

#
# # 构造graph
# train_graph = tf.Graph()
#
# with train_graph.as_default():
#     # 获得模型输入
#     input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()
#
#     training_decoder_output, predicting_decoder_output = seq2seq_model(input_data,
#                                                                        targets,
#                                                                        lr,
#                                                                        target_sequence_length,
#                                                                        max_target_sequence_length,
#                                                                        source_sequence_length,
#                                                                        len(source_letter_to_int),
#                                                                        len(target_letter_to_int),
#                                                                        encoding_embedding_size,
#                                                                        decoding_embedding_size,
#                                                                        rnn_size,
#                                                                        num_layers)
#
#     training_logits = tf.identity(training_decoder_output.rnn_output, 'logits') #[batch_size, sequence_length, num_decoder_symbols]
#     predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions') #[batch_size, sequence_length]
#
#     masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')  #[batch_size, sequence_length]
#
#     with tf.name_scope("optimization"):
#         # Loss function
#         cost = tf.contrib.seq2seq.sequence_loss(
#             training_logits,
#             targets,
#             masks)
#
#         # Optimizer
#         optimizer = tf.train.AdamOptimizer(lr)
#
#         # Gradient Clipping
#         gradients = optimizer.compute_gradients(cost)
#         capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
#         train_op = optimizer.apply_gradients(capped_gradients)
#
# # 将数据集分割为train和validation
# train_source = source_int[batch_size:]
# train_target = target_int[batch_size:]
# # 留出一个batch进行验证
# valid_source = source_int[:batch_size]
# valid_target = target_int[:batch_size]
# (valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(
#     get_batches(valid_target, valid_source, batch_size,
#                 source_letter_to_int['<PAD>'],
#                 target_letter_to_int['<PAD>']))
#
# display_step = 50  # 每隔50轮输出loss
#
# import os
#
# if not os.path.exists('./checkpoints/'):
#     os.mkdir('./checkpoints/')
#
# checkpoint = "checkpoints/trained_model.ckpt"
# with tf.Session(graph=train_graph) as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for epoch_i in range(1, epochs + 1):
#         for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
#                 get_batches(train_target, train_source, batch_size,
#                             source_letter_to_int['<PAD>'],
#                             target_letter_to_int['<PAD>'])):
#
#             _, loss = sess.run(
#                 [train_op, cost],
#                 {input_data: sources_batch,
#                  targets: targets_batch,
#                  lr: learning_rate,
#                  target_sequence_length: targets_lengths,
#                  source_sequence_length: sources_lengths})
#
#             if batch_i % display_step == 0:
#                 # 计算validation loss
#                 validation_loss = sess.run(
#                     [cost],
#                     {input_data: valid_sources_batch,
#                      targets: valid_targets_batch,
#                      lr: learning_rate,
#                      target_sequence_length: valid_targets_lengths,
#                      source_sequence_length: valid_sources_lengths})
#                     #Epoch   1/60 Batch    0/77 - Training Loss:  3.400  - Validation loss:  3.396
#                 print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
#                       .format(epoch_i,
#                               epochs,
#                               batch_i,
#                               len(train_source) // batch_size,
#                               loss,
#                               validation_loss[0]))
#
#     # 保存模型
#
#     saver = tf.train.Saver()
#     saver.save(sess, checkpoint)
#     print('Model Trained and Saved')


#####################预测###############################
def source_to_seq(text):
    '''
    对源数据进行转换
    '''
    sequence_length = 7
    return [source_letter_to_int.get(word, source_letter_to_int['<UNK>']) for word in text] + [source_letter_to_int['<PAD>']]*(sequence_length-len(text))


# 输入一个单词
input_word = 'cbadmzyx'
text = source_to_seq(input_word)

checkpoint = "./checkpoints/trained_model.ckpt"

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # 加载模型
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')

    answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                      target_sequence_length: [len(input_word)] * batch_size,
                                      source_sequence_length: [len(input_word)] * batch_size})[0]

pad = source_letter_to_int["<PAD>"]

print('原始输入:', input_word)

print('\nSource')
print('  Word 编号:    {}'.format([i for i in text]))
print('  Input Words: {}'.format(" ".join([source_int_to_letter[i] for i in text])))

print('\nTarget')
print('  Word 编号:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format(" ".join([target_int_to_letter[i] for i in answer_logits if i != pad])))