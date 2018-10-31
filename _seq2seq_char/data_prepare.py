import numpy as np
import time
import tensorflow as tf


# print(source_data)

def extract_character_vocab(data):
    '''
    构造映射表
    '''
    special_words = ['<PAD>', '<UNK>', '<GO>',  '<EOS>']

    set_words = list(set([character for line in data.split('\n') for character in line]))
    # 这里要把四个特殊字符添加进词典
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int





def process_decoder_input(data, vocab_to_int, batch_size):
    '''
    补充<GO>，并移除最后一个字符
    data = [[1,2,3,4,5,6,7,8],[11,12,13,14,15,16,17,18],[11,12,13,14,15,16,17,18]]
    ending =[[ 1  2  3  4  5  6  7] [11 12 13 14 15 16 17] [11 12 13 14 15 16 17]]
    decoder_input = [[15  1  2  3  4  5  6  7] [15 11 12 13 14 15 16 17] [15 11 12 13 14 15 16 17]]
    '''
    # cut掉最后一个字符
    ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return decoder_input


def pad_sentence_batch(sentence_batch, pad_int):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]



def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    '''
    定义生成器，用来获取batch
    pad_targets_batch: array([[24, 10, 10,  4, 20,  3,  0,  0],
       [10, 14, 27, 15,  7,  7, 28,  3],
       [25,  7,  6, 23,  3,  0,  0,  0],
       [18, 21, 22, 16, 29,  3,  0,  0],
       [12, 26, 21,  4, 13,  3,  0,  0]])
    targets_lengths =  [6, 8, 5, 6, 6]
    '''
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        # 补全序列
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # 记录每条记录的长度
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths

if __name__ == '__main__':
    def get_data(file):
        with open(file, "r", encoding='utf-8') as f:
            return f.read()

    batch_size =5
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
    # 将数据集分割为train和validation
    train_source = source_int[batch_size:]
    train_target = target_int[batch_size:]
    # 留出一个batch进行验证
    valid_source = source_int[:batch_size]
    valid_target = target_int[:batch_size]
    (valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(
        get_batches(valid_target, valid_source, batch_size,
                    source_letter_to_int['<PAD>'],
                    target_letter_to_int['<PAD>']))

    display_step = 50  # 每隔50轮输出loss

    ret=get_batches(train_target, train_source, batch_size,
                source_letter_to_int['<PAD>'],
                target_letter_to_int['<PAD>'])
    # print(source_int)
    # print(target_int_to_letter)
    print(source_int_to_letter)
    print(next(ret))