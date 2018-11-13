import pandas as pd
import re
import jieba
path='./data/cnn_data.csv'
df  = pd.read_csv(path,header=None,encoding='gbk',names=['question','labels']).dropna()
import logging
jieba.setLogLevel(logging.INFO)
import numpy as np

def clean_str(string):
    string = re.sub(r"[^\u4e00-\u9fff]", " ", string)
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # return string.strip().lower()
    return string.strip()


def seperate_line(line):
    return ' '.join(list(jieba.cut(line,cut_all=False)))

# list0=list(df[df['labels']==0]['question'])#30468



# print([clean_str(seperate_line(line)) for line in list0])


# list_0 = [[1,0],[1,0],[1,0]]
# list_1 = [[0,1],[0,1],[0,1],[0,1]]
# y=list_0+ list_1
# # y = np.concatenate([list_1, list_0], 0)
# print(y)
# shuffle_indices = np.random.permutation(np.arange(7))
# print(shuffle_indices)
# y=np.array(y)
# print(y[shuffle_indices])

###################tf.concat#####################
# import tensorflow as tf
# te=tf.get_variable('te',shape=[3,1,1,5])
# l = [te for i in range(3)]
# h_pool = tf.concat(l,3)
# h_pool_flat = tf.reshape(h_pool, [-1,3*5 ])
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     ret=sess.run([h_pool,h_pool_flat])
#     print(ret[0])
#     print(ret[1])


s ='3,4,5'
te = map(int,s.split(','))
print(list(te))


x=np.array([1,23,20,0])
x2=np.array([1,23,2,3])
x3=np.array([1,23,34,9])
x4=np.array([1,0,0,9])
x1 =[[x,x2,x3,x],[x,x2,x3,x]]
print(np.array(x1).shape)
