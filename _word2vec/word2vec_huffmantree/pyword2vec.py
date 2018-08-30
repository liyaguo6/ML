__author__ = 'multiangle'

import math

from word2vec_huffmantree.WordCount import WordCounter,MulCounter
import word2vec_huffmantree.File_Interface as FI
from word2vec_huffmantree.HuffmanTree import HuffmanTree
import numpy as np
import jieba
from sklearn import preprocessing

class Word2Vec():
    def __init__(self, vec_len, learn_rate=0.025, win_len=5, model='cbow'):
        self.cutted_text_list = None
        self.vec_len = vec_len
        self.learn_rate = learn_rate
        self.win_len = win_len
        self.model = model
        self.word_dict = None  # each element is a dict, including: word,possibility,vector,huffmancode
        self.huffman = None    # the object of HuffmanTree

    def Load_Word_Freq(self,word_freq_path):
        # load the info of word frequence
        # will generate a word dict
        if self.word_dict is not None:
            raise RuntimeError('the word dict is not empty')
        word_freq = FI.load_pickle(word_freq_path)
        self.__Gnerate_Word_Dict(word_freq)

    def __Gnerate_Word_Dict(self,word_freq):
        # generate a word dict
        # which containing the word, freq, possibility, a random initial vector and Huffman value
        if not isinstance(word_freq,dict) and not isinstance(word_freq,list):
            raise ValueError('the word freq info should be a dict or list')

        word_dict = {}
        if isinstance(word_freq,dict):
            # if word_freq is in type of dictionary
            sum_count = sum(word_freq.values())
            for word in word_freq:
                temp_dict = dict(
                    word = word,
                    freq = word_freq[word],
                    possibility = word_freq[word]/sum_count,
                    vector = np.random.random([1,self.vec_len]),
                    Huffman = None
                )
                word_dict[word] = temp_dict
        else:
            # if word_freq is in type of list
            freq_list = [x[1] for x in word_freq]
            sum_count = sum(freq_list)

            for item in word_freq:
                temp_dict = dict(
                    word = item[0],
                    freq = item[1],
                    possibility = item[1]/sum_count,
                    vector = np.random.random([1,self.vec_len]),
                    Huffman = None
                )
                word_dict[item[0]] = temp_dict
        self.word_dict = word_dict
    def Import_Model(self,model_path):
        model = FI.load_pickle(model_path)  # a dict, {'word_dict','huffman','vec_len'}
        self.word_dict = model.word_dict
        self.huffman = model.huffman
        self.vec_len = model.vec_len
        self.learn_rate = model.learn_rate
        self.win_len = model.win_len
        self.model = model.model

    def Export_Model(self,model_path):
        data=dict(
            word_dict = self.word_dict,
            huffman = self.huffman,
            vec_len = self.vec_len,
            learn_rate = self.learn_rate,
            win_len = self.win_len,
            model = self.model
        )
        FI.save_pickle(data,model_path)

    def Train_Model(self,text_list):
        """
        利用WordCount.py统计词频，self.__Gnerate_Word_Dict生成word_dict
        利用HuffmanTree.py 构造树形结构,生成节点所在的二进制码,初始化各非叶节点的中间向量和叶节点中的词向量。
        :param text_list:
        :return:
        """
        print(self.word_dict)
        print(self.huffman)
        # generate the word_dict and huffman tree
        if self.huffman==None:
            # if the dict is not loaded, it will generate a new dict
            if self.word_dict==None :
                wc = WordCounter(text_list)
                self.__Gnerate_Word_Dict(wc.count_res.larger_than(3))
                self.cutted_text_list = wc.text_list
            # generate a huffman tree according to the possibility of words
            self.huffman = HuffmanTree(self.word_dict,vec_len=self.vec_len)
        print('word_dict and huffman tree already generated, ready to train vector')

        # start to train word vector
        before = (self.win_len-1) >> 1
        after = self.win_len-1-before

        if self.model=='cbow':
            method = self.__Deal_Gram_CBOW
        else:
            method = self.__Deal_Gram_SkipGram

        if self.cutted_text_list:
            # print(self.cutted_text_list)
            # if the text has been cutted
            total = self.cutted_text_list.__len__()
            count = 0
            for line in self.cutted_text_list:
                line_len = line.__len__()
                for i in range(line_len):
                    method(line[i],line[max(0,i-before):i]+line[i+1:min(line_len,i+after+1)])
                count += 1
                # print('{c} of {d}'.format(c=count,d=total))

        else:
            # if the text has note been cutted
            for line in text_list:
                line = list(jieba.cut(line,cut_all=False))
                line_len = line.__len__()
                for i in range(line_len):
                    method(line[i],line[max(0,i-before):i]+line[i+1:min(line_len,i+after+1)])
        print('word vector has been generated')
        # self.t__Deal_Gram_CBOW('数学',['研究','复数','概念','几何'])
        # print('#'*20)
        # self.t__Deal_Gram_CBOW('数学',['研究','复数','概念','几何'])
    def __Deal_Gram_CBOW(self,word,gram_word_list):

        if not self.word_dict.__contains__(word):
            return

        word_huffman = self.word_dict[word]['Huffman']
        gram_vector_sum = np.zeros([1,self.vec_len])
        for i in range(gram_word_list.__len__())[::-1]:
            item = gram_word_list[i]
            if self.word_dict.__contains__(item):
                gram_vector_sum += self.word_dict[item]['vector']
            else:
                gram_word_list.pop(i)
        # print(gram_vector_sum)
        if gram_word_list.__len__()==0:
            return
        e = self.__GoAlong_Huffman(word_huffman,gram_vector_sum,self.huffman.root)
        for item in gram_word_list:
            self.word_dict[item]['vector'] += e
            self.word_dict[item]['vector'] = preprocessing.normalize(self.word_dict[item]['vector'])  #更新窗口长度的词向量
        # print(self.word_dict)
    def __Deal_Gram_SkipGram(self,word,gram_word_list):

        if not self.word_dict.__contains__(word):
            return

        word_vector = self.word_dict[word]['vector']
        for i in range(gram_word_list.__len__())[::-1]:
            if not self.word_dict.__contains__(gram_word_list[i]):
                gram_word_list.pop(i)

        if gram_word_list.__len__()==0:
            return

        for u in gram_word_list:
            u_huffman = self.word_dict[u]['Huffman']
            e = self.__GoAlong_Huffman(u_huffman,word_vector,self.huffman.root)
            self.word_dict[word]['vector'] += e
            self.word_dict[word]['vector'] = preprocessing.normalize(self.word_dict[word]['vector'])

    def __GoAlong_Huffman(self,word_huffman,input_vector,root):
        """
        训练中间向量和词向量
        :param word_huffman:
        :param input_vector:
        :param root:
        :return:
        """
        node = root   #哈夫曼树对象
        # print(node)
        e = np.zeros([1,self.vec_len])
        for level in range(word_huffman.__len__()):
            huffman_charat = word_huffman[level]
            q = self.__Sigmoid(input_vector.dot(node.value.T))
            # print(node.value)
            grad = self.learn_rate * (1-int(huffman_charat)-q)
            e += grad * node.value
            # print(e)
            node.value += grad * input_vector
            node.value = preprocessing.normalize(node.value)  #更新根节点Θ
            # print(node.value)
            if huffman_charat=='0':  #将当前节点切换到路径上的下一节点
                node = node.right
            else:
                node = node.left
        # print(node.value)
        return e

    def __Sigmoid(self,value):
        return 1/(1+math.exp(-value))

if __name__ == '__main__':
    #读取预料
    with open('./static/test',encoding='utf-8') as f:
        text_list = []
        for line in f:
            if len(line) !='':
                text_list.append([line.replace('\t','').replace('\n','').replace(' ','')])
    ww = Word2Vec(3)
    ww.Train_Model(text_list)

    #把最后一层的哈夫曼树保存到pickle文件中
    FI.save_pickle(ww.word_dict,'./static/wv.pkl')
    data = FI.load_pickle('./static/wv.pkl')
    print(data)
    # 抽取每个词的词向量保存到pickle文件中
    # x = {}
    # for key in data:
    #     temp = data[key]['vector']
    #     temp = preprocessing.normalize(temp)
    #     x[key] = temp
    # FI.save_pickle(x,'./static/normal_wv.pkl')

    #读取pickle文件中的词向量，计算相似度
    # x = FI.load_pickle('./static/normal_wv.pkl')
    # def cal_simi(data,key1,key2):
    #     return data[key1].dot(data[key2].T)[0][0]
    # keys=list(x.keys())
    # for key in keys:
    #     print(key,'\t',cal_simi(x,'研究',key))


    # ps:
    # 哈夫曼树数据结构展示(重点)

    # {'数学': {'word': '数学', 'freq': 17, 'possibility': 0.3269230769230769,
    #         'vector': array([[0.81232832, 0.2732206, 0.51524091]]), 'Huffman': '11'},
    #  '研究': {'word': '研究', 'freq': 5, 'possibility': 0.09615384615384616,
    #         'vector': array([[0.65121999, 0.40657625, 0.64078724]]), 'Huffman': '001'},
    #  '中': {'word': '中', 'freq': 5, 'possibility': 0.09615384615384616,
    #        'vector': array([[0.49411877, 0.68572384, 0.53444313]]), 'Huffman': '000'},
    #  '复数': {'word': '复数', 'freq': 4, 'possibility': 0.07692307692307693,
    #         'vector': array([[0.18973421, 0.97940378, 0.06905915]]), 'Huffman': '1011'},
    #  '古希腊': {'word': '古希腊', 'freq': 3, 'possibility': 0.057692307692307696,
    #          'vector': array([[0.4996288, 0.74989563, 0.4336215]]), 'Huffman': '1001'},
    #  '数学家': {'word': '数学家', 'freq': 3, 'possibility': 0.057692307692307696,
    #          'vector': array([[0.53525117, 0.2706997, 0.8001424]]), 'Huffman': '1000'},
    #  '概念': {'word': '概念', 'freq': 3, 'possibility': 0.057692307692307696,
    #         'vector': array([[0.89649673, 0.39364914, 0.20330759]]), 'Huffman': '0101'},
    #  '形式': {'word': '形式', 'freq': 3, 'possibility': 0.057692307692307696,
    #         'vector': array([[-0.00874932, 0.4448266, 0.89557398]]), 'Huffman': '0100'},
    #  '科学': {'word': '科学', 'freq': 3, 'possibility': 0.057692307692307696,
    #         'vector': array([[0.97674141, 0.02713428, 0.21269684]]), 'Huffman': '0111'},
    #  '使用': {'word': '使用', 'freq': 3, 'possibility': 0.057692307692307696,
    #         'vector': array([[0.90620257, 0.32031571, 0.27603397]]), 'Huffman': '0110'},
    #  '发展': {'word': '发展', 'freq': 3, 'possibility': 0.057692307692307696,
            # 'vector': array([[0.73338075, 0.5018207, 0.45861602]]), 'Huffman': '1010'}}







