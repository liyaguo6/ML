import os
import sys
import logging
import multiprocessing
import time
import json
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np

def output_vocab(vocab):
    for k, v in vocab.items():
        print(k)


def embedding_sentences(sentences, embedding_size=128, window=5, min_count=5, file_to_load=None, file_to_save=None):
    """
    训练词向量
    :param sentences:[["医院"，'治疗','疾病'],……,]
    :param embedding_size:
    :param window:
    :param min_count:
    :param file_to_load:
    :param file_to_save:
    :return:
    """
    if file_to_load is not None:
        w2vModel = Word2Vec.load(file_to_load)
    else:
        w2vModel = Word2Vec(sentences, size=embedding_size, window=window, min_count=min_count,
                            workers=multiprocessing.cpu_count())
        if file_to_save is not None:
            w2vModel.save(file_to_save)
    all_vectors = []
    embeddingDim = w2vModel.vector_size
    embeddingUnknown = [0 for i in range(embeddingDim)]
    for sentence in sentences:
        this_vector = []
        for word in sentence:
            if word in w2vModel.wv.vocab:
                this_vector.append(w2vModel[word])
            else:
                this_vector.append(embeddingUnknown)
        all_vectors.append(this_vector)
    return all_vectors

if __name__ == '__main__':
    sentences=[["医院",'治疗', '疾病'], ["门诊",'治疗', '感冒','呕吐']]
    x = np.array(embedding_sentences(sentences=sentences,file_to_load='./data/runs/word2vec_model'))
    print(x.shape)
    print(x)


def generate_word2vec_files(input_file, output_model_file, output_vector_file, size=128, window=5, min_count=5):
    start_time = time.time()
    """
    另一种训练词向量模式，所有语句都分万词后保存在文件中
    """

    model = Word2Vec(LineSentence(input_file), size=size, window=window, min_count=min_count,
                     workers=multiprocessing.cpu_count())
    model.save(output_model_file)
    model.wv.save_word2vec_format(output_vector_file, binary=True)

    end_time = time.time()
    print("used time : %d s" % (end_time - start_time))


def run_main():
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 4:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    input_file, output_model_file, output_vector_file = sys.argv[1:4]


# if __name__ == '__main__':

    # path = './data/clean_data.txt'
    # sentences=[]
    # with open(path,'r',encoding='utf-8') as f:
    #     for item in f.readlines():
    #         sentences.append(item.replace("\n","").split(" "))

    # embedding_sentences(sentences=sentences,file_to_save='./data/model')
    # w2vModel = Word2Vec.load('./data/model')
    # ret=embedding_sentences(sentences=sentences[:3],min_count=1,file_to_load='./data/model')
    # print(w2vModel.most_similar('医院'))
    # generate_word2vec_files('./data/clean_data.txt',output_model_file='./data/model2', output_vector_file='./data/model2_vector')
    # w2vModel2 = Word2Vec.load('./data/model2')
    # print(w2vModel2.most_similar('医院'))
    # w2vModel3 = Word2Vec.load('./data/model2_vector')
    # print(w2vModel3['医院'])