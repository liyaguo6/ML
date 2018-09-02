from gensim.models import word2vec
import logging
import jieba
# logging.basicConfig(format='%(asctime)s : %(levelname)s :  %(message)s', level=logging.INFO)
# raw_wods = ['欧几里得 西元前三世纪的古希腊数学家 现在被认为是几何之父 此画为拉斐尔的研究作品 雅典学院 数学 是利用符号语言研究数量 结构 变化以及空间等概念的一门学科',
#             '数学拾遗 清代丁取忠撰 直到 经过中国数学名词审查委员会研究 算学 数学 两词的使用状况后 确认以 数学 表示今天意义']

raw_wods=['结构 偏序 全序 拓扑结构 邻域 极限 连通性 维数 style border px solid ddd text align center margin auto cellspacing px px px px 数论 群论']
sentences = [ list(jieba.cut(i)) for i in raw_wods]
print(sentences)
# model = word2vec.Word2Vec(sentences,min_count=2)
# print(model.similarity('数学','几何'))