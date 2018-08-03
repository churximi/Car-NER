#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：测试gensim使用，处理中文语料
时间：2017年3月9日 21:30:16
"""

import logging
from gensim.models.word2vec import Word2Vec
from Funcs.TextSta import TextSta

# 主程序
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',  # 查看输出
                    level=logging.INFO)

print("请选择大语料的分词文本...")
T = TextSta("")
sentences = T.sen()    # 获取句子列表，每个句子又是词汇的列表

print('训练Word2vec模型（可尝试修改参数）...')
model = Word2Vec(sentences,
                 size=100,  # 词向量维度
                 min_count=5,  # 词频阈值
                 window=5)  # 窗口大小

# 输出词语原始向量
print(model[u"不错"])

# 计算两个词的相似度/相关程度
y1 = model.similarity(u"不错", u"好")
print("【不错】和【好】的相似度为：", y1)

# 计算某个词的相关词列表
y2 = model.most_similar(u"不错", topn=10)  # 20个最相关的
print("和【不错】最相关的词有：\n")
for item in y2:
    print(item[0], item[1])

# 寻找对应关系
print("书-不错，质量-")
y3 = model.most_similar([u'质量', u'不错'], [u'书'], topn=3)
for item in y3:
    print(item[0], item[1])

# 寻找不合群的词
y4 = model.doesnt_match(u"书 书籍 教材 很".split())
print("不合群的词：", y4)

# 保存模型，以便重用
# model.save(u"书评.model")
# 对应的加载方式
# model_2 = word2vec.Word2Vec.load("text8.model")

# 以一种C语言可以解析的形式存储词向量
# model.save_word2vec_format(u"书评.model.bin", binary=True)
# 对应的加载方式
# model_3 = word2vec.Word2Vec.load_word2vec_format("text8.model.bin", binary=True)

if __name__ == "__main__":
    pass
