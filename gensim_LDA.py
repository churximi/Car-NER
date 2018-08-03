#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：LDA学习
时间：2017年3月2日 14:33:41
"""

from gensim import corpora
from gensim import models
from Funcs.TextSta import TextSta

T = TextSta("/Users/simon/myprojects/ana3/词向量/亚马逊中文书评/全部书评分词_去除标点.txt")  # 类TextSta
sentences = T.sen()  # 获取句子列表

dictionary = corpora.Dictionary(sentences)  # 建立字典索引
corpus = [dictionary.doc2bow(sentence) for sentence in sentences]  # bow向量

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

tfidf.save("model.tfidf")
# tfidf = models.TfidfModel.load("model.tfidf")

lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=200)
corpus_lda = lda[corpus_tfidf]

for i in range(10):
    print(lda.print_topic(i))
