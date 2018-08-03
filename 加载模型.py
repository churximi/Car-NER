#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：加载训练好的词向量模型
时间：2017年07月30日16:23:13
"""

import logging
import gensim

# 主程序
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
model = gensim.models.Word2Vec.load("/Users/simon/myprojects/ana3/词向量/model/word2vec模型.model")  # 加载
print(model.wv["南京"])

word = input("输入一个词...\n")
while word != "q":
    try:
        for item in model.most_similar(word, topn=10):
            print(item[0], item[1])
    except:
        pass

    word = input("输入一个词...\n")


"""
vector = model["好"]
print("词向量：%s" % vector)

# 寻找对应关系
y = model.most_similar(positive=['中国', '北京'], negative=['华盛顿'], topn=2)
for item in y:
    print(item[0], item[1])


more_examples = ["he his she", "big bigger bad", "going went being"]
for example in more_examples:
    a, b, x = example.split()
    predicted = model.most_similar([x, b], [a])[0][0]
    print("'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted))

# 寻找不合群的词
# y3 = model.doesnt_match("breakfast cereal dinner lunch".split())
"""

if __name__ == "__main__":
    pass
