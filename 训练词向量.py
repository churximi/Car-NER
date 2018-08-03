#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：word2vec模型训练方法2，利用LineSentence读入文本（感觉较好）
时间：2017年09月20日13:55:43
备注：直接在程序中加入分词功能
"""

import logging
import jieba
import os
import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# 主程序
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def jieba_seg(file_path, file_after_seg, user_dic):
    jieba.load_userdict(user_dic)  # 加载用户词典
    output = open(file_after_seg, "w+")

    with open(file_path) as f:
        for line in f:
            words = jieba.cut(line)
            output.write(" ".join(words))

    output.close()


def save(model_save_path, vocab_file_out, word2vec_file_out):
    model = gensim.models.Word2Vec.load(model_save_path)

    f1 = open(vocab_file_out, "w+")

    array_all = model.wv[model.wv.index2word[0]]
    f1.write(model.wv.index2word[0] + "\n")
    for i in range(1, len(model.wv.index2word)):
        word = model.wv.index2word[i]
        f1.write(word + "\n")
        array_all = np.row_stack((array_all, model.wv[word]))
        if i % 1000 == 0:
            print(i)

    print(len(array_all))
    np.savetxt(word2vec_file_out, array_all, fmt="%.8f")
    f1.close()


def train_model(file_after_seg, model_save_path, vocab_file_out, word2vec_file_out):
    model = Word2Vec(LineSentence(open(file_after_seg, "r")),
                     size=300, min_count=0, sg=1, workers=2)

    model.save(model_save_path)
    save(model_save_path, vocab_file_out, word2vec_file_out)


if __name__ == "__main__":
    original_data = "/Users/simon/myprojects/ana3/Medical_NER2/original_data/医学汇总数据.txt"
    data_after_seg = "/Users/simon/myprojects/ana3/DuReader/临时结果/问题_实体.txt"
    seg_dic = "/Users/simon/myprojects/ana3/Medical_NER2/user_dic/用户词典.txt"
    out_vocab = "files/vocab.txt"
    out_word2vec = "files/word2vec.txt"
    model_path = "model/word2vec模型.model"

    if not os.path.exists(data_after_seg):
        jieba_seg(original_data, data_after_seg, seg_dic)
    if not os.path.exists(model_path):
        train_model(data_after_seg, model_path, out_vocab, out_word2vec)
    else:
        save(model_path, out_vocab, out_word2vec)
