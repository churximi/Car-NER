#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：word2vec模型训练
时间：2017年12月01日10:45:47
备注：直接在程序中加入分词功能
"""

import logging
import jieba
import os
import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


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
    np.savetxt(word2vec_file_out, array_all)  # 保存词向量矩阵
    f1.close()


def train_model(file_after_seg, model_save_path, vocab_file_out, word2vec_file_out):
    model = Word2Vec(LineSentence(open(file_after_seg, "r")),
                     size=50, min_count=0, sg=1, workers=2)

    model.save(model_save_path)
    save(model_save_path, vocab_file_out, word2vec_file_out)


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    original_data = ""  # 训练语料
    data_after_seg = "/Users/simon/myprojects/ana3/Medical_NER/data/分字语料.txt"  # 有则不分词
    seg_dic = "user_dic/用户词典.txt"  # 分词词典

    out_vocab = "vocab.txt"
    out_word2vec = "word2vec.txt"
    model_path = "word2vec模型.model"

    if not os.path.exists(data_after_seg):  # 训练新的注意删除旧文本
        jieba_seg(original_data, data_after_seg, seg_dic)
    else:
        print("注意：存在旧的分词文本，请确认是否正确！")

    if not os.path.exists(model_path):
        train_model(data_after_seg, model_path, out_vocab, out_word2vec)
    else:
        print("注意：存在旧的word2vec模型，请确认是否正确！")
        save(model_path, out_vocab, out_word2vec)


if __name__ == "__main__":
    main()
