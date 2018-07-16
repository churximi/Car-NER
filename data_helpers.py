#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
author：Simon
date：
about：
"""

import os
import re
import jieba
from utils import ngram, get_lines, update_file, sentence_seg, judge_zh, get_logger
from collections import Counter

class DataHelpers:

    def __init__(self):
        # paras
        self.all_data_file = "data/all_data.txt"
        self.all_data_clean = "temp/all_data_clean.txt"
        self.data_after_seg = "temp/data_after_seg.txt"

        self.all_dicts_path = "all_dicts"
        self.known_words_file = "temp/known_words.txt"
        self.unknown_words_file = "temp/unknown_words_fre.txt"
        self.words_fre_file = "temp/words_fre.txt"
        self.gram2_words_fre_file = "temp/gram2_words_fre.txt"

        self.user_dict_one = "all_dicts/entities.txt"
        self.user_dict_two = "all_dicts/special_user_dict.txt"

        self.helper_logger = "log/helper.log"

        # logging
        self.logger = get_logger(self.helper_logger)

        self.get_known_words()
        if not os.path.exists(self.all_data_clean):
            self.preprocess()

        self.all_words = self.jieba_seg()
        self.word_fre = self.get_wordfre()

        self.get_unknown_words()

        self.new_word_fre = ngram(self.all_words, 2)
        self.get_2gram_words()

    def get_known_words(self):
        file_names = os.listdir(self.all_dicts_path)
        known_words = []

        for name in file_names:
            if name.endswith("txt"):
                dict_file = os.path.join(self.all_dicts_path, name)
                words = get_lines(dict_file)
                self.logger.info("file:{}, num of words:{}".format(name, len(words)))
                update_file(words, dict_file)
                known_words.extend(words)

        update_file(known_words, self.known_words_file)
        self.logger.info("num of known words: {}".format(len(known_words)))

    def preprocess(self):
        contents = []
        with open(self.all_data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().lower()
                if judge_zh(line):
                    sentences = sentence_seg(line)
                    for sen in sentences:
                        if judge_zh(sen):
                            contents.append(sen)
        update_file(contents, self.all_data_clean)

    def jieba_seg(self):
        self.logger.info("word seg processing...")
        jieba.load_userdict(self.user_dict_one)
        jieba.load_userdict(self.user_dict_two)

        out = open(self.data_after_seg, "w+", encoding="utf-8")

        all_words = []
        with open(self.all_data_clean, encoding="utf-8") as f:
            for index, line in enumerate(f):
                if index % 1000 == 0:
                    print("processing lines:{}".format(index))

                words = list(jieba.cut(line.strip()))
                all_words += words
                out.write(" ".join(words) + "\n")
        out.close()

        return all_words

    def get_wordfre(self):
        self.logger.info("get_wordfre processing...")
        cnt = Counter(self.all_words)
        word_fre = cnt.most_common()
        with open(self.words_fre_file, "w+", encoding="utf-8") as out:
            for word, fre in word_fre:
                out.write("{}\t{}\n".format(word, fre))

        return word_fre

    def get_unknown_words(self):
        known_words = get_lines(self.known_words_file)
        with open(self.unknown_words_file, "w+", encoding="utf-8") as fout:
            for word, fre in self.word_fre:
                if word not in known_words:
                    fout.write("{}\t{}\n".format(word, fre))

    def get_2gram_words(self):
        comb_pat = re.compile('[,，。！!；;？?()（）“”"\-、：:*~【】]+')
        with open(self.gram2_words_fre_file, "w+", encoding="utf-8") as fout:
            for item in self.new_word_fre:
                if not comb_pat.search(item[0]):
                    fout.write("{}\t{}\n".format(item[0], item[1]))


if __name__ == "__main__":
    dh = DataHelpers()
