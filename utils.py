#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import json
import logging
import nltk
import re
from collections import Counter
from conlleval import return_report


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def test_ner(results, path):
    """
    Run perl script to evaluate model
    """
    output_file = os.path.join(path, "predict.txt")
    with open(output_file, "w") as f:
        to_write = []
        for block in results:
            for line in block:
                to_write.append(line + "\n")
            to_write.append("\n")

        f.writelines(to_write)
    eval_lines = return_report(output_file)
    return eval_lines


def load_config(config_file):
    """
    Load configuration of the model
    parameters are stored in json format
    """
    with open(config_file, encoding="utf8") as f:
        return json.load(f)


def save_model(sess, model, path, logger):
    checkpoint_path = os.path.join(path, "ner.ckpt")
    model.saver.save(sess, checkpoint_path)
    logger.info("model saved")


def ngram(par_list, par_n):
    """n-gram"""
    temp = nltk.ngrams(par_list, par_n)
    newwords_list = ["".join(item) for item in temp]
    cnt = Counter(newwords_list)
    words = cnt.most_common()

    return words


def get_lines(file_path):
    """get lines"""
    with open(file_path, "r", encoding="utf-8") as f:
        line_list = []
        for line in f:
            temp = line.strip().lower()
            if temp not in line_list:
                line_list.append(temp)

    return line_list


def update_file(words, dict_file):
    """ update files

    """
    with open(dict_file, "w+", encoding="utf-8") as fout:
        for word in words:
            fout.write(word + "\n")


def sentence_seg(sen):
    """"""
    seg_punc = ".;；？?！!"
    sen_pat = re.compile("(.*?[.;；？?！!])")
    sen = sen.strip()

    if sen[-1] not in seg_punc:
        sen += "。"

    return re.findall(sen_pat, sen)


def judge_zh(content):
    """"""
    zh_pat = re.compile('[\u4e00-\u9fa5]+')
    if zh_pat.search(content):
        return True


def write_data(file_path, data):
    """"""
    with open(file_path, "w+", encoding="utf-8") as fout:
        for item in data:
            for char, label in zip(item[0], item[1]):
                if char.strip():
                    fout.write("{} {}\n".format(char, label))
            fout.write("\n")


if __name__ == "__main__":
    pass
