#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
author：Simon
date：
about：
"""

from utils import get_lines, write_data, get_logger
from sklearn.model_selection import train_test_split


class Annotate:

    def __init__(self):
        # paras
        self.annotate_logger = "log/annotate.log"
        self.entities_file = "all_dicts/entities.txt"
        self.data_after_seg = "temp/data_after_seg.txt"

        self.train_file = "data/train.txt"
        self.dev_file = "data/dev.txt"
        self.test_file = "data/test.txt"

        # logging
        self.logger = get_logger(self.annotate_logger)

        # annotate
        self.entities = get_lines(self.entities_file)
        self.annotate()

    def create_labels(self, words):
        chars = []
        labels = []

        for word in words:
            if word in self.entities:
                chars += list(word)
                labels.append("B-e")
                labels += ["I-e"] * (len(word) - 1)
            else:
                chars += list(word)
                labels += ["O"] * len(words)

        assert len(chars) == len(labels)

        return chars, labels

    def annotate(self):
        self.logger.info("annotate processing...")
        char_labels = []

        with open(self.data_after_seg, "r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                if index % 1000 == 0:
                    self.logger.info("processing lines:{}".format(index))

                words = line.strip().split(" ")
                chars, labels = self.create_labels(words)
                char_labels.append((chars, labels))

        self.logger.info("num of sentences:{}".format(len(char_labels)))
        train_dev, test = train_test_split(char_labels, test_size=0.1)
        train, dev = train_test_split(train_dev, test_size=0.1)

        write_data(self.train_file, train)
        write_data(self.dev_file, dev)
        write_data(self.test_file, test)
        self.logger.info("train/dev/test: {} / {} /{}".format(len(train), len(dev), len(test)))


if __name__ == "__main__":
    anno = Annotate()
