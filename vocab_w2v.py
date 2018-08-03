#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：
时间：
"""

with open("files/vocab.txt") as f:
    vocab = [line.strip("\n") for line in f]

with open("files/word2vec.txt") as f:
    word2vec = [line.strip("\n") for line in f]

with open("files/word_embeddings.txt", "w+") as out:
    for a, b in zip(vocab, word2vec):
        out.write(a + " " + b + "\n")

if __name__ == "__main__":
    pass
