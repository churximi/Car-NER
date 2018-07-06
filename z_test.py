#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
author：Simon
date：
about：
"""

import re

test_string = "这是一个测试。万得公司是一家金融公司。小米公司是一家互联网公司。万得公司是一家金融公司。"
print(len(test_string))

company_list = ["万得公司", "小米公司"]
infos = [{"text": "南京", "type": "location", "pos": 7},
         {"text": "小米公司", "type": "company", "pos": 29},
         {"text": "测试", "type": "test", "pos": 4},
         {"text": "。万", "type": "test", "pos": 6}]

original_labels = ["O"] * len(test_string)
print(original_labels)

company_areas = []
for company in company_list:
    start_ends = [i.span() for i in re.finditer(company, test_string)]
    for start, end in start_ends:
        company_areas.extend(list(range(start, end)))
        original_labels[start] = "B-COM"
        original_labels[start + 1: end] = ["I-COM"] * (end - start - 1)

print(original_labels)

for info in infos:
    if info["type"] != "company":
        start = info["pos"]
        end = info["pos"] + len(info["text"])
        areas = list(range(start, end))
        if set(areas).intersection(set(company_areas)):
            print("预标注词【{}】与公司名存在交集，不予标注...".format(info["text"]))
            continue
        else:
            original_labels[start] = "B-{}".format(info["type"])
            original_labels[start + 1: end] = ["I-{}".format(info["type"])] * (end - start - 1)

print(original_labels)


if __name__ == "__main__":
    pass
