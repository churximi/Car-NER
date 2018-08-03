#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：
时间：
"""
import json

f1 = open("/Users/simon/myprojects/ana3/DuReader/data3/preprocessed/testset/search_test.json")
f2 = open("/Users/simon/myprojects/ana3/DuReader/data3/preprocessed/testset/zhidao_test.json")
out = open("/Users/simon/myprojects/ana3/DuReader/临时结果/测试集问题.json", "w+")

all_records = []

for i, line in enumerate(f1):
    record = {"question_id": '',
              "questions": [],
              "fact_or_opinion": ''
              }
    if i % 100 == 0:
        print(i)
    sample = json.loads(line.strip())
    record["question_id"] = sample["question_id"]
    record["questions"].append(sample["segmented_question"])
    record["fact_or_opinion"] = sample["fact_or_opinion"]
    for item in sample["documents"]:
        record["questions"].append(item["segmented_title"])

    all_records.append(record)

for i, line in enumerate(f2):
    record = {"question_id": '',
              "questions": [],
              "fact_or_opinion": ''
              }
    if i % 100 == 0:
        print(i)
    sample = json.loads(line.strip())
    record["question_id"] = sample["question_id"]
    record["questions"].append(sample["segmented_question"])
    record["fact_or_opinion"] = sample["fact_or_opinion"]
    for item in sample["documents"]:
        record["questions"].append(item["segmented_title"])

    all_records.append(record)

print(len(all_records))
json.dump(all_records, out, indent=4, ensure_ascii=False)
out.close()

if __name__ == "__main__":
    pass

