汽车实体识别代码主要说明

2018-06-22

1、结巴分词，可更改
2、简易标点分句
3、需要人工构建初始实体词典
4、维护一个停用词表和非实体此表
5、词表自动去重
6、侧重于中文数据，过滤英文
7、考虑加入更多数据清洗操作
8、数据预处理后获取词频统计信息，考虑输出更多其他统计信息
9、字母、数字、中文组合实体的问题
10、随便用了一个字向量文件，考虑更换
11、新发现的实体词需要人工审核修正，并手动加入entities词典
