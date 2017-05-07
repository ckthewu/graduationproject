# coding:utf-8
import jieba, os, json
from CONST import DICT, KINDLIST, NEWSPATH, IDFDICT, DATAPATH
jieba.set_dictionary(DICT)
idf_dict = {}# 存储idf值的字典
# 字典初始化
with open(IDFDICT, 'r') as f:
    for line in f.readlines():
        word, _, idf, c = line.rstrip('\n').split(" ")
        # 提取出其中idf值小于四（停用词）的词， 存入字典
        if 4 < float(idf):
            idf_dict[word.decode("utf-8")] = {"index": len(idf_dict), "idf": float(idf)}
dict_length = len(idf_dict)
print dict_length

# 获取文本向量
def get_vector(filepath):
    # 初始化输出序列
    out_list = [{'count': 0, 'word': ''} for x in range(dict_length)]
    MAXPOOLSIZE = 10 # 降维使用的框大小
    word_count = 0 # 文本总词数统计
    # 统计词频
    with open(filepath, "r") as f:
        for l in f.readlines():
            for i in jieba.cut(l, cut_all=True):
                word_count += 1
                if i in idf_dict:
                    ind = int(idf_dict[i]["index"])
                    out_list[ind]['count'] += 1
                    out_list[ind]['word'] = i
    # 计算tf-idf
    for index, d in enumerate(out_list):
        if d['word'] in idf_dict and idf_dict[d['word']].has_key('idf'):
            tf = d['count'] / float(word_count)
            idf = idf_dict[d['word']]['idf']
            out_list[index] = tf*idf
        else:
            out_list[index] = 0
    # 使用maxpooling的思想对向量降维
    newList = []
    i = 0
    while i < dict_length - MAXPOOLSIZE:
        newList.append(reduce(max, out_list[i : i + MAXPOOLSIZE]))
        i += MAXPOOLSIZE
    # 判断向量是否全0（无效）
    for i in newList:
        if i != 0:
            return newListe
    return 0


# 数据写入
def write_data():
    # 分别为训练集 测试集1 2
    tranDataFile = open(DATAPATH + 'trandata.json', 'w')
    tranLableFile = open(DATAPATH + 'tranlable.json', 'w')
    testDataFile1 = open(DATAPATH + 'testdata1.json', 'w')
    testLableFile1 = open(DATAPATH + 'testlable1.json', 'w')
    testDataFile2 = open(DATAPATH + 'testdata2.json', 'w')
    testLableFile2 = open(DATAPATH + 'testlable2.json', 'w')
    testcount1 = testcount2 = 0
    trancount = 0
    for kind in KINDLIST:
        nameList = os.listdir(NEWSPATH + kind)
        oneHotArray = [0 for x in range(len(KINDLIST))]
        oneHotArray[KINDLIST.index(kind)] = 1
        for i, name in enumerate(nameList):
            print "read " + name
            v = get_vector(NEWSPATH + kind + os.sep + name)
            if v != 0:
                r = i % 6
                if r == 0:
                    print name + 'add to test1'
                    testDataFile1.write(json.dumps(v) + '\n')
                    testLableFile1.write(json.dumps(oneHotArray) + '\n')
                    testcount1 += 1
                elif r == 3:
                    print name + 'add to test2'
                    testDataFile2.write(json.dumps(v) + '\n')
                    testLableFile2.write(json.dumps(oneHotArray) + '\n')
                    testcount2 += 1
                else:
                    print name + 'add to tran'
                    tranDataFile.write(json.dumps(v) + '\n')
                    tranLableFile.write(json.dumps(oneHotArray) + '\n')
                    trancount += 1
    tranDataFile.close()
    tranLableFile.close()
    testDataFile1.close()
    testLableFile1.close()
    testDataFile2.close()
    testLableFile2.close()
    print 'ok'
    print 'test set size %d/%d. tran set size %d' % (testcount1, testcount2, trancount)

write_data()
# v = get_vector("../sohunews/news/tech/6a2f6a2c47f592b3-91413306c0bb3300")
# print len(v)
