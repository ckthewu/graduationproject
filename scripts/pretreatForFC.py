# coding:utf-8
import jieba, os, json
from CONST import DICT, KINDLIST, NEWSPATH, IDFDICT, DATAPATH
jieba.set_dictionary(DICT)
count = 0
dict = {}
# with open('idf', 'r') as f:
#     for index, line in enumerate(f.readlines()):
#         l = line.rstrip("\n").split(" ")
#         word, idf = l[0], l[1]
#         dict[word.decode("utf-8")] = {"index": count, "idf": float(idf)}
#         count += 1

with open(IDFDICT, 'r') as f:
    for line in f.readlines():
        word, _, idf, c = line.rstrip('\n').split(" ")
        if len(word) < 10 and 4 < float(idf):
            dict[word.decode("utf-8")] = {"index": count, "idf": float(idf)}
            count += 1
print count


def get_vector(filepath):
    out_list = [{'count': 0, 'word': ''} for x in range(count)]
    MAXSIZE = 10
    word_count = 0
    
    with open(filepath, "r") as f:
        for l in f.readlines():
            for i in jieba.cut(l, cut_all=True):
                word_count += 1
                if i in dict:
                    ind = int(dict[i]["index"])
                    out_list[ind]['count'] += 1
                    out_list[ind]['word'] = i

    for index, d in enumerate(out_list):
        if d['word'] in dict and dict[d['word']].has_key('idf'):
            tf = d['count'] / float(word_count)
            idf = dict[d['word']]['idf']

            out_list[index] = tf*idf
        else:
            out_list[index] = 0
    newList = []
    i = 0
    while i < count - MAXSIZE:
        newList.append(reduce(max, out_list[i : i + MAXSIZE]))
        i += MAXSIZE
    
    flag = True
    for i in newList:
        if i != 0:
            flag = False
            break
    if flag:
        return 0
    return newList

def write_data():
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
            #tranArray.append({'x': numpy.asarray(get_vector(NEWSPATH + kind + os.sep + name)), 'y': numpy.asarray(oneHotArray)})
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
