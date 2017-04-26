# coding:utf-8
import jieba, os, json, word2vec
from CONST import DICT, IDFDICT, W2VMODEL, KINDLIST, NEWSPATH, DATAPATH
jieba.set_dictionary(DICT)
stop_word = {}
model = word2vec.load(W2VMODEL)
add_array = [0 for x in range(100)]
with open(IDFDICT, 'r') as f:
    for line in f.readlines():
        word, _, idf, c = line.rstrip('\n').split(" ")
        if float(idf) < 4:
            stop_word[word.decode("utf-8")] = 1


def get_word_vector(filepath):
    out_list = []
    word_count = 0
    with open(filepath, "r") as f:
        for l in f.readlines():
            for word in jieba.cut(l, cut_all=True):
                if word_count >= 100:
                    break
                if not word in stop_word and word in model:
                    out_list.extend(model[word].tolist())
                    word_count += 1
            if word_count >= 100:
                break
    if word_count != 100:
        return 0

    return out_list

def write_data():
    tranDataFile = open(DATAPATH + 'trandata_wv.json', 'w')
    tranLableFile = open(DATAPATH + 'tranlable_wv.json', 'w')
    testDataFile1 = open(DATAPATH + 'testdata1_wv.json', 'w')
    testLableFile1 = open(DATAPATH + 'testlable1_wv.json', 'w')
    testDataFile2 = open(DATAPATH + 'testdata2_wv.json', 'w')
    testLableFile2 = open(DATAPATH + 'testlable2_wv.json', 'w')
    testcount1 = testcount2 = 0
    trancount = 0
    for kind in KINDLIST:
        name_list = os.listdir(NEWSPATH + kind)
        one_hot_array = [0 for x in range(len(KINDLIST))]
        one_hot_array[KINDLIST.index(kind)] = 1
        for i, name in enumerate(name_list):
            print "read " + name
            v = get_word_vector(NEWSPATH + kind + os.sep + name)
            if v != 0:
                r = i % 8
                if r == 0:
                    print name + 'add to test1'
                    testDataFile1.write(json.dumps(v) + '\n')
                    testLableFile1.write(json.dumps(one_hot_array) + '\n')
                    testcount1 += 1
                elif r == 4:
                    print name + 'add to test2'
                    testDataFile2.write(json.dumps(v) + '\n')
                    testLableFile2.write(json.dumps(one_hot_array) + '\n')
                    testcount2 += 1
                else:
                    print name + 'add to tran'
                    tranDataFile.write(json.dumps(v) + '\n')
                    tranLableFile.write(json.dumps(one_hot_array) + '\n')
                    trancount += 1
            #tranArray.append({'x': numpy.asarray(getVector(NEWSPATH + kind + os.sep + name)), 'y': numpy.asarray(one_hot_array)})
    tranDataFile.close()
    tranLableFile.close()
    testDataFile1.close()
    testLableFile1.close()
    testDataFile2.close()
    testLableFile2.close()
    print 'ok'
    print 'test set size %d/%d. tran set size %d' % (testcount1, testcount2, trancount)

write_data()
# v = get_word_vector("../sohunews/news/tech/6a2f6a2c47f592b3-91413306c0bb3300")
# print len(v)
# print v
