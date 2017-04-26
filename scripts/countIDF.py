# coding:utf-8
import jieba, os, math
from CONST import KINDLIST, DICT, NEWSPATH, IDFFILE
def countIDF(allpath):
    tempobject = {}
    with open(allpath, "r") as f:
        for line in f.readlines():
            if line:
                for word in jieba.cut(line.strip(), cut_all=True):
                    if word.encode("utf-8") in out_obj:
                        out_obj[word.encode("utf-8")]["count"] += 1
                        if word.encode("utf-8") in tempobject:
                            tempobject[word.encode("utf-8")] += 1
                        else:
                            out_obj[word.encode("utf-8")]["hasarticle"] += 1
                            tempobject[word.encode("utf-8")] = 1
                    else:
                        out_obj[word.encode("utf-8")] = {"count": 1, "hasarticle": 1}
def objectsort(a, b):
    if a["IDF"] > b["IDF"]:
        return -1
    if a["IDF"] < b["IDF"]:
        return 1
    return 0

jieba.set_dictionary(DICT)

count = 0
out_obj = {}
out_list = []

for kind in KINDLIST:
    name_list = os.listdir(NEWSPATH  + kind)
    for name in name_list:
        print "read " + name
        count += 1
        countIDF(NEWSPATH  + kind + os.sep + name)

for k in out_obj:
    if out_obj[k]["hasarticle"] > 10:
        out_list.append({"word": k, "count": out_obj[k]["count"], "hasarticle": out_obj[k]["hasarticle"],
                    "IDF": math.log(float(count)/(out_obj[k]["hasarticle"] + 1))})
out_list = sorted(out_list, objectsort)

with open(IDFFILE, 'w') as f:
    for i in out_list:
        f.write(i["word"] + " " + str(i["IDF"]) + "\n")