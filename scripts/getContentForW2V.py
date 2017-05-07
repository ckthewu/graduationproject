# coding: utf-8
import sys, jieba, re, os
from CONST import NEWSPATH, W2VSOURCEFILE
default_encoding="utf-8"
if(default_encoding!=sys.getdefaultencoding()):
    reload(sys)
    sys.setdefaultencoding(default_encoding)

docre = re.compile(r"<doc>.*?</doc>", re.S)# 提取每篇新闻
contentre = re.compile(r"<content>(.*?)</content>", re.S)# 提取新闻内容
i = 0
with open(W2VSOURCEFILE, 'wb') as wf:
    # 读取每个.txt文件
    for news in os.listdir(NEWSPATH):
        # 从中提取所有的新闻
        with open(NEWSPATH + news, "r") as rf:
            doc_list = docre.findall(rf.read())
        # 将每个新闻分词后 写入输出文件的一行
        for doc in doc_list:
            content = contentre.findall(doc)[0].strip()
            wf.write(" ".join(jieba.cut(content.decode('GB18030').encode("utf-8")))+'\n')
            print "step %s" % i
            i += 1
print "ok"
