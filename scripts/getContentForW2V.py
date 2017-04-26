# coding: utf-8
import sys, jieba, re, os
from CONST import NEWSPATH, W2VSOURCEFILE
default_encoding="utf-8"
if(default_encoding!=sys.getdefaultencoding()):
    reload(sys)
    sys.setdefaultencoding(default_encoding)

docre = re.compile(r"<doc>.*?</doc>", re.S)
contentre = re.compile(r"<content>(.*?)</content>", re.S)
i = 0
with open(W2VSOURCEFILE, 'wb') as wf:
    for news in os.listdir(NEWSPATH):
        with open(NEWSPATH + news, "r") as rf:
            doc_list = docre.findall(rf.read())
        for doc in doc_list:
            content = contentre.findall(doc)[0].strip()
            wf.write(" ".join(jieba.cut(content.decode('GB18030').encode("utf-8")))+'\n')
            print "step %s" % i
            i += 1
print "ok"