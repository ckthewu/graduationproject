# coding: utf-8
import sys, re, os

from CONST import NEWSPATH, SOURCENEWSPATH
default_encoding="utf-8"
if(default_encoding!=sys.getdefaultencoding()):
    reload(sys)
    sys.setdefaultencoding(default_encoding)

kind_dict = {"business": 'fortune', "cul": 'culture', "mil": 'mil', "sports": 'sports', "learning": 'tech'}
article_count = {"fortune": 0, "culture": 0, "mil": 0, "sports": 0, "tech": 0}
MAXSIZE = 1500
doc_list = []
have_list = {}

docre = re.compile(r"<doc>.*?</doc>", re.S)
urlre = re.compile(r"<url>(.*?)</url>", re.S)
docnore = re.compile(r"<docno>(.*?)</docno>", re.S)
contentre = re.compile(r"<content>(.*?)</content>", re.S)

def readDOC(path):
    with open(path, "r") as f:
        doc_list = docre.findall(f.read())
    print len(doc_list)

    for doc in doc_list:
        url = urlre.findall(doc)
        if url:
            url = url[0]
            kind = url.split("//")[1].split(".")[0]
            if kind in kind_dict.keys():
                kind = kind_dict[kind]
                if article_count[kind] < MAXSIZE:
                    content = contentre.findall(doc)[0].strip()
                    docno = docnore.findall(doc)[0]
                    if content and docno and not docno in have_list:
                        if len(content) > 0:
                            print 'read ' + docno
                            with open(NEWSPATH + kind + '/' + docno, 'w') as f:
                                try:
                                    f.write(content.decode('GB18030'))
                                    have_list[docno] = 1
                                    article_count[kind] += 1
                                    print 'ok'
                                except UnicodeEncodeError as e:
                                    print 'error'
                                    print content
                                    continue
                                    
    flag = 0
    for key in article_count.keys():
        if article_count[key] < MAXSIZE:
            break
        flag += 1
    print article_count
    return flag == len(article_count.keys())


for kind in article_count.keys():
    dirList = os.listdir(NEWSPATH + kind)
    for name in dirList:
        have_list[name] = 1
    article_count[kind] = len(dirList)
    
for news in os.listdir(SOURCENEWSPATH):
    if readDOC(SOURCENEWSPATH + news):
        print '够了'
        break
