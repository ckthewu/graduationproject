# coding: utf-8
import sys, re, os

from CONST import NEWSPATH, SOURCENEWSPATH
default_encoding="utf-8"
if(default_encoding!=sys.getdefaultencoding()):
    reload(sys)
    sys.setdefaultencoding(default_encoding)

kind_dict = {"business": 'fortune', "cul": 'culture', "mil": 'mil', "sports": 'sports', "learning": 'tech'} # url到分类标准的映射
article_count = {"fortune": 0, "culture": 0, "mil": 0, "sports": 0, "tech": 0}# 当前各类文章数量
MAXSIZE = 1500 # 每类文章所需数量
doc_list = []
have_list = {} # 已存在的文章

docre = re.compile(r"<doc>.*?</doc>", re.S) # 提取文本
urlre = re.compile(r"<url>(.*?)</url>", re.S) # 提取url 用于分类
docnore = re.compile(r"<docno>(.*?)</docno>", re.S) # 提取文本编号 用于命名
contentre = re.compile(r"<content>(.*?)</content>", re.S) # 提取内容

# 读取每个`.txt`文件 从中解析文本
def readDOC(path):
    # 获取所有包含文本的xml
    with open(path, "r") as f:
        doc_list = docre.findall(f.read())

    for doc in doc_list:
        url = urlre.findall(doc)
        if url:
            url = url[0]
            kind = url.split("//")[1].split(".")[0] # 从url中解析出类别标识符 在搜狐新闻中类别名就是主机名
            if kind in kind_dict.keys():
                kind = kind_dict[kind] # 类名标准化
                if article_count[kind] < MAXSIZE: # 判断该列别数量是否饱和 饱和则跳过
                    content = contentre.findall(doc)[0].strip()
                    docno = docnore.findall(doc)[0]
                    if content and docno and not docno in have_list:# 如果这个文本符合有内容且事先不存在的需求
                        if len(content) > 0:
                            print 'read ' + docno
                            #写入数据
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
    # 判断是否每类都已经饱和了
    for key in article_count.keys():
        if article_count[key] < MAXSIZE:
            return True;
    print article_count
    return False;


for kind in article_count.keys(): # 从各类文章存储的文件夹中获取已经含有的文章名
    dirList = os.listdir(NEWSPATH + kind)
    for name in dirList:
        have_list[name] = 1
    article_count[kind] = len(dirList)

for news in os.listdir(SOURCENEWSPATH):
    # 全都饱和则停止
    if readDOC(SOURCENEWSPATH + news):
        print '够了'
        break
