# coding: utf-8
import json, random
from scripts.CONST import DATAPATH
import tensorflow as tf
from scripts.CONST import KINDMAP, KINDLIST, FATHERPATH

# w矩阵初始化
def weight_variable(shape, name = None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

# b向量初始化
def bias_variable(shape, name = None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

# 添加summaries
def variable_summaries(var, name):
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

# 获取随机batch
def getBatch(xarr, yarr, size):
    indexarr = [x for x in range(len(xarr))]
    batchx = []
    batchy = []
    for i in range(size):
        index = indexarr.pop(random.randint(0, len(indexarr)-1))
        batchx.append(xarr[index])
        batchy.append(yarr[index])
    return (batchx, batchy)


learning_rate = 5e-2
batch_size = 100
tran_times = 500
outSize = 5
# 数据集初始化
tranDataArray = []
tranLableArray = []
testDataArray1 = []
testLableArray1 = []
testDataArray2 = []
testLableArray2 = []
with open(DATAPATH + 'trandata.json', 'r') as f:
    for l in f.readlines():
        tranDataArray.append(json.loads(l.rstrip("\n")))
    print 'ok'
    print "trandata size " + str(len(tranDataArray))

with open(DATAPATH + 'tranlable.json', 'r') as f:
    for l in f.readlines():
        tranLableArray.append(json.loads(l.rstrip("\n")))
    print 'ok'
    print "tranlable size " + str(len(tranLableArray))

with open(DATAPATH + 'testdata1.json', 'r') as f:
    for l in f.readlines():
        testDataArray1.append(json.loads(l.rstrip("\n")))
    print 'ok'
    print "testdata 1 size " + str(len(testDataArray1))

with open(DATAPATH + 'testlable1.json', 'r') as f:
    for l in f.readlines():
        testLableArray1.append(json.loads(l.rstrip("\n")))
    print 'ok'
    print "testlable 1 size " + str(len(testLableArray1))

with open(DATAPATH + 'testdata2.json', 'r') as f:
    for l in f.readlines():
        testDataArray2.append(json.loads(l.rstrip("\n")))
    print 'ok'
    print "testdata 2 size " + str(len(testDataArray2))

with open(DATAPATH + 'testlable2.json', 'r') as f:
    for l in f.readlines():
        testLableArray2.append(json.loads(l.rstrip("\n")))
    print 'ok'
    print "testlable 2 size " + str(len(testLableArray2))

def getCategoryData(dataArray, lableArray):
    categoryArray = [ [] for x in range(len(lableArray[0]))]
    categoryLable = [ [] for x in range(len(lableArray[0]))]
    if len(dataArray) != len(lableArray) or len(lableArray[0]) != outSize:
        pass
    else:
        for i in range(len(lableArray)):
            categoryArray[lableArray[i].index(1)].append(dataArray[i])
            categoryLable[lableArray[i].index(1)].append(lableArray[i])
    return (categoryArray, categoryLable)
testCategoryData1, testCategoryLable1 = getCategoryData(testDataArray1, testLableArray1)
testCategoryData2, testCategoryLable2 = getCategoryData(testDataArray2, testLableArray2)

L = len(testDataArray1[0])
print L

# 定义网络
# 输入x 为一长度为L的文本向量， 答案y_real 为长度为outSize的one-hot向量
x = tf.placeholder("float", [None, L])
y_real = tf.placeholder("float", [None, outSize], name='correct_output')
keepProb = tf.placeholder(tf.float32)

# 两个隐藏层 3000/1024个神经元全连接
W_fc1 = weight_variable([L, 3000])
b_fc1 = bias_variable([3000])
out_fc1 = tf.nn.dropout(tf.nn.softmax(tf.matmul(x, W_fc1) + b_fc1), keepProb)

W_fc2 = weight_variable([3000, 1024])
b_fc2 = bias_variable([1024])
out_fc2 = tf.nn.dropout(tf.nn.softmax(tf.matmul(out_fc1, W_fc2) + b_fc2), keepProb)

# 输出层 outSize个神经元
W_ol = weight_variable((1024, outSize))
b_ol = bias_variable([outSize])
y = tf.nn.softmax(tf.matmul(out_fc2, W_ol) + b_ol, name='output')


#交叉熵loss函数
cross_entropy = -tf.reduce_sum(y_real * tf.log(y), name='loss')

# 定义训练的梯度下降算法 学习率 以及最小化的函数
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# 定义正确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_real, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")

# 将loss和正确率加入summary
variable_summaries(cross_entropy, 'loss')
variable_summaries(accuracy, 'accuracy')

# 初始化变量等
init = tf.global_variables_initializer()
saver = tf.train.Saver()
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter("/tmp/model_logs_fc2", sess.graph)
    sess.run(init)

    for i in range(tran_times):
        batch_xs, batch_ys = getBatch(tranDataArray, tranLableArray, batch_size)
        sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_real: batch_ys, keepProb: 0.5})
        if i % (tran_times/20) == 0:
            summary, acc = sess.run([summary_op, accuracy], feed_dict={x: testDataArray1, y_real: testLableArray1, keepProb: 1})
            summary_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))

    line = ['fc2', '测试集1',]
    print '训练集正确率'
    print sess.run(accuracy, feed_dict={x: tranDataArray, y_real: tranLableArray, keepProb: 1})
    print '测试集1正确率'
    ac = str(sess.run(accuracy, feed_dict={x: testDataArray1, y_real: testLableArray1, keepProb: 1}))
    print ac
    line.append(ac)
    for i in range(outSize):
        ac = str(sess.run(accuracy, feed_dict={x: testCategoryData1[i], y_real: testCategoryLable1[i], keepProb: 1}))
        print '%s类的正确率为%s' % (KINDMAP[KINDLIST[i]], str(ac))
        line.append(ac)
    with open(FATHERPATH + '/outtable.csv', 'a') as f:
        f.write(','.join(line) + '\n')
        line = ['fc2', '测试集2',]

    print '测试集2正确率'
    ac = str(sess.run(accuracy, feed_dict={x: testDataArray2, y_real: testLableArray2, keepProb: 1}))
    print ac
    line.append(ac)
    for i in range(outSize):
        ac = str(sess.run(accuracy, feed_dict={x: testCategoryData2[i], y_real: testCategoryLable2[i], keepProb: 1}))
        print '%s类的正确率为%s' % (KINDMAP[KINDLIST[i]], str(ac))
        line.append(ac)
    with open(FATHERPATH + '/outtable.csv', 'a') as f:
        f.write(','.join(line) + '\n')

    # save_path = saver.save(sess, "/tmp/model_fc2.ckpt")
    # print "模型存储位置: ", save_path
