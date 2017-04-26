# coding: utf-8
import json, random, datetime
from scripts.CONST import DATAPATH
import tensorflow as tf


# 权值矩阵的初始化
def weight_variable(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)


# 常数向量的初始化
def bias_variable(shape, name = None):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)


# 二维卷积
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


# 使用2x2的max pool方法
def max_pool_2x2(x, patchHeight):
    return tf.nn.max_pool(x, ksize=[1, sentenceSize-patchHeight+1, 1, 1],strides=[1, 1, 1, 1], padding='VALID')


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


# 获取随机的batch
def get_batch(xarr, yarr, size):

    indexarr = [x for x in range(len(xarr))]
    batchx = []
    batchy = []
    for i in range(size):
        index = indexarr.pop(random.randint(0, len(indexarr)-1))
        batchx.append(xarr[index])
        batchy.append(yarr[index])
    return (batchx, batchy)

# 数据集的初始化
def dataInit():
    tranDataArray = []
    tranLableArray = []
    testDataArray1 = []
    testLableArray1 = []
    testDataArray2 = []
    testLableArray2 = []
    with open(DATAPATH + 'trandata_wv.json', 'r') as f:
        for l in f.readlines():
            newArray = json.loads(l.rstrip("\n"))
            tranDataArray.append(newArray)
        print 'ok'
        print "trandata size " + str(len(tranDataArray))

    with open(DATAPATH + 'tranlable_wv.json', 'r') as f:
        for l in f.readlines():
            tranLableArray.append(json.loads(l.rstrip("\n")))
        print 'ok'
        print "tranlable size " + str(len(tranLableArray))

    with open(DATAPATH + 'testdata1_wv.json', 'r') as f:
        for l in f.readlines():
            newArray = json.loads(l.rstrip("\n"))
            testDataArray1.append(newArray)
        print 'ok'
        print "testdata size " + str(len(testDataArray1))

    with open(DATAPATH + 'testlable1_wv.json', 'r') as f:
        for l in f.readlines():
            testLableArray1.append(json.loads(l.rstrip("\n")))
        print 'ok'
        print "testlable size " + str(len(testLableArray1))

    with open(DATAPATH + 'testdata2_wv.json', 'r') as f:
        for l in f.readlines():
            newArray = json.loads(l.rstrip("\n"))
            testDataArray2.append(newArray)
        print 'ok'
        print "testdata size " + str(len(testDataArray2))

    with open(DATAPATH + 'testlable2_wv.json', 'r') as f:
        for l in f.readlines():
            testLableArray2.append(json.loads(l.rstrip("\n")))
        print 'ok'
        print "testlable size " + str(len(testLableArray2))

    return (tranDataArray, tranLableArray, testDataArray1, testLableArray1, testDataArray2, testLableArray2)

wvSize = 100
sentenceSize = 100
patchNum = 128
batchSize = 200
learningRate = 5e-1
numSteps = 1000
outSize = 5
patchHeights = [2, 3, 4, 5]
NEAR_0 = 1e-15
tranDataArray, tranLableArray, testDataArray1, testLableArray1, testDataArray2, testLableArray2 = dataInit()






acrate = tf.Variable(0.0, name='acrate')

# 输入输出的占位符。 x 为二维向量 sentenceSize*wvSize
x = tf.placeholder(tf.float32, shape=[None, 10000])
x_image = tf.reshape(x, [-1, 100, 100, 1])
y = tf.placeholder(tf.float32, shape=[None, outSize])


## 卷积层
hPools = []
for patchHeight in patchHeights:
    W = tf.Variable(tf.truncated_normal([patchHeight, wvSize, 1, patchNum], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[patchNum]))
    hConv = tf.nn.relu(conv2d(x_image, W) + b)
    hPool = max_pool_2x2(hConv, patchHeight)
    hPools.append(hPool)

## 全连接层

patchNumAll = patchNum*len(patchHeights)
hPool = tf.concat(3, hPools)
hPoolFlat = tf.reshape(hPool, [-1, patchNumAll])

keepProb = tf.placeholder(tf.float32)
hDrop = tf.nn.dropout(hPoolFlat, keepProb)

W = tf.Variable(tf.truncated_normal([patchNumAll, outSize], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[outSize]))

l2_reg_lambda=0.001
l2_loss = tf.nn.l2_loss(W)
l2_loss += tf.nn.l2_loss(b)

scores = tf.nn.xw_plus_b(hDrop, W, b, name="scores")

predictions = tf.argmax(scores, 1, name="predictions")
losses = tf.nn.softmax_cross_entropy_with_logits(scores, y)
loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

correct_prediction = tf.equal(tf.argmax(scores,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')

globalStep = tf.Variable(0)
learning_rate = tf.train.exponential_decay(learningRate, globalStep, numSteps, 0.99, staircase=True)#学习率递减

train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss,  global_step=globalStep)

# cross_entropy = -tf.reduce_sum(y * tf.log(scores + NEAR_0), name='loss')
# train_step = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(scores, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')

#
variable_summaries(loss, 'loss')
variable_summaries(accuracy, 'accuracy')

# important step
init = tf.global_variables_initializer()
saver = tf.train.Saver()
summary_op = tf.summary.merge_all()


# 正式计算
with tf.Session() as sess:

    summary_writer = tf.summary.FileWriter("/tmp/model_logs_cnn1", sess.graph)
    sess.run(init)
    for i in range(numSteps):
        batch_xs, batch_ys = get_batch(tranDataArray, tranLableArray, batchSize)
        #sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y: batch_ys, keepProb: 1.0})
        feed_dict = {x: batch_xs, y: batch_ys, keepProb: 0.5}
        _, step = sess.run([train_step, globalStep], feed_dict)
        if step % 50 == 0:
            # summary, acc = sess.run([summary_op, accuracy], feed_dict={x: testDataArray, y: testLableArray, keepProb: 1.0})
            # summary_writer.add_summary(summary, i)
            # update = tf.assign(acrate, acc)
            # sess.run(update)
            # print('Accuracy at step %s: %s' % (i, acc))
            train_accuracy, summary = sess.run([accuracy, summary_op],
                                               feed_dict={x: testDataArray1, y: testLableArray1, keepProb: 1.0})
            summary_writer.add_summary(summary, step)
            current_step = tf.train.global_step(sess, globalStep)
            print "%s step %d, training accuracy %g" % \
                  (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_step, train_accuracy)



    print '测试集1正确率'
    print sess.run(accuracy, feed_dict={x: testDataArray1, y: testLableArray1, keepProb: 1.0})
    print '测试集2正确率'
    print sess.run(accuracy, feed_dict={x: testDataArray2, y: testLableArray2, keepProb: 1.0})
    save_path = saver.save(sess, "/tmp/model_cnn.ckpt")
    print "Model saved in file: ", save_path