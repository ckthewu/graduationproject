# coding: utf-8
import json, random
import tensorflow as tf
from scripts.CONST import DATAPATH, KINDMAP, KINDLIST, FATHERPATH
def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def getBatch(xarr, yarr, size):
    indexarr = [x for x in range(len(xarr))]
    batchx = []
    batchy = []
    for i in range(size):
        index = indexarr.pop(random.randint(0, len(indexarr)-1))
        batchx.append(xarr[index])
        batchy.append(yarr[index])
    return (batchx, batchy)

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
    if len(dataArray) != len(lableArray) or len(lableArray[0]) != 5:
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
x = tf.placeholder("float", [None, L], name="input")

W = weight_variable([L, 5], name="weight")

b = bias_variable([5], name="bias")
y = tf.nn.softmax(tf.matmul(x, W) + b, name='output')
y_ = tf.placeholder("float", [None, 5], name='correct_output')
acrate = tf.Variable(0.0, name='acrate')



#交叉熵cost函数
NEAR_0 = 1e-10
cross_entropy = -tf.reduce_sum(y_ * tf.log(y + NEAR_0), name='loss')

variable_summaries(cross_entropy, 'loss')
variable_summaries(acrate, 'acrate')

train_step = tf.train.AdamOptimizer(5e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter("/tmp/model_logs_fc", sess.graph)
    sess.run(init)

    for i in range(10000):
        batch_xs, batch_ys = getBatch(tranDataArray, tranLableArray, 100)
        sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
        if i % 500 == 0:
            summary, acc = sess.run([summary_op, accuracy], feed_dict={x: testDataArray1, y_: testLableArray1})
            summary_writer.add_summary(summary, i)
            update = tf.assign(acrate, acc)
            sess.run(update)
            print('Accuracy at step %s: %s' % (i, acc))


    # print '训练集正确率'
    # print sess.run(accuracy, feed_dict={x: tranDataArray, y_: tranLableArray})
    # print '测试集正确率'
    # print sess.run(accuracy, feed_dict={x: testDataArray1, y_: testLableArray1})
    # print sess.run(accuracy, feed_dict={x: testDataArray2, y_: testLableArray2})
    line = ['fc', '测试集1',]
    print '训练集正确率'
    print sess.run(accuracy, feed_dict={x: tranDataArray, y_: tranLableArray})
    print '测试集1正确率'
    ac = str(sess.run(accuracy, feed_dict={x: testDataArray1, y_: testLableArray1}))
    print ac
    line.append(ac)
    for i in range(5):
        ac = str(sess.run(accuracy, feed_dict={x: testCategoryData1[i], y_: testCategoryLable1[i]}))
        print '%s类的正确率为%s' % (KINDMAP[KINDLIST[i]], str(ac))
        line.append(ac)
    with open(FATHERPATH + '/outtable.csv', 'a') as f:
        f.write(','.join(line) + '\n')
        line = ['fc', '测试集2',]
    print '测试集2正确率'
    ac = str(sess.run(accuracy, feed_dict={x: testDataArray2, y_: testLableArray2}))
    print ac
    line.append(ac)
    for i in range(5):
        ac = str(sess.run(accuracy, feed_dict={x: testCategoryData2[i], y_: testCategoryLable2[i]}))
        print '%s类的正确率为%s' % (KINDMAP[KINDLIST[i]], str(ac))
        line.append(ac)
    with open(FATHERPATH + '/outtable.csv', 'a') as f:
        f.write(','.join(line) + '\n')
    # save_path = saver.save(sess, "/tmp/model.ckpt")
    # print "Model saved in file: ", save_path
