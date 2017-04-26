# coding: utf-8
import json, random
from scripts.CONST import DATAPATH
import tensorflow as tf
def weight_variable(shape, name = None):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name = None):
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

L = len(testDataArray1[0])
print L
x = tf.placeholder("float", [None, L])
y_ = tf.placeholder("float", [None, 5], name='correct_output')

# 一个隐藏层 L个神经元全连接
W1 = weight_variable([L, L])
b1 = bias_variable([L])
y1 = tf.nn.softmax(tf.matmul(x, W1) + b1)

W2 = weight_variable((L, 5))
b2 = bias_variable([5])
y = tf.nn.softmax(tf.matmul(y1, W2) + b2, name='output')



# 正确率变量
acrate = tf.Variable(0.0, name='acrate')



#交叉熵cost函数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y), name='loss')

variable_summaries(cross_entropy, 'loss')
variable_summaries(acrate, 'acrate')

train_step = tf.train.AdamOptimizer(5e-2).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
summary_op = tf.summary.merge_all()

with tf.Session() as sess:

    summary_writer = tf.summary.FileWriter("/tmp/model_logs_fc1", sess.graph)
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = getBatch(tranDataArray, tranLableArray, 100)
        sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
        if i % 50 == 0:
            summary, acc = sess.run([summary_op, accuracy], feed_dict={x: testDataArray1, y_: testLableArray1})
            summary_writer.add_summary(summary, i)
            update = tf.assign(acrate, acc)
            sess.run(update)
            print('Accuracy at step %s: %s' % (i, acc))


    print '训练集正确率'
    print sess.run(accuracy, feed_dict={x: tranDataArray, y_: tranLableArray})
    print '测试集正确率'
    print sess.run(accuracy, feed_dict={x: testDataArray1, y_: testLableArray1})
    print sess.run(accuracy, feed_dict={x: testDataArray2, y_: testLableArray2})
    # save_path = saver.save(sess, "/tmp/model.ckpt")
    # print "Model saved in file: ", save_path
