# coding: utf-8
import json, numpy, random, math, datetime
import tensorflow as tf
def weight_variable(shape, name = None):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name = None):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    # 用pooling对付跨步大丢失信息问题
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


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


originLength = 100
newSize = 10
addArray = [0 for x in range(newSize*newSize - originLength)]
tranDataArray = []
tranLableArray = []
testDataArray = []
testLableArray = []
skipSize = 1
with open('jieba/trandata_wv.json', 'r') as f:
    i = 0
    for l in f.readlines():
        if i % skipSize == 0:
            newArray = json.loads(l.rstrip("\n"))
            newArray.extend(addArray)
            tranDataArray.append(newArray)
        i += 1

    print 'ok'
    print "trandata size " + str(len(tranDataArray))

with open('jieba/tranlable_wv.json', 'r') as f:
    i = 0
    for l in f.readlines():
        if i % skipSize == 0:
            tranLableArray.append(json.loads(l.rstrip("\n")))
        i += 1

    print 'ok'
    print "tranlable size " + str(len(tranLableArray))

with open('jieba/testdata1_wv.json', 'r') as f:
    i = 0
    for l in f.readlines():
        if i % skipSize == 0:
            newArray = json.loads(l.rstrip("\n"))
            newArray.extend(addArray)
            testDataArray.append(newArray)
        i += 1

    print 'ok'
    print "testdata size " + str(len(testDataArray))

with open('jieba/testlable1_wv.json', 'r') as f:
    i = 0
    for l in f.readlines():
        if i % skipSize == 0:
            testLableArray.append(json.loads(l.rstrip("\n")))
        i += 1

    print 'ok'
    print "testlable size " + str(len(testLableArray))



#input
x = tf.placeholder(tf.float32, shape=[None, newSize*newSize])
x_image = tf.reshape(x, [-1,newSize,newSize,1])
y = tf.placeholder(tf.float32, shape=[None, 5])

#convolution 1
W_conv1 = weight_variable([2, 2, 1, 32])
b_conv1 = bias_variable([32])
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_conv1 = conv2d(x_image, W_conv1) + b_conv1
#h_pool1 = max_pool_2x2(h_conv1)

#convolution 2
W_conv2 = weight_variable([2, 2, 32, 64])
b_conv2 = bias_variable([64])
# h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_conv2 = conv2d(h_conv1, W_conv2) + b_conv2
#h_pool2 = max_pool_2x2(h_conv2)

#full-connected 1
W_fc1 = weight_variable([newSize* newSize * 64, 1500])
b_fc1 = bias_variable([1500])
h_conv2_flat = tf.reshape(h_conv2, [-1, newSize* newSize*64])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
#dropout
keep_prob1 = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob1)

#full-connected 2
W_fc2 = weight_variable([1500, 1000])
b_fc2 = bias_variable([1000])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#dropout
keep_prob2 = tf.placeholder(tf.float32)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob2)

#full-connected 3
W_fc3 = weight_variable([1000, 500])
b_fc3 = bias_variable([500])
h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
#dropout
keep_prob3 = tf.placeholder(tf.float32)
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob3)

#readout
W_out = weight_variable([500, 5])
b_out = bias_variable([5])
y_out = tf.matmul(h_fc3_drop, W_out) + b_out


#交叉熵cost函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_out), reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-1).minimize(cross_entropy)

sess = tf.Session()
# important step
sess.run(tf.global_variables_initializer())

# 正式计算

for i in range(1):
    batch_xs, batch_ys = getBatch(tranDataArray, tranLableArray, 1)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob1: 0.3, keep_prob2: 0.3, keep_prob3: 0.3})
    if i % 10 == 0:
        print(compute_accuracy(testDataArray, testLableArray))
