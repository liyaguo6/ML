import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels
print("MNIST loaded")
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])  # None is for infinite
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax归一化处理
actv = tf.nn.softmax(tf.matmul(x, W) + b)
# 损失函数
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(actv), reduction_indices=1))
learning_rate = 0.01
# 采用梯度下降法来优化参数
optm = tf.train.GradientDescentOptimizer(learning_rate)
# 训练的过程就是最小化这个误差值
train = optm.minimize(cost)

# 预测
pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))  # 比较真实值与预测值之间的索引是否一样
accr = tf.reduce_mean(tf.cast(pred, 'float'))  # 把Ture转化为1，False转化为0

init = tf.global_variables_initializer()

training_epochs = 50
batch_size = 100
display_step = 5

sess = tf.Session()
sess.run(init)
# MINI-BATCH LEARNING
for epoch in range(training_epochs):
    avg_cost = 0.
    num_batch = int(mnist.train.num_examples / batch_size)
    for i in range(num_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={x: batch_xs, y: batch_ys})
        feeds = {x: batch_xs, y: batch_ys}
        avg_cost += sess.run(cost, feed_dict=feeds) / num_batch
    # DISPLAY
    if epoch % display_step == 0:
        feeds_train = {x: batch_xs, y: batch_ys}
        feeds_test = {x: mnist.test.images, y: mnist.test.labels}
        train_acc = sess.run(accr, feed_dict=feeds_train)
        test_acc = sess.run(accr, feed_dict=feeds_test)
        print("Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f"
              % (epoch, training_epochs, avg_cost, train_acc, test_acc))
print("DONE")


