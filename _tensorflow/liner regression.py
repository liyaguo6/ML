import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 随机生成1000个点，围绕在y=0.1x+0.3的直线周围
num_points = 1000
vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]
# plt.scatter (x_data,y_data,c='r')
# plt.show()


# 生成1维的W矩阵，取值是[-1,1]之间的随机数

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')
b = tf.Variable(tf.zeros([1]), name='b')
# 经过计算得出预估值y
y = W * x_data + b
# print(x_data)
# print(y_data)
# 以预估值y和实际值y_data之间的均方误差作为损失
# 以预估值y和实际值y_data之间的均方误差作为损失
loss = tf.reduce_mean(tf.square(y - y_data), name='loss')
# 采用梯度下降法来优化参数
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 训练的过程就是最小化这个误差值
train = optimizer.minimize(loss, name='train')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # print(W.eval())
    print("W =", sess.run(W), "b =", sess.run(b), "loss =", sess.run(loss))
    for i in range(20):
        sess.run(train)
        print("W =", sess.run(W), "b =", sess.run(b), "loss =", sess.run(loss))
    plt.plot(x_data,sess.run(W)*x_data+sess.run(b),"r-")
    plt.scatter(x_data,y_data,c='y',marker='+')
    plt.show()