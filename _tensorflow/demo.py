import tensorflow as tf
import os
# w = tf.Variable([[0.5,1]])
# x = tf.Variable([[2.0],[1.0]])
# y = tf.matmul(w,x)
# print(y)
# y=tf.zeros([3, 4], "float32") #==> [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
# tensor=tf.ones([2, 3], "int32") #==> [[1, 1, 1], [1, 1, 1]]
# y=tf.ones_like(tensor)

# Constant 1-D Tensor populated with value list.
# y = tf.constant([1, 2, 3, 4, 5, 6, 7]) #=> [1 2 3 4 5 6 7]

# Constant 2-D tensor populated with scalar value -1.
# tensor = tf.constant(-1.0, shape=[2, 3]) #=> [[-1. -1. -1.]
#                                               #[-1. -1. -1.]]

# tensor=tf.linspace(10.0, 12.0, 5, name="linspace")
# tensor=tf.range(7, 10, 1) #==> [3, 6, 9, 12, 15]

# norm = tf.random_normal([1,100], mean=0, stddev=1)
# init_op = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init_op)
#     d=norm.eval().reshape(100,)

# import matplotlib.pyplot as plt
# plt.hist(d)
# plt.show()

# state = tf.Variable(3)
# new_value = tf.add(state, tf.constant(1))
# update = tf.assign(state, new_value)  #赋值操作
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(state))
#     for _ in range(3):
#         sess.run(update)
#         print(sess.run(state))


#ndarray转化为tensor类型
# import numpy as np
# a = np.zeros((3,3))
# ta = tf.convert_to_tensor(a)
# # print(ta)
# with tf.Session() as sess:
#      print(sess.run(ta))

# input1 = tf.placeholder(tf.float32)
# input2 = tf.placeholder(tf.float32)
# output = tf.matmul(input1,input2)
# with tf.Session() as sess:
#     print(sess.run([output], feed_dict={input1:[[7,2.3]], input2:[[2.3],[0]]}))