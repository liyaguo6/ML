import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#from tensorflow.examples.tutorials.mnist import input_data
import input_data


print ("Download and Extract MNIST dataset")
mnist = input_data.read_data_sets('data/', one_hot=True)
print (" tpye of 'mnist' is %s" % (type(mnist)))
print (" number of trian data is %d" % (mnist.train.num_examples))
print (" number of test data is %d" % (mnist.test.num_examples))

print ("What does the data of MNIST look like?")
trainimg   = mnist.train.images
trainlabel = mnist.train.labels
testimg    = mnist.test.images
testlabel  = mnist.test.labels
print (" type of 'trainimg' is %s"    % (type(trainimg)))
print (" type of 'trainlabel' is %s"  % (type(trainlabel)))
print (" type of 'testimg' is %s"     % (type(testimg)))
print (" type of 'testlabel' is %s"   % (type(testlabel)))
print (" shape of 'trainimg' is %s"   % (trainimg.shape,))
print (" shape of 'trainlabel' is %s" % (trainlabel.shape,))
print (" shape of 'testimg' is %s"    % (testimg.shape,))
print (" shape of 'testlabel' is %s"  % (testlabel.shape,))

# How does the training data look like?
print ("How does the training data look like?")
nsample = 2
# print(trainimg.shape[0]) #55000
randidx = np.random.randint(trainimg.shape[0], size=nsample)  #从55000个样本随机抽取5个样本
# print(randidx)
# for i in randidx:
#     curr_img   = np.reshape(trainimg[i, :], (28, 28)) # 28 by 28 matrix
# #     print(trainimg[i, :])
#     curr_label = np.argmax(trainlabel[i, :] ) # Label
# #     print(trainlabel[i, :])
#     plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
#     plt.title("" + str(i) + "th Training Data "
#               + "Label is " + str(curr_label))
#     print ("" + str(i) + "th Training Data "
#            + "Label is " + str(curr_label))
#     plt.show()
print ("Batch Learning? ")
batch_size = 100
batch_xs, batch_ys = mnist.train.next_batch(batch_size)
print ("type of 'batch_xs' is %s" % (type(batch_xs)))
print ("type of 'batch_ys' is %s" % (type(batch_ys)))
print ("shape of 'batch_xs' is %s" % (batch_xs.shape,))
print ("shape of 'batch_ys' is %s" % (batch_ys.shape,))