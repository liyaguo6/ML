import numpy as np
encoder_state = np.random.rand(3,5,20)
print(encoder_state)

import tensorflow as tf
with tf.Session() as sess:
    print(sess.run(tf.random_uniform((3,4,10))))