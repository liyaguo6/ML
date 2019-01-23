import tensorflow as tf
from image_process import preprocess_for_train
import numpy as np
import matplotlib.pyplot as plt
def build_input(data_path, batch_size,resize,mode=None):
    #读取一个文件夹下匹配的文件
    files = tf.train.match_filenames_once(data_path)
    #把文件放入文件队列中
    filename_queue = tf.train.string_input_producer(files, shuffle=True)
    #创建一个reader
    reader = tf.TFRecordReader()
    #从文件中读取一个样例。也可以使用read_up_to函数一次性读取多个样例
    _, serialized_example = reader.read(filename_queue)
    #解析一个样本
    features = tf.parse_single_example(
          serialized_example,
          features={
              'height':tf.FixedLenFeature([], tf.int64),
              'width':tf.FixedLenFeature([], tf.int64),
              'channel':tf.FixedLenFeature([], tf.int64),
              'label': tf.FixedLenFeature([], tf.int64),
              'image_raw': tf.FixedLenFeature([], tf.string),
          })
    # 组合样例中队列最多可以存储的样例个数
    capacity = 500+3*batch_size

    image,label = features['image_raw'],features['label']
    height,width,channel = features['height'],features['width'],features['channel']

    #原始图像数据解析出像素矩阵，并根据图像尺寸还原图像
    decoded_images=tf.decode_raw(image,tf.uint8)
    retyped_images = tf.cast(decoded_images, tf.float32)
    retyped_height = tf.cast(height,tf.int32)
    retyped_width = tf.cast(width,tf.int32)
    retyped_channel = tf.cast(channel,tf.int32)
    labels = tf.cast(label,tf.int32)

    if mode =='float':
        #把原始像素矩阵[0,255]转化为实数类型像素矩阵[0,1]
        # tf.reshape与set_shape的区别
        # decoded_images.set_shape([retyped_height,retyped_width ,retyped_channel])
        reshaped_images=tf.reshape(retyped_images,[retyped_height,retyped_width ,3])
        distorted_image = preprocess_for_train(reshaped_images,resize,resize,None)
        # 组合样例两种方法一种是tf.train.batch;另一种是tf.train.shuffle_batch，输入的shape一定要明确
        example_batch,label_batch = tf.train.shuffle_batch([distorted_image,labels],batch_size=batch_size,capacity=capacity,min_after_dequeue=500)

    else:
        # 把原始像素矩阵[0,255]

        reshaped_images=tf.reshape(retyped_images,[retyped_height,retyped_width ,3])

        distorted_image = tf.image.resize_images(reshaped_images, [resize, resize], method=0)
        # 组合样例两种方法一种是tf.train.batch;另一种是tf.train.shuffle_batch，输入的shape一定要明确
        example_batch,label_batch = tf.train.shuffle_batch([distorted_image,labels],batch_size=batch_size,capacity=capacity,min_after_dequeue=500)
    return example_batch,label_batch

batch_size=12
data_path = "./data/testing.tfrecords*"

# example_batch,label_batch=build_input(data_path,batch_size,resize=32,mode='float')
example_batch,label_batch=build_input(data_path,batch_size,resize=32,mode='int')
if __name__ == '__main__':

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord=tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        for i in range(2):
            cur_example_batch , cur_label_batch = sess.run(
                [example_batch ,label_batch]
            )
            print(cur_example_batch.shape,cur_label_batch.shape)
            print("######################")

        coord.request_stop()
        coord.join(threads)
