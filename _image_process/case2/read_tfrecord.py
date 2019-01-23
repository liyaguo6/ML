import tensorflow as tf
from image_process import preprocess_for_train
import numpy as np
import matplotlib.pyplot as plt
"""
读取数据
"""
def build_input(data_path, batch_size,num_class,reszie,mode='train'):
    #读取一个文件夹下匹配的文件
    files = tf.train.match_filenames_once(data_path)
    #把文件放入文件队列中
    filename_queue = tf.train.string_input_producer(files, shuffle=True)
    # #创建一个reader，
    reader = tf.TFRecordReader()
    # #从文件中读取一个样例。也可以使用read_up_to函数一次性读取多个样例
    _, serialized_example = reader.read(filename_queue)
    # #解析一个样本
    features = tf.parse_single_example(
          serialized_example,
        features={
            "image/encoded": tf.FixedLenFeature([], tf.string),
            "image/height": tf.FixedLenFeature([], tf.int64),
            "image/width": tf.FixedLenFeature([], tf.int64),
            "image/filename": tf.FixedLenFeature([], tf.string),
            "image/class/label": tf.FixedLenFeature([], tf.int64),
            'image/channels': tf.FixedLenFeature([], tf.int64),
        }

        )

    # 组合样例中队列最多可以存储的样例个数
    capacity = 500+3*batch_size
    #读取一个样例中的特征
    image,label = features['image/encoded'],features['image/class/label']
    height,width,channel = features['image/height'],features['image/width'],features['image/channels']




    # #tf.decode_raw 可以将字符串解析成图像对应的像素数组
    # decoded_images=tf.decode_raw(image,tf.uint8)
    # retyped_images = tf.cast(decoded_images, tf.float32)
    # retyped_height = tf.cast(height,tf.int32)
    # retyped_width = tf.cast(width,tf.int32)
    # retyped_channel = tf.cast(channel,tf.int32)
    # labels = tf.cast(label,tf.int32)
    # decoded_images.set_shape([height,width,channel])
    # resize =32
    # reshaped_images=tf.reshape(decoded_images,[retyped_height,retyped_width ,retyped_channel])
    # distored_image = tf.image.resize_images(decoded_images,[32,32],method=np.random.randint(4))
    # distored_image = preprocess_for_train(reshaped_images,resize,resize,None)
    # images,labels= tf.train.shuffle_batch([distored_image,labels ],batch_size=batch_size,capacity=capacity,min_after_dequeue=500)
    # return images,labels


    ###tf.image.decode_jpeg#############
    image_raw = tf.image.decode_jpeg(image, channels=3)
    retyped_height = tf.cast(height, tf.int32)
    retyped_width = tf.cast(width,tf.int32)
    retyped_channel = tf.cast(channel,tf.int32)
    labels = tf.cast(label,tf.int32)
    # image_resize = tf.image.resize_images(image_raw,[32,32],method=np.random.randint(4))
    image_resize=tf.image.resize_image_with_crop_or_pad(image_raw, reszie, reszie)
    images,labels= tf.train.shuffle_batch([image_resize,labels ],batch_size=batch_size,capacity=capacity,min_after_dequeue=500)
    labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    labels = tf.sparse_to_dense(
        tf.concat(values=[indices, labels], axis=1),
        [batch_size, num_class], 1.0, 0.0)
    tf.summary.image('images', images)
    return images,labels



data_path ='./data/train-*'
example_batch, label_batch=build_input(data_path, batch_size=32, num_class=4,reszie=128)

def show(image):
    plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        for i in range(1):
            cur_example_batch , cur_label_batch = sess.run([example_batch ,label_batch])
            print(cur_example_batch,cur_label_batch)
            for image in cur_example_batch:
                show(image)
        coord.request_stop()
        coord.join(threads)


