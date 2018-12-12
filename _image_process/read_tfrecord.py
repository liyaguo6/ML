import tensorflow as tf
from image_process import preprocess_for_train
files = tf.train.match_filenames_once("./data/testing.tfrecords*")
filename_queue = tf.train.string_input_producer(files, shuffle=True)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
      serialized_example,
      features={
          'height':tf.FixedLenFeature([], tf.int64),
          'width':tf.FixedLenFeature([], tf.int64),
          'channel':tf.FixedLenFeature([], tf.int64),
          'label': tf.FixedLenFeature([], tf.int64),
          'image_raw': tf.FixedLenFeature([], tf.string),
      })
batch_size=64
capacity = 500+3*batch_size
image,label = features['image_raw'],features['label']
height,width,channel = features['height'],features['width'],features['channel']

decoded_images=tf.decode_raw(image,tf.uint8)
retyped_images = tf.cast(decoded_images, tf.float32)
retyped_height = tf.cast(height,tf.int32)
retyped_width = tf.cast(width,tf.int32)
retyped_channel = tf.cast(channel,tf.int32)
labels = tf.cast(label,tf.int32)
# decoded_images.set_shape([retyped_height,retyped_width ,retyped_channel])
reshaped_images=tf.reshape(retyped_images,[retyped_height,retyped_width ,retyped_channel])
resize = 50
distored_image = preprocess_for_train(reshaped_images,resize,resize,None)
example_batch,label_batch = tf.train.shuffle_batch([distored_image,labels],batch_size=batch_size,capacity=capacity,min_after_dequeue=500)
# #
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord=tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    for i in range(2):
        cur_example_batch , cur_label_batch = sess.run(
            [example_batch ,label_batch]
        )
        print(cur_example_batch,cur_label_batch)
    coord.request_stop()
    coord.join(threads)
