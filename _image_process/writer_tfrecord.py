import glob
import tensorflow as tf
import numpy as np
import os.path
from tensorflow.python.platform import gfile

# 原始输入数据的目录，这个目录下有5个子目录，每个子目录底下保存这属于该
# 类别的所有图片。
INPUT_DATA = r"D:\datasets\flower_photos"
# 测试数据和验证数据比例。
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10


# 读取数据并将数据分割成训练数据、验证数据和测试数据。
def create_image_lists(testing_percentage, validation_percentage):
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True

    # 初始化各个数据集。
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0

    # 读取所有的子目录。
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取一个子目录中所有的图片文件。
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue
        print("processing:", dir_name)

        i = 0
        # 处理图片数据。
        for file_name in file_list:
            i += 1
            # 读取并解析图片，将图片转化为299*299以方便inception-v3模型来处理。
            image_raw_data = gfile.FastGFile(file_name, 'rb').read()
            image = tf.image.decode_jpeg(image_raw_data)
            # 随机划分数据聚。
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(image)
                validation_labels.append(current_label)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(image)
                testing_labels.append(current_label)
            else:
                training_images.append(image)
                training_labels.append(current_label)
            if i % 200 == 0:
                print(i, "images processed.")
        current_label += 1

    # 将训练数据随机打乱以获得更好的训练效果。
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)

    return [training_images, training_labels,
            validation_images, validation_labels,
            testing_images, testing_labels]

#写成生成器的模式
def training_data_generator(data_list):
    training_images = data_list[0]
    n_training_example = len(training_images)
    training_labels = data_list[1]
    for i in range(n_training_example):
        yield training_images[i], training_labels[i]


def validation_data_generator(data_list):
    validation_images = data_list[2]
    n_validation_example = len(validation_images)
    validation_labels = data_list[3]
    for i in range(n_validation_example):
        yield validation_images[i], validation_labels[i]


def testing_data_generator(data_list):
    testing_images = data_list[4]
    n_testing_example = len(testing_images)
    testing_labels = data_list[5]
    for i in range(n_testing_example):
        yield testing_images[i], testing_labels[i]


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 将数据转化为tf.train.Example格式。
def _make_example(sess,label, image):
    image_shape= sess.run(image).shape
    image_value = sess.run(image).tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height':_int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'channel': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_value)
    }))
    return example


dataset = create_image_lists(10,10)
training_data_collections=training_data_generator(dataset)
validation_data_collections=validation_data_generator(dataset)
testing_data_collections=testing_data_generator(dataset)


num_shards = 4
instances_per_shard = 2000
with tf.Session() as sess:
    for i in range(num_shards):
        filename = ('./data/training_data.tfrecords-%.5d-of-%.5d' % (i, num_shards))
        # 将Example结构写入TFRecord文件。
        writer = tf.python_io.TFRecordWriter(filename)
        j =0
        for training_data in training_data_collections:
            if j >instances_per_shard:break
            else:
                example=_make_example(sess,image=training_data[0],label=training_data[1])
                j+=1
        # Example结构仅包含当前样例属于第几个文件以及是当前文件的第几个样本。
            writer.write(example.SerializeToString())
    print("TFRecord训练文件已保存。")
    with tf.python_io.TFRecordWriter("./data/validation.tfrecords") as writer:
        for validation_data in validation_data_collections:
            example = _make_example(sess,
             image=validation_data[0], label=validation_data[1])
            writer.write(example.SerializeToString())
    with tf.python_io.TFRecordWriter("./data/testing.tfrecords") as writer:
        for testing_data in testing_data_collections:
            example = _make_example(sess,
             image=testing_data[0], label=testing_data[1])
            writer.write(example.SerializeToString())