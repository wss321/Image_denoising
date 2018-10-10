import tensorflow as tf
import os
import numpy as np

file_dir = './data/fine'


def pares_tf(example_proto):
    features = {"id": tf.FixedLenFeature((), tf.int64),
                "data": tf.FixedLenFeature((256, 256), tf.float32),
                "label": tf.FixedLenFeature((), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["id"], tf.one_hot(parsed_features["label"] - 1, 5), parsed_features["data"]


def get_test_one_shot(shuffle_buffer_size=10000):
    file_path = [os.path.join(file_dir, 'TFcodeX_{}.tfrecord'.format(i)) for i in range(9, 11)]
    for i in file_path:
        if not os.path.exists(i):
            raise FileNotFoundError('{} not exist.'.format(i))
    pano = tf.data.TFRecordDataset([file_path])
    pano = pano.map(pares_tf)
    if shuffle_buffer_size:
        pano = pano.shuffle(buffer_size=shuffle_buffer_size)

    return pano.make_one_shot_iterator().get_next()


def get_train_one_shot(shuffle_buffer_size=10000):
    file_path = [os.path.join(file_dir, 'TFcodeX_{}.tfrecord'.format(i)) for i in range(1, 9)]
    for i in file_path:
        if not os.path.exists(i):
            raise FileNotFoundError('{} not exist.'.format(i))
    pano = tf.data.TFRecordDataset([file_path])
    pano = pano.map(pares_tf)
    if shuffle_buffer_size:
        pano = pano.shuffle(buffer_size=shuffle_buffer_size)

    return pano.make_one_shot_iterator().get_next()


def load_train_data_1C(shuffle_buffer_size=10000):
    test_iterator = get_train_one_shot(shuffle_buffer_size=shuffle_buffer_size)
    labels = []
    test_data = []
    with tf.Session() as sess:
        while True:
            try:
                id, one_hot_label, img = sess.run(test_iterator)
                labels.append(one_hot_label)
                test_data.append(img)
            except:
                break
    return np.asarray(test_data).reshape(-1, 256, 256, 1), np.asarray(labels)


def load_test_data_1C(shuffle_buffer_size=10000):
    test_iterator = get_test_one_shot(shuffle_buffer_size=shuffle_buffer_size)
    labels = []
    test_data = []
    with tf.Session() as sess:
        while True:
            try:
                id, one_hot_label, img = sess.run(test_iterator)
                labels.append(one_hot_label)
                test_data.append(img)
            except:
                break
    return np.asarray(test_data).reshape(-1, 256, 256, 1), np.asarray(labels)
