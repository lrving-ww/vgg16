import os
import glob
from skimage import io, transform
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from sklearn.utils import shuffle
import cv2
import random
from datetime import datetime
 
from modelCFG import *
from model import *
from alexnet import *
 
 
def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            print('reading the image: %s' % (im))
            img = io.imread(im)
            img = transform.resize(img, model_CFG['image_size'])
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asanyarray(labels, np.int32)
 
 
def read_img_gen(path):
 
    image = cv2.imread(path)
    image = cv2.resize(image, (model_CFG['IMAGE_WIDTH'], model_CFG['IMAGE_HEIGHT']))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    normalizer_image = image / 255.0 - 0.5
 
    return normalizer_image
 
 
def generator(image_size, path, batch_size=model_CFG['batch_size'], is_test=model_CFG['is_test']):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
 
    # 1. create list of the image paths
    image_path_list = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            image_path_list.append((im, idx))
 
    random.shuffle(image_path_list)
    num_samples = len(image_path_list) if not is_test else model_CFG['test_size']
 
    # 2. fetch the data
    for offset in range(0, num_samples, batch_size):
        batch_samples = image_path_list[offset:(offset + batch_size)]
        batch_images = []
        batch_labels = []
        for batch_sample in batch_samples:
            img = read_img_gen(batch_sample[0])
            batch_images.append(img)
            label = one_hot_encoder(batch_sample[1], model_CFG['n_classes'])
            batch_labels.append(label)
 
        x = np.reshape(np.array(batch_images), (-1, 224, 224, 3))
        y = np.reshape(np.array(batch_labels), (-1, model_CFG['n_classes']))
        yield x, y
 
 
def one_hot_encoder(val, nclasses):
    label = np.zeros(nclasses)
    label[int(val)] = 1
    return label
 
 
def generator_test_data(path=model_CFG['test_path']):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
 
    # 1. create list of the image paths
    image_path_list = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            image_path_list.append((im, idx))
 
    random.shuffle(image_path_list)
 
    images = []
    labels = []
    print('Explored here!')
    for sample in image_path_list[:127]:
        img = read_img_gen(sample[0])
        images.append(img)
        label = one_hot_encoder(sample[1], model_CFG['n_classes'])
        labels.append(label)
 
    x = np.reshape(np.array(images), (-1, 224, 224, 3))
    y = np.reshape(np.array(labels), (-1, model_CFG['n_classes']))
    return x, y
 
 
def train(path, keep_prob=model_CFG['keep_prob'], n_classes=model_CFG['n_classes']):
    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input')
    y = tf.placeholder(tf.int64, shape=[None, n_classes], name='labels')
 
    x_test, y_test = generator_test_data()
 
    # Alexnet
#     output = alexnet(x, keep_prob, n_classes)
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
 
    # VGG16
    output = vgg16(x, keep_prob, n_classes)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
 
    train_op = tf.train.AdamOptimizer(learning_rate=model_CFG['learning_rate']).minimize(loss)
    correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    correct = tf.cast(correct, tf.float32)
    accuracy = tf.reduce_mean(correct)
 
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init)
 
        for epoch_index in range(model_CFG['num_epochs']):
            print('epoch :', epoch_index+1)
            step = 0
            for X_batch, y_batch in generator(model_CFG['image_size'], path=path, batch_size=model_CFG['batch_size']):
                feed_dict = {x: X_batch, y: y_batch}
                _, train_loss = sess.run([train_op, loss], feed_dict=feed_dict)
                step += 1
                print("Step [%d], training loss :  %g" % (step, train_loss))
 
            val_loss, val_accuracy = sess.run([loss, accuracy], feed_dict={x:x_test, y:y_test})
            print("Epoch [%d], valuation loss :  %g, valuation accuracy: %g" % (epoch_index, val_loss, val_accuracy))
 
        saver.save(sess, 'save_model/vgg16_model.ckpt', global_step=step)
 
 
def test(path, keep_prob=1):
    x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input')
    output = vgg16(x, keep_prob, model_CFG['n_classes'])
    score = tf.nn.softmax(output)
    f_cls = tf.argmax(score, 1)
 
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, 'save_model/vgg16_model.ckpt-100')
 
    images, labels = read_img(path)
    for i, img in enumerate(images):
        img = transform.resize(img, (1, 224, 224, 3))
        pred, _score = sess.run([f_cls, score], feed_dict={x: img})
        prob = round(np.max(_score), 4)
 
        print('{} animal class is: {}, score:{}. The prediction is {}'.format(i, int(pred), prob, int(pred)==labels[i]))
        # print('{} animal ground truth class is: {}'.format(i, labels[i]))
 
    sess.close()