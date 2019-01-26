#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 19:47:51 2019

@author: shinong
"""

import tensorflow as tf
import numpy as np
from datetime import datetime
import cv2
import os
from tensorflow.examples.tutorials.mnist import input_data
#定义网络参数
lr = 0.001
batch_size = 64
display_step = 5
epochs = 10
#keep_prob = 0.5
n_cls=18
#max_steps=50000
max_steps=10
#定义卷积操作
def conv_op(input_op, name, kh, kw, n_out, dh, dw):
    input_op = tf.convert_to_tensor(input_op)
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                shape = [kh, kw, n_in, n_out],
                                dtype = tf.float32,
                                initializer = tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding = 'SAME')
        bias_init_val = tf.constant(0.0, shape = [n_out], dtype = tf.float32)
        biases = tf.Variable(bias_init_val, trainable = True, name = 'b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name = scope)
        return activation
 
#定义全连接操作
def fc_op(input_op, name, n_out):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',
                                shape = [n_in, n_out],
                                dtype = tf.float32,
                                initializer = tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape = [n_out], dtype = tf.float32), name = 'b')
        # tf.nn.relu_layer对输入变量input_op与kernel做矩阵乘法加上bias，再做RELU非线性变换得到activation
        activation = tf.nn.relu_layer(input_op, kernel, biases, name = scope) 
        return activation
    
#定义池化层
def mpool_op(input_op, name, kh, kw, dh, dw):
    return  tf.nn.max_pool(input_op,
                           ksize = [1, kh, kw, 1],
                           strides = [1, dh, dw, 1],
                           padding = 'SAME',
                           name = name)
def inference_op(input_op, keep_prob,n_cls):
    # block 1 -- outputs 112x112x64
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1)
    conv1_2 = conv_op(conv1_1,  name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1)
    pool1 = mpool_op(conv1_2,   name="pool1",   kh=2, kw=2, dw=2, dh=2)
 
    # block 2 -- outputs 56x56x128
    conv2_1 = conv_op(pool1,    name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1)
    conv2_2 = conv_op(conv2_1,  name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1)
    pool2 = mpool_op(conv2_2,   name="pool2",   kh=2, kw=2, dh=2, dw=2)
 
    # # block 3 -- outputs 28x28x256
    conv3_1 = conv_op(pool2,    name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1)
    conv3_2 = conv_op(conv3_1,  name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1)
    conv3_3 = conv_op(conv3_2,  name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1)    
    pool3 = mpool_op(conv3_3,   name="pool3",   kh=2, kw=2, dh=2, dw=2)
 
    # block 4 -- outputs 14x14x512
    conv4_1 = conv_op(pool3,    name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1)
    conv4_2 = conv_op(conv4_1,  name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
    conv4_3 = conv_op(conv4_2,  name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1)
    pool4 = mpool_op(conv4_3,   name="pool4",   kh=2, kw=2, dh=2, dw=2)
 
    # block 5 -- outputs 7x7x512
    conv5_1 = conv_op(pool4,    name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1)
    conv5_2 = conv_op(conv5_1,  name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
    conv5_3 = conv_op(conv5_2,  name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1)
    pool5 = mpool_op(conv5_3,   name="pool5",   kh=2, kw=2, dw=2, dh=2)
 
    # flatten
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")
 
    # fully connected
    fc6 = fc_op(resh1, name="fc6", n_out=4096)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")
 
    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")
 
    logits = fc_op(fc7_drop, name="fc8", n_out=n_cls)
    return logits
def inference_op2(input_op, keep_prob,n_cls):
    # block 1 -- outputs 112x112x64
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=32, dh=1, dw=1)
    conv1_2 = conv_op(conv1_1,  name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1)
    pool1 = mpool_op(conv1_2,   name="pool1",   kh=2, kw=2, dw=2, dh=2)
 
    # block 2 -- outputs 56x56x128
    conv2_1 = conv_op(pool1,    name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1)
    conv2_2 = conv_op(conv2_1,  name="conv2_2", kh=3, kw=3, n_out=256, dh=1, dw=1)
    pool2 = mpool_op(conv2_2,   name="pool2",   kh=2, kw=2, dh=2, dw=2)
 
#    # # block 3 -- outputs 28x28x256
#    conv3_1 = conv_op(pool2,    name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1)
#    conv3_2 = conv_op(conv3_1,  name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1)
#    conv3_3 = conv_op(conv3_2,  name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1)    
#    pool3 = mpool_op(conv3_3,   name="pool3",   kh=2, kw=2, dh=2, dw=2)
 
#    # block 4 -- outputs 14x14x512
#    conv4_1 = conv_op(pool3,    name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1)
#    conv4_2 = conv_op(conv4_1,  name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
#    conv4_3 = conv_op(conv4_2,  name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1)
#    pool4 = mpool_op(conv4_3,   name="pool4",   kh=2, kw=2, dh=2, dw=2)
# 
#    # block 5 -- outputs 7x7x512
#    conv5_1 = conv_op(pool4,    name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1)
#    conv5_2 = conv_op(conv5_1,  name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
#    conv5_3 = conv_op(conv5_2,  name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1)
#    pool5 = mpool_op(conv5_3,   name="pool5",   kh=2, kw=2, dw=2, dh=2)
 
    # flatten
    shp = pool2.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool2, [-1, flattened_shape], name="resh1")
 
    # fully connected
    fc6 = fc_op(resh1, name="fc6", n_out=512)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")
 
#    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096)
#    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")
 
    logits = fc_op(fc6_drop, name="fc8", n_out=n_cls)
    return logits
def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
 
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    # 转换为float32类型，并做归一化处理
    img = tf.cast(img, tf.float32)# * (1. / 255)
    label = tf.cast(features['label'], tf.int64)
    return img, label
def train():
    batch_size = 128
    test_size = 256
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1
    teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1

    X = tf.placeholder("float", [None, 28, 28, 1])
    Y = tf.placeholder("float", [None, 10]) 
    keep_prob = tf.placeholder("float")
    output = inference_op2(X, keep_prob, 10)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
#    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    train_op = tf.train.AdadeltaOptimizer(0.001, 0.95).minimize(cost)
    predict_op = tf.argmax(output, 1)
    saver = tf.train.Saver()  
    with tf.Session() as sess:
    # you need to initialize all variabels
        tf.global_variables_initializer().run()
        
        for i in range(100):
            training_batch=zip(range(0,len(trX),batch_size),
                       range(batch_size,len(trX)+1,batch_size))
            for start, end in training_batch:
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          keep_prob: 0.5})
                                       
            test_indices = np.arange(len(teX)) # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]
        
            print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                        sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                       keep_prob: 1.0})))

        saver.save(sess, './model_mnist/model.ckpt', global_step=i)

#    x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input')
#    y = tf.placeholder(dtype=tf.float32, shape=[None, n_cls], name='label')
#    keep_prob = tf.placeholder(tf.float32)
#    output = inference_op(x, keep_prob, n_cls)
#    #probs = tf.nn.softmax(output)
# 
#    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
#    #train_step = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
#    train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
# 
#    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output,1), tf.argmax(y, 1)), tf.float32))
#            _, loss_val = sess.run([train_step, loss], feed_dict={x:batch_x, y:batch_y, keep_prob:0.8})
def test():
    batch_size = 128
    test_size = 256
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1
    teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1
    X = tf.placeholder("float", [None, 28, 28, 1])
    Y = tf.placeholder("float", [None, 10]) 
  
    keep_prob = tf.placeholder(tf.float32)
    logits =  inference_op2(X,keep_prob,10)              
#    pred = tf.argmax(logits,axis=1)  
#    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits ,1), tf.argmax(Y, 1)), tf.float32))
    predict_op = tf.argmax(logits, 1)           
    saver = tf.train.Saver()   
    with tf.Session() as sess:      
        saver.restore(sess, './model_mnist/model.ckpt-99')                            
        for i in range(50):
            
            test_indices = np.arange(len(teX)) # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]
        
            print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                        sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                       keep_prob: 1.0})))
if __name__=='__main__':
  #  tf.reset_default_graph()
#    test_dir='/home/shinong/Desktop/dacheqian/'            
#    model='/home/shinong/Desktop/model/'
##    test(test_dir,model)
#    train()
    test()
