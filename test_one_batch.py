#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:14:06 2019

@author: shinong
"""

import tensorflow as tf
import numpy as np
import os
import time
import cv2
from train_vgg16 import *
import collections
def decorate(func):
    def call_back(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        print('[INFO]:此次操作耗时<{:.2f}>秒'.format(end - start))
        return ret

    return call_back
def load_img(path):
    img=cv2.imread(path)
    img = img / 255.0
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = cv2.resize(crop_img, (224, 224))  
    return resized_img

def load_data(data_dir):
    contents = os.listdir(data_dir)
    classes = [each for each in contents if os.path.isdir(data_dir + each)]
#    batch_size = 20
    batch=[]
    labels=[]
#    image=[]
    for each in classes:
        print("Starting {} images".format(each))
        class_path = data_dir + each
        files = os.listdir(class_path)
        for ii, file in enumerate(files, 1):
            
            img = load_img(os.path.join(class_path, file))
            batch.append(img.reshape((1, 224, 224, 3)))
            labels.append(each)
#            if ii % batch_size == 0 or ii == len(files):
#                image = np.concatenate(batch)
#                batch=[]
    return batch,labels
#@decorate
def load_data2(data_dir): 
    batch=[]
#    labels=[]
    print("Starting {} images".format(data_dir))
    files = os.listdir(data_dir)
    for ii, file in enumerate(files, 1):        
        img = load_img(os.path.join(data_dir, file))
        batch.append(img.reshape((1, 224, 224, 3)))
#        labels.append(each)
    return batch
def test():
    
#    image_set=load_data2('/home/shinong/Desktop/validation/3')
    image_set=load_data2('/media/shinong/study/Resnet/data/validation/通泉草782')
    X_input = tf.placeholder("float", [None, 224, 224, 3])
#    Y = tf.placeholder("float", [None, 10]) 
    keep_prob = tf.placeholder("float")
    output = inference_op(X_input, keep_prob, 10)
#    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
#    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
#    train_op = tf.train.AdadeltaOptimizer(0.001, 0.95).minimize(cost)
#    train_op =tf.train.GradientDescentOptimizer(0.005).minimize(cost)
    predict_op = tf.argmax(output, 1)
#    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output,1), tf.argmax(Y, 1)), tf.float32))
    saver = tf.train.Saver()  
    with tf.Session() as sess:
        saver.restore(sess, './model_mnist/model.ckpt-19999')  
#        for batch in range(len(image_set),64):
        batch1=np.concatenate(image_set[:])
        pred=sess.run(predict_op,feed_dict={X_input: batch1,
                                                       keep_prob: 1.0})
        
        print('[INFO]:gt:<0> ---> out:<{}>'.format(pred)) 
#            start_time = time.time()
        
        b = collections.Counter(pred)
        print(b)
#        print(b,len(pred),(b[0]/len(pred))*100)
    v=[]
    for k1,v1 in b.items():
        v.append(v1)
    print(max(v)/len(pred))    
           
if __name__=='__main__':
    test()