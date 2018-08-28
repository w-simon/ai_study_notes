# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
images = tf.Variable(mnist.test.images, name='images')

#tf.reset_default_graph()

with tf.name_scope('input'):
    # mnist data维度 28*28=784
    x = tf.placeholder(tf.float32, [None, 784], name='input-x')
    # 0-9 numer => 10 classes
    y = tf.placeholder(tf.float32, [None, 10], name='input-y')

# tf Graph Input
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

with tf.name_scope('summaries'):
    # Set model weights
    W = tf.Variable(tf.random_normal([784, 10]))
    tf.summary.histogram('weights', W)
    b = tf.Variable(tf.zeros([10]))
    tf.summary.histogram('bias', b)

with tf.name_scope('Wx_plus_b'):
    # 构建模型
    pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax分类
    tf.summary.histogram('pred', pred)

with tf.name_scope('cost'):
    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
    tf.summary.scalar('cost', cost)

LOG_DIR = "log_simple_mnist"
metadata = os.path.join(LOG_DIR, 'metadata.tsv')
training_epochs = 40
batch_size = 100
display_step = 1

#参数设置
#learning_rate = 0.001
#learning_rate = 0.01
learning_rate = 0.3

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

with open(metadata, 'w') as metadata_file:
    for row in mnist.test.labels:
        metadata_file.write('%d\n' %  np.argmax(row, 0))

merged = tf.summary.merge_all()

# start session
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    # add embeddings.
    saver = tf.train.Saver([images])
    sess.run(images.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = "metadata.tsv"
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(summary_writer, config)

    # Initializing
    tf.global_variables_initializer().run()
    index = 0
    
    # start training
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, summary = sess.run([optimizer, cost, merged], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
           
            if i % 50 == 0:
                index += 1
                summary_writer.add_summary(summary, index)

            # 显示训练中的详细信息
        if (epoch+1) % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
         
    summary_writer.close()
    print("Finished!")
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

