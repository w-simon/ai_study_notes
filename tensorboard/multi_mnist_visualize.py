# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
images = tf.Variable(mnist.test.images, name='images')

#参数设置
#learning_rate = 0.001
#learning_rate = 0.01
#learning_rate = 0.1
#learning_rate = 0.6
learning_rate = 0.8

training_epochs = 25
batch_size = 100
display_step = 1

LOG_DIR = "log_multi_mnist"
metadata = os.path.join(LOG_DIR, 'metadata.tsv')

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data 输入 (img shape: 28*28)
n_classes = 10  # MNIST 列别 (0-9 ，一共10类)

# tf Graph input
with tf.name_scope('input'):
    x = tf.placeholder("float", [None, n_input], name='input-x')
    y = tf.placeholder("float", [None, n_classes], name='input-y')

# tf Graph Input
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

# Create model
def multilayer_perceptron(x, weights, biases):
    
    with tf.name_scope('layer_1'):
        with tf.name_scope('weight'):
            W = weights['h1']
        with tf.name_scope('biases'):
            b = biases['b1']
        tf.summary.histogram('layer_1/weights', W)
        tf.summary.histogram('layer_1/biases', b)
        # Hidden layer with RELU activation
        #layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.add(tf.matmul(x, W), b)
        tf.summary.histogram('layer_1/pre_act', layer_1)
        layer_1 = tf.nn.relu(layer_1)
        tf.summary.histogram('layer_1/act', layer_1)

    with tf.name_scope('layer_2'):
        with tf.name_scope('weight'):
            W = weights['h2']
        with tf.name_scope('biases'):
            b = biases['b2']

        tf.summary.histogram('layer_2/weights', W)
        tf.summary.histogram('layer_2/biases', b)
        # Hidden layer with RELU activation
        #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.add(tf.matmul(layer_1, W), b)
        tf.summary.histogram('layer_2/pre_act', layer_2)
        layer_2 = tf.nn.relu(layer_2)
        tf.summary.histogram('layer_2/act', layer_2)
    
    with tf.name_scope('out_layer'):
        with tf.name_scope('weight'):
            W = weights['out']
        with tf.name_scope('biases'):
            b = biases['out']

        tf.summary.histogram('out_layer/weights', W)
        tf.summary.histogram('out_layer/biases', b)
        # Output layer with linear activation
        #out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        out_layer = tf.matmul(layer_2, W) + b
        tf.summary.histogram('out/value', out_layer)
    
    return out_layer
    
# Store layers weight & bias
weights = {
        'h1': tf.Variable(tf.random_normal( [n_input, n_hidden_1]), name='layer_1_weights'),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='layer_2_weights'),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name='out_layer_weights')
}
biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='layer_1_biases'),
        'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='layer_2_biases'),
        'out': tf.Variable(tf.random_normal([n_classes]), name='out_layer_biases')
}

# 构建模型
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    tf.summary.scalar('cost', cost)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with open(metadata, 'w') as metadata_file:
    for row in mnist.test.labels:
        metadata_file.write('%d\n' %  np.argmax(row, 0))

# 初始化变量
init = tf.global_variables_initializer()

merged = tf.summary.merge_all()

# 启动session
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
    
    sess.run(init)
    index = 0

    # 启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # 遍历全部数据集
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, summary = sess.run([optimizer, cost, merged], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
            
            if i % 50 == 0:
                index += 1
                summary_writer.add_summary(summary, index)

        # 显示训练中的详细信息
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print (" Finished!")

    # 测试 model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))



