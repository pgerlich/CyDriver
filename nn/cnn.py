print "Loading Vehicle data"

import input_data
data = input_data.read_data_sets('1456057393.21')

print "Configuring net"

import tensorflow as tf
import os.path
sess = tf.InteractiveSession()

#Initialize weights
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#Initialize biases
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#Setup convolution strides
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#Setup 2x2 max pooling
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#Our image vectors
x = tf.placeholder(tf.float32, shape=[None, 1024]) #Image vectors - size of pi input
y_ = tf.placeholder(tf.float32, shape=[None, 4]) #One-hot vectors - TODO: change to 4

print "Setting up Convolutional layers"

#Conv layer 1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,32,32,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Conv 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Conv 2
W_conv3 = weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

#Dense layer
W_fc1 = weight_variable([4 * 4 * 128, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

#Dropout to prevent overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Softmax readout layer
W_fc2 = weight_variable([1024, 4])
b_fc2 = bias_variable([4])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv + 1e-9))

with tf.name_scope("train") as scope:
  train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)

with tf.name_scope("test") as scope:
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  accuracy_summary = tf.scalar_summary("accuracy", accuracy)

merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/cydriver_logs", sess.graph_def)

sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()

print "Training network"

for i in range(1000):
  if i % 100 == 0:  # Record summary data, and the accuracy
    feed = {x: data.test.images, y_: data.test.labels, keep_prob: 1.0}
    result = sess.run([merged, accuracy], feed_dict=feed)
    summary_str = result[0]
    acc = result[1]
    writer.add_summary(summary_str, i)
    saver.save(sess, 'model.ckpt', global_step= i + 1)
    print("Accuracy at step %s: %s" % (i, acc))
  else:
    batch_xs, batch_ys = data.train.next_batch(5)
    feed = {x: batch_xs, y_: batch_ys, keep_prob: 0.5}
    sess.run(train_step, feed_dict=feed)


print("test accuracy %g"%accuracy.eval(feed_dict={
    x: data.test.images, y_: data.test.labels, keep_prob: 1.0}))

# for image in data.test.images:
#   image = image.reshape(1, image.shape[0])
#   feed_dict = {x: image, keep_prob: 1.0}
#   classification = sess.run(y_conv, feed_dict)
#   print "Classification", classification



save_path = saver.save(sess, "model.ckpt")
print("Model saved in file: %s" % save_path)