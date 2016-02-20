print "Loading Vehicle data"

import input_data
data = input_data.read_data_sets('1455953378.16')

print "Configuring net"

import tensorflow as tf
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
x = tf.placeholder(tf.float32, shape=[None, 10000]) #Image vectors - size of pi input
y_ = tf.placeholder(tf.float32, shape=[None, 3]) #One-hot vectors - TODO: change to 4

W = tf.Variable(tf.zeros([10000,3])) #Weights - change to 4
b = tf.Variable(tf.zeros([3])) #Biases - change to 4

sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x,W) + b) #Softmax label for prediction

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

print "Setting up Convolutional layers"

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(5000):
  batch = data.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: data.test.images, y_: data.test.labels}))


# #First conv layer - 5x5 patches w/ 32 features
# W_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = bias_variable([32])

# x_image = tf.reshape(x, [-1,100,100,1])

# #Apply conv layer, max pool to reduce size
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)

# #Second conv layer, 5x5 w/ 32 features
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])

# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)

# #Dense conv layer
# W_fc1 = weight_variable([5*5*80, 10000])
# b_fc1 = bias_variable([10000])

# h_pool2_flat = tf.reshape(h_pool2, [-1, 5*5*80])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# #Dropout layer to prevent overfitting
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# #Softmax layer
# W_fc2 = weight_variable([10000, 3])
# b_fc2 = bias_variable([3])

# y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# print "Training network"

# cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv)) #Cross entropy reduction
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #Not sure?
# correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) 
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# sess.run(tf.initialize_all_variables())

# for i in range(20000):
#   batch = data.train.next_batch(50)

#   if i%100 == 0:
#     train_accuracy = accuracy.eval(feed_dict={
#         x: batch[0], y_: batch[1], keep_prob: 1.0})
#     print("step %d, training accuracy %g"%(i, train_accuracy))

#     print "happened"

#   train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# print("test accuracy %g"%accuracy.eval(feed_dict={
#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
