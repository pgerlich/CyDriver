import tensorflow as tf
import os.path
import urllib
import sys, tty, termios
import numpy 
import cv2

#Reset up the NN
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
x = tf.placeholder(tf.float32, shape=[None, 10000]) #Image vectors - size of pi input
y_ = tf.placeholder(tf.float32, shape=[None, 3]) #One-hot vectors - TODO: change to 4

W = tf.Variable(tf.zeros([10000,3])) #Weights - change to 4
b = tf.Variable(tf.zeros([3])) #Biases - change to 4

#Add label to graph nodes
y = tf.nn.softmax(tf.matmul(x,W) + b) #Softmax label for prediction

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

print "Setting up Convolutional layers"

#Conv layer 1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,100,100,1])

#Max pooling and stuff
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Conv 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Dense layer
W_fc1 = weight_variable([25 * 25 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 25 * 25 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout to prevent overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Softmax readout layer
W_fc2 = weight_variable([1024, 3])
b_fc2 = bias_variable([3])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

import socket
import sys

# # Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#Connect the socket to the port where the server is listening
server_address = ('192.168.0.1', 10000)
print >>sys.stderr, 'connecting to %s port %s' % server_address
sock.connect(server_address)

if os.path.isfile("model.ckpt"):
	ckpt = tf.train.get_checkpoint_state(".")

	print "Model found. Loading..."

	#load NN
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)

		print "Loaded model succesfully"
	else:
		print "No model loaded"
else:
	print "No trained model, exiting"
	sys.exit(1)

def convertModeToVal(mode):
	if mode == 0:
		return "FWD"
	if mode == 1:
		return "LEFT"
	if mode == 2:
		return "RIGHT"

def convertVectorToLabel(vec):
	vec = vec.tolist()
	maxVal = max(vec)
	return vec.index(maxVal)


def labelVideoInput():
	print "Searching for stream"
	last38 = []
	labelIndex = 0

	#Stream model
	stream=urllib.urlopen('http://192.168.0.1:8080/?action=stream')
	bytes=''

	print "Connected to stream"

	while(True):
		#Load from stream
		bytes+=stream.read(22500)
		a = bytes.find('\xff\xd8')
		b = bytes.find('\xff\xd9')

		if a!=-1 and b!=-1:
			jpg = bytes[a:b+2]
			bytes= bytes[b+2:]

			i = cv2.imdecode(numpy.fromstring(jpg, dtype=numpy.uint8),cv2.COLOR_BGR2GRAY)

			# Our operations on the frame come here
			gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
			imgBeforeFlatten = cv2.resize(gray, (100, 100))
			img = imgBeforeFlatten.reshape(1, imgBeforeFlatten.shape[0] * imgBeforeFlatten.shape[1])

			#Classify image with NN
			feed_dict = {x: img, keep_prob: 1.0}
			classification = sess.run(y, feed_dict)

			#print classification
			label = convertVectorToLabel(classification)

			#Populate at first, treat like queue there after
			if len(last38) <= 37:
				last38.append(label)
			else:
				last38[labelIndex % 38] = label #Fill up last38 in FIFO order

			labelIndex = labelIndex + 1

			currentMode = max(set(last38), key=last38.count) #Most occurin in last 38 samples

			if len(last38) >= 38:
				print currentMode
				sock.sendall(convertModeToVal(currentMode))

			# Display the resulting frame
			cv2.imshow('frame', imgBeforeFlatten)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

labelVideoInput()
