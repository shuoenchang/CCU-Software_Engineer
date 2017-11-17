from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])   # x is the value for each pixel
W = tf.Variable(tf.zeros([784, 10]))    # Weight for each pixel
b = tf.Variable(tf.zeros([10]))         # bias
y = tf.nn.softmax(tf.matmul(x, W) + b)        # y = softmax(Wx+b)
y_ = tf.placeholder(tf.float32, [None, 10])   # y' for the real label
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),           # cross_entropy as
                                              reduction_indices=[1]))   # loss function
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # use GradientDescentOptimizer with rate is 0.5
tf.global_variables_initializer().run()   # initial variables
for i in range(1000):   # training step 1000 times
  batch_xs, batch_ys = mnist.train.next_batch(100)  # random select 100 data for training
  train_step.run({x: batch_xs, y_: batch_ys})

# For calculate accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))