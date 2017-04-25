import numpy
import tensorflow as tf

from tweets_reader import TweetsReader
from tweets_batcher import TweetsBatcher

# Read Anna's data first, see README
reader = TweetsReader('Train5_dataset.csv', 'Test5_dataset.csv')

train_data, train_target = reader.get_train_data()
test_data,  test_target  = reader.get_test_data()

nr_of_cats   = numpy.shape(train_target)[1]
nr_of_words  = numpy.shape(train_data)[1]

batcher = TweetsBatcher(train_data, train_target)

# x is training data of (None) instances of word vectors 
x = tf.placeholder(tf.float32, [None, nr_of_words])

# weights and biases (nr_of_cats is a hotvector for categories)
W = tf.Variable(tf.zeros([nr_of_words, nr_of_cats]))
b = tf.Variable(tf.zeros([nr_of_cats]))

# y is my prediction
# y = Wx + b 
y = tf.nn.softmax(tf.matmul(x, W) + b)

# y' is the target 
y_ = tf.placeholder(tf.float32, [None, nr_of_cats])

# the loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# gradient descent with step 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# fill W and b with zeros
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# graph for evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 1000 iterations
for i in range(1000):
  # take random data from training data
  batch_xs, batch_ys = batcher.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  if (i % 10 == 0):
    print "Batch ", i,
    print " TRAIN ACC:",
    print(sess.run(accuracy, feed_dict={x: train_data.todense(), y_: train_target.todense()})),
    print " TEST ACC:",
    print(sess.run(accuracy, feed_dict={x: test_data.todense(), y_: test_target.todense()}))    

# evaluate using test data

print "FINAL TEST: ",
print(sess.run(accuracy, feed_dict={x: test_data.todense(), y_: test_target.todense()}))

