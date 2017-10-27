import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 28*28])
W = tf.Variable(tf.zeros([28*28, 10]))
b = tf.Variable(tf.zeros([10]))
h = tf.nn.softmax(tf.matmul(x, W) + b)
y = tf.placeholder(tf.float32, [None, 10])

with tf.name_scope("cost"):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # Add scalar summary for cost
    tf.summary.scalar("cost", cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(h, 1)) # Count correct predictions
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # Cast boolean to float to average
    # Add scalar summary for accuracy
    tf.summary.scalar("accuracy", accuracy)

with tf.Session() as sess:
    # create a log writer. run 'tensorboard --logdir=./logs/nn_logs'
    writer = tf.summary.FileWriter("./logs/nn_logs", sess.graph) # for 0.8
    merged = tf.summary.merge_all()

    # you need to initialize all variables
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        summary, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y: mnist.test.labels})
        writer.add_summary(summary, i)  # Write summary
        if i % 20 == 0:
            print("accuracy : ", acc * 100, "%")

