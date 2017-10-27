import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

keep_prob = tf.placeholder(tf.float32)

def weight_variable(shape):
    # 모두 다 0으로 초기화를 하면 학습이 잘 되지 않고 발산해버림
    initial = tf.truncated_normal(shape, stddev=0.1) # 절단 정규분포를 따르는 난수값 생성
    return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
    # 스트라이드는 1로 패딩은 0으로
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # 2*2 멕스 풀링
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 이미지를 3차원 데이터로 reshape 28x28x1
x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = weight_variable([5, 5, 1, 32])    # 5x5x1 conv, 32 output
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])   # 5x5x32 conv, 64 output
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024]) #  64 * 7 * 7 inputs, 1024 output
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

with tf.name_scope("cost"):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # Add scalar summary for cost
    tf.summary.scalar("cost", cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1)) # Count correct predictions
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # Cast boolean to float to average
    # Add scalar summary for accuracy
    tf.summary.scalar("accuracy", accuracy)

sess.run(tf.global_variables_initializer())
# create a log writer. run 'tensorboard --logdir=./logs/cnn_logs'
writer = tf.summary.FileWriter("./logs/cnn_logs", sess.graph)  # for 0.8
merged = tf.summary.merge_all()

avg_acc = 0
for i in range(20000):
    batch = mnist.train.next_batch(50)

    train_accuracy, _ = sess.run([accuracy, train_step], feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
    avg_acc += train_accuracy

    if i % 100 == 99:
        print("step ", i, ", training accuracy", avg_acc, "%")
        avg_acc = 0
        summary, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        writer.add_summary(summary, i)  # Write summary
