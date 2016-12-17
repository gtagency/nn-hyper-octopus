import tensorflow as tf

import numpy as np

g = tf.Graph()
with g.as_default():
    X = tf.placeholder(tf.float32, [178 , 13], name ='X')
    Y = tf.placeholder(tf.float32, [178, 3], name = 'Y')
    # W = tf.Variable(tf.random_uniform([13, 3], -.01, .01), name='W')
    W = tf.Variable(tf.zeros([13, 3]), name='W')
    b =  tf.Variable(tf.zeros([3]), name='b')
    pred = tf.sigmoid(tf.matmul(X, W) + b)
    cost = tf.reduce_mean(tf.square(Y - pred))
    train_step = tf.train.GradientDescentOptimizer(.0004).minimize(cost)
    init = tf.initialize_all_variables()

sess=tf.InteractiveSession(graph=g)
sess.run(init)

input_data = np.loadtxt("data/wine.data",float,"#",",")
extractedData = input_data[:,0]
input_data = np.delete(input_data, 0, 1)
extractedData = extractedData.tolist();
extractedData[:] = [x - 1 for x in extractedData]
extractedData = np.eye(3)[extractedData]
data = {X: input_data.reshape(178, 13), Y: extractedData.reshape(178, 3)}
epochs = 0
while((1 - sess.run(cost, feed_dict=data)) < .97):
    for i in range(10000):
        sess.run(train_step, feed_dict=data)
    epochs += 1
    print(1 - sess.run(cost, feed_dict=data))
print(sess.run(cost, feed_dict=data))
print(str(epochs) + " epochs")



# print(sess.run(cost, feed_dict=data))
# # print(sess.run(W))
# print(sess.run(b))


# print(input_data)

# import tensorflow as tf
# g = tf.Graph()
# with g.as_default():
#     X = tf.placeholder(tf.float32, [4, 2], name='X')
#     Y = tf.placeholder(tf.float32, [4, 1], name='Y')
#     W1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name='w1')
#     b1 = tf.Variable(tf.zeros([2]), name='b1')
#     W2 = tf.Variable(tf.random_uniform([2, 1], -1, 1), name='w2')
#     b2 = tf.Variable(tf.zeros([1]), name='b2')
#     layer_one = tf.sigmoid(tf.matmul(X, W1) + b1)
#     pred = tf.sigmoid(tf.matmul(layer_one, W2) + b2)
#     cost = tf.reduce_mean(((Y * tf.log(pred)) + ((1 - Y) * tf.log(1.0 - pred))) * -1)
#     train_step = tf.train.GradientDescentOptimizer(.01).minimize(cost)
#     init = tf.initialize_all_variables()
