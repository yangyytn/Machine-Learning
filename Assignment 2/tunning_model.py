import tensorflow as tf
import numpy as np
import time
import matplotlib .pyplot as plt

with np.load("notMNIST.npz") as data :
    Data, Target = data ["images"], data["labels"]
    posClass = 2
    negClass = 9
    dataIndx = (Target==posClass) + (Target==negClass)
    Data = np.reshape(Data, [-1, 28*28])
    Data = Data[dataIndx]/255.
    Target = Target[dataIndx].reshape(-1, 1)
    Target[Target==posClass] = 1
    Target[Target==negClass] = 0
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data, Target = Data[randIndx], Target[randIndx]
    trainData, trainTarget = Data[:3500], Target[:3500]
    validData, validTarget = Data[3500:3600], Target[3500:3600]
    testData, testTarget = Data[3600:], Target[3600:]


train_size, num_feature = np.shape(trainData)
num_label = 1


x = tf.placeholder("float", shape=[None, num_feature])
y_true = tf.placeholder("float", shape=[None, num_label])

W = tf.Variable(tf.truncated_normal(shape=[num_feature, num_label],stddev=0.5))
b = tf.Variable(tf.zeros([num_label]))
y = tf.matmul(x, W) + b

for learning_rate in [0.005]:
    for batch_size in [500]:
        for weight_decay in [0.0]:
            loss = 0.5*tf.reduce_mean(tf.squared_difference(y, y_true)) + 0.5*weight_decay*tf.reduce_sum(tf.square(W))
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


            delta = tf.abs(y_true - y)
            correct_prediction = tf.cast(tf.less(delta, 0.5), tf.int32)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            epoch_ratio = int(train_size/batch_size)
            iteration = 20000

            loss_history = []

            with tf.Session() as s:
		start = time.time()
                tf.global_variables_initializer().run()
                tf.initialize_all_variables().run()

                for step in xrange(iteration):
                    offset = (step*batch_size) % num_label
                    
                    batch_data = trainData[offset:(offset + batch_size)]
                    batch_target = trainTarget[offset:(offset + batch_size)]
                
                    train_step.run(feed_dict={x: batch_data, y_true: batch_target})

                    if step%epoch_ratio==0:
                        randIndx = np.arange(len(trainData))
                        np.random.shuffle(randIndx)
                        trainData, trainTarget = trainData[randIndx], trainTarget[randIndx]
                        loss_history.append(s.run(loss, feed_dict={x: trainData, y_true: trainTarget}))

                finish = time.time()
		procee = finish - start
                print 'mse: %s' % s.run(loss, feed_dict={x: trainData, y_true: trainTarget})
                print 'training accuracy: %s' % accuracy.eval({x: trainData, y_true: trainTarget})
                print 'validation accuracy: %s' % accuracy.eval({x: validData, y_true: validTarget})
                print 'test accuracy: %s' % accuracy.eval({x: testData, y_true: testTarget})
		print 'Operation Time: %s' %str(procee)

                plt.plot(range(1,len(loss_history)+1), loss_history, '-', label='Learning Rate: %s' % learning_rate)

plt.title('1.1 Tuning the learning rate for linear regression model')
plt.xlabel('Number of epochs')
plt.ylabel('MSE(Training Loss Function)')
plt.grid(True)
plt.legend()
plt.show()
