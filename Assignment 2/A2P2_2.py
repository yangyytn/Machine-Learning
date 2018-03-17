import tensorflow as tf
import numpy as np
import matplotlib .pyplot as plt
import pdb


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


train_size, rowsize = np.shape(trainData)
colsize = 1


x = tf.placeholder("float", shape=[None, rowsize])
y_true = tf.placeholder("float", shape=[None, colsize])

W = tf.Variable(tf.truncated_normal(shape=[rowsize, colsize],stddev=0.5))
b = tf.Variable(tf.zeros([colsize]))
y = tf.matmul(x, W) + b


pic = plt.figure()
p1 = pic.add_subplot(111)

for learning_rate in [0.001]:
    weight_decay = 0.01

    for train_model in ['SGD Optimizer', 'Adam Optimizer']:
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y)) + 0.5*weight_decay*tf.reduce_sum(tf.square(W))
        if train_model == 'SGD Optimizer':
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        else:
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

      	delta = tf.abs(y_true - tf.sigmoid(y))
        correct_prediction = tf.cast(tf.less(delta, 0.5), tf.int32)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        batch_size = 500
        epoch_ratio = int(train_size/batch_size)
        iteration = 5000


        with tf.Session() as ss:
	    loss_history = []
            tf.global_variables_initializer().run()
            
            for step in xrange(iteration):
                gap = (step*batch_size) % colsize
                
                batch_data = trainData[gap:(gap + batch_size)]
                batch_target = trainTarget[gap:(gap + batch_size)]
            
                train_step.run(feed_dict={x: batch_data, y_true: batch_target})

                if step%epoch_ratio==0:
                    randIndx = np.arange(len(trainData))
                    np.random.shuffle(randIndx)
                    trainData, trainTarget = trainData[randIndx], trainTarget[randIndx]
                    loss_history.append(s.run(loss, feed_dict={x: trainData, y_true: trainTarget}))


            p1.plot(range(1,len(loss_history)+1), loss_history, '-', label='model: %s' % train_model)

p1.set_title('2.1.2 Cross Entropy Loss Comperison between SGD and Adam Optimizer')
p1.set_xlabel('Number of epochs')
p1.set_ylabel('Cross Entropy Loss')
p1.legend()
p1.grid(True)
plt.show()
