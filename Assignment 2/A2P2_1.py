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


train_size, num_feature = np.shape(trainData)
colsize = 1
train_model = "Logistic"

x = tf.placeholder("float", shape=[None, num_feature])
y_true = tf.placeholder("float", shape=[None, colsize])


W = tf.Variable(tf.truncated_normal(shape=[num_feature, colsize],stddev=0.5))
b = tf.Variable(tf.zeros([colsize]))
y = tf.matmul(x, W) + b


pic1 = plt.figure()
p1 = fig_loss.add_subplot(111)
pic1 = plt.figure()
p2 = fig2.add_subplot(111)

for learning_rate in [0.005]:
	weight_decay = 0.01

	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y)) + 0.5*weight_decay*tf.reduce_sum(tf.square(W))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	equallist = tf.abs(y_true-tf.sigmoid(y))
	correct_prediction = tf.less(equallist, 0.5)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	batch_size = 500
	number_batches = int(train_size/batch_size)
	iterations = 5000

	valid_loss_history = []
	loss_history = []
	train_history = []
	valid_history = []

	    tf.global_variables_initializer().run()    
	    for iteration in xrange(iterations):
		gap = (iteration*batch_size) % colsize
		
		batch_data = trainData[gap:(gap + batch_size)]
		batch_target = trainTarget[gap:(gap + batch_size)]
	    
		optimizer.run(feed_dict={x: batch_data, y_true: batch_target})

		if iteration%number_batches==0:
		    randIndx = np.arange(len(trainData))
		    np.random.shuffle(randIndx)
		    trainData, trainTarget = trainData[randIndx], trainTarget[randIndx]
		    loss_history.append(ss.run(loss, feed_dict={x: trainData, y_true: trainTarget}))
		    valid_loss_history.append(ss.run(loss, feed_dict={x: validData, y_true: validTarget}))
		    train_history.append(accuracy.eval({x: trainData, y_true: trainTarget}))
		    valid_history.append(accuracy.eval({x: validData, y_true: validTarget}))

	    ax1.plot(range(1,len(loss_history)+1), loss_history, '-', label='train loss')
	    ax2.plot(range(1,len(train_history)+1), train_history, '-', label='train accuracy')
	    ax1.plot(range(1,len(valid_loss_history)+1), valid_loss_history, '-', label='valid loss')
	    ax2.plot(range(1,len(valid_history)+1), valid_history, '-', label='valid accuracy')

	    print 'loss: %s' % ss.run(loss, feed_dict={x: trainData, y_true: trainTarget})
	    print 'valid loss: %s' % ss.run(loss, feed_dict={x: validData, y_true: validTarget})
	    print 'train accuracy: %s' % accuracy.eval({x: trainData, y_true: trainTarget})
	    print 'validation accuracy: %s' % accuracy.eval({x: validData, y_true: validTarget})
	    print 'Test Accuracy: %s' % accuracy.eval({x: testData, y_true: testTarget})

p1.set_title('2.1.1 Cross Entropy Loss Curves for training and valid data')
p2.set_title('2.1.1 Accuracy Curves for training and valid data')
p1.set_xlabel('Number of epoch')
p2.set_xlabel('Number of epoch')
p1.set_ylabel('Cross Entropy Loss')
p2.set_ylabel('Accuracy')
p1.grid(True)
p2.grid(True)
p1.legend()
p2.legend()
plt.show()
