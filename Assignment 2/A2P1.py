import tensorflow as tf
import numpy as np
import time
import matplotlib .pyplot as plt

# Parameters
learning_rate = 0.0
iterations = 20000
display_step = 500
weight_decay_param = tf.constant(0.0)

def notMNIST_two():
	with np.load("notMNIST.npz") as data :
		Data, Target = data ["images"], data["labels"]
		posClass = 2
		negClass = 9
		dataIndx = (Target==posClass) + (Target==negClass)
		Data = Data[dataIndx]/255.0
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
		return trainData, trainTarget, validData, validTarget, testData, testTarget

def notMNIST():
    with np.load("notMNIST.npz") as data:
        Data, Target = data ["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
        return trainData, trainTarget, validData, validTarget, testData, testTarget

def faceScrub():
    data_path = 'data.npy'
    target_path = 'target.npy'
    task = 0
    data = np.load(data_path)/255.   
    data = np.reshape(data, [-1, 32,32])
    target = np.load(target_path)
    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)
    trBatch = int(0.8*len(rnd_idx))
    validBatch = int(0.1*len(rnd_idx))
    trainData, validData, testData = data[rnd_idx[1:trBatch],:], data[rnd_idx[trBatch+1:trBatch + validBatch],:], data[rnd_idx[trBatch + validBatch+1:-1],:]
    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], target[rnd_idx[trBatch+1:trBatch + validBatch], task], target[rnd_idx[trBatch + validBatch + 1:-1], task]
    return trainData, trainTarget, validData, validTarget, testData, testTarget

def Q1(q11=0, q12=0,q13=0,q14=0):
	trainData, trainTarget, validData, validTarget, testData, testTarget = notMNIST_two()
	if q11:
		part1(trainData,trainTarget)
	elif q12:
		part2(trainData,trainTarget)
	elif q13:
		part3(trainData,trainTarget,validData,validTarget)
	else:
		part4()


def part1(trainData,trainTarget):
	shape = trainData.shape[1]*trainData.shape[2]
	trainData = trainData.reshape(trainData.shape[0], trainData.shape[1]*trainData.shape[2])
	samplenumber = trainData.shape[0]
	X = tf.placeholder(tf.float32, shape=(None, shape))
	Y = tf.placeholder(tf.float32, shape=(None, 1))
	W = tf.Variable(tf.zeros((shape, 1)), name="weight")
	b = tf.Variable(tf.zeros(1), name="bias")

	prediction = tf.add(tf.matmul(X,W),b)
	
	Ld = tf.losses.mean_squared_error(Y,prediction)
	Lw = weight_decay_param * tf.norm(W) / 2.0
	totalL = tf.reduce_mean(Ld)/2. + Lw
	

	LearningRates = [0.005,0.001,0.0001]


	
	for i in LearningRates:
		losses = list()
		init = tf.global_variables_initializer()
		with tf.Session() as ss:
			ss.run(init)
			SGDoptimizer = tf.train.GradientDescentOptimizer(i).minimize(loss=totalL)
			print ("learning rate: " + str(i))

			num_batches =  int(trainData.shape[0] / 500)
			print ("num_batches: " + str(num_batches))
			for iteration in xrange(iterations):
				offset = (iteration*500) % 1
				result = None
			
				train_Batch = trainData[offset: (offset+500)]
				train_Target = trainTarget[offset: (offset+500)]
				ss.run(SGDoptimizer, feed_dict={X: train_Batch, Y: train_Target})
				if iteration%num_batches==0:
				        randIndx = np.arange(len(trainData))
				        np.random.shuffle(randIndx)
				        trainData, trainTarget = trainData[randIndx], trainTarget[randIndx]
				        losses.append(ss.run(totalL, feed_dict={X: trainData, Y: trainTarget}))

				
			plt.plot(range(1,len(losses)+1),losses,'-', label='Learning Rate: %s' % i)
	plt.xlabel('epoch')
	plt.ylabel('mse')
	plt.legend()
	plt.show()


def part2(trainData,trainTarget):
	shape = trainData.shape[1]*trainData.shape[2]
	trainData = trainData.reshape(trainData.shape[0], trainData.shape[1]*trainData.shape[2])
	samplenumber = trainData.shape[0]
	X = tf.placeholder(tf.float32, shape=(None, shape))
	Y = tf.placeholder(tf.float32, shape=(None, 1))
	W = tf.Variable(tf.ones((shape, 1)), name="weight")
	b = tf.Variable(tf.ones(1), name="bias")

	prediction = tf.add(tf.matmul(X,W),b)
	
	Ld = tf.losses.mean_squared_error(Y,prediction)
	Lw = weight_decay_param * tf.nn.l2_loss(W)
	totalL = tf.reduce_mean(Ld)/2 + tf.reduce_sum(Lw)

	LearningRate = 0.005
	Batches = [500,1500,3500]
	losses = list()
	for batch in Batches:
		SGDoptimizer = tf.train.GradientDescentOptimizer(LearningRate).minimize(loss=totalL)
		init = tf.global_variables_initializer()
		print ("batch size: " + str(batch))

		with tf.Session() as ss:
			start = time.time()
			ss.run(init)
			num_batches =  int(trainData.shape[0]/float(batch))
			print ("num_batches: " + str(num_batches))
			for iteration in range(int(iterations/num_batches)):
				result = None
				for i in range(num_batches):
					train_Batch = trainData[i*batch: (i+1) * batch]
					train_Target = trainTarget[i*batch: (i+1) * batch]
					ss.run(SGDoptimizer, feed_dict={X: train_Batch, Y: train_Target})

			train_loss = ss.run(totalL, feed_dict={X: trainData, Y: trainTarget})
			print("Test loss: " + str(train_loss))
			losses.append(train_loss)
			finish = time.time()
			process_time = finish - start
			print ("Batch size: " + str(batch) + " Time: " + str(process_time))
	print (losses)

	
def part3(trainData,trainTarget,validData,validTarget):
	trainloss,validloss = part3_helper(trainData,trainTarget,validData,validTarget)
	#validloss = part3_helper(validData,validTarget)
	print (trainloss,validloss)

def part3_helper(trainData,trainTarget,validData,validTarget):
	shape = trainData.shape[1]*trainData.shape[2]
	trainData = trainData.reshape(trainData.shape[0], trainData.shape[1]*trainData.shape[2])
	samplenumber = trainData.shape[0]
	validData = validData.reshape(validData.shape[0], validData.shape[1]*validData.shape[2])
	

	X = tf.placeholder(tf.float32, shape=(None, shape))
	Y = tf.placeholder(tf.float32, shape=(None, 1))
	W = tf.Variable(tf.ones((shape, 1)), name="weight")
	b = tf.Variable(tf.ones(1), name="bias")
	
	prediction = tf.add(tf.matmul(X,W),b)

	LearningRates = 0.005
	
	weight_decays = [0.0,0.001,0.1,1.0]

	losses = list()
	valid_losses = list()
	for weight in weight_decays:
		print ("current weight decay: " + str(weight))
		Ld = tf.losses.mean_squared_error(Y,prediction)
		Lw = weight * tf.nn.l2_loss(W)
		totalL = tf.reduce_mean(Ld)/2 + tf.reduce_sum(Lw)

		SGDoptimizer = tf.train.GradientDescentOptimizer(LearningRates).minimize(loss=totalL)
		init = tf.global_variables_initializer()
		#print ("learning rate: " + str(i))

		with tf.Session() as ss:
			ss.run(init)
			num_batches =  int(trainData.shape[0] / 500)
			print ("num_batches: " + str(num_batches))
			for iteration in range(iterations):
				result = None
				for j in range(num_batches):
					train_Batch = trainData[j*500: (j+1) * 500]
					train_Target = trainTarget[j*500: (j+1) * 500]					
					ss.run(SGDoptimizer, feed_dict={X: train_Batch, Y: train_Target})
			
			test_loss = ss.run(totalL, feed_dict={X: trainData, Y: trainTarget})
			valid_loss = ss.run(totalL, feed_dict = {X:validData, Y:validTarget})				
			losses.append(test_loss)
			valid_losses.append(valid_loss)
	return (losses,valid_losses)

def part4():
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

	W = tf.Variable(tf.zeros([num_feature, num_label]))
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
		        tf.global_variables_initializer().run()
		        tf.initialize_all_variables().run()
			start = time.time()
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

		        
		        print 'mse: %s' % s.run(loss, feed_dict={x: trainData, y_true: trainTarget})
		        print 'training accuracy: %s' % accuracy.eval({x: trainData, y_true: trainTarget})
		        print 'validation accuracy: %s' % accuracy.eval({x: validData, y_true: validTarget})
		        print 'test accuracy: %s' % accuracy.eval({x: testData, y_true: testTarget})
			finish = time.time()
			process_time = finish - start
		        plt.plot(range(1,len(loss_history)+1), loss_history, '-', label='weight decay: %s' % weight_decay)

	plt.xlabel('epoch')
	plt.ylabel('mse')
	plt.legend()
	plt.show()
		
if __name__ == '__main__':
	Q1(0,0,1,0)

