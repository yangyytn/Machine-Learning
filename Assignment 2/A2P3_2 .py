def part2():

    # best learning rate = 0.001
    # best weight decay coefficient = 0.001

    # load data
    trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation('data.npy','target.npy', 0)

    batch_size = 300

    if batch_size == 300:
        trainData = np.concatenate((trainData[:],trainData[:153]))
        trainTarget = np.concatenate((trainTarget[:],trainTarget[:153]))

    epoch_size = len(trainData)
    valid_size = len(validData)
    test_size = len(testData)
    numBatches = epoch_size/batch_size
    print("Number of batches: %d" % numBatches)

    trainData = np.reshape(trainData,[epoch_size,-1])
    validData = np.reshape(validData,[valid_size,-1])
    testData = np.reshape(testData,[test_size,-1])

    trainTarget = reshape_target(trainTarget,6)
    validTarget = reshape_target(validTarget,6)
    testTarget = reshape_target(testTarget,6)

    trainTarget = np.reshape(trainTarget,[epoch_size,-1])
    validTarget = np.reshape(validTarget,[valid_size,-1])
    testTarget = np.reshape(testTarget,[test_size,-1])


    W = tf.Variable(tf.random_normal(shape=[trainData.shape[1], 6], stddev=0.35, seed=521), name="weights")
    b = tf.Variable(0.0, name="biases")
    X = tf.placeholder(tf.float32, name="input_x")
    y_target = tf.placeholder(tf.float32, name="target_y")


    y_pred = tf.matmul(X,W) + b

    l_rate = tf.placeholder(tf.float32, [], name="learning_rate")
    W_lambda = tf.placeholder(tf.float32, [], name="weight_decay")

    logits_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_target)/2.0, name="logits_loss")
    W_decay = tf.reduce_sum(tf.multiply(W,W))*W_lambda/2.0
    loss = logits_loss + W_decay
    y_pred_sigmoid = tf.sigmoid(y_pred)
    correct_predictions = tf.equal(tf.argmax(y_pred_sigmoid, 1), tf.argmax(y_target, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), 0)

    # training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
    train = optimizer.minimize(loss=loss)


    each_l_rate = 0.005
    each_W_lambda = 0.001

    # initialize session
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    sess.run(W)
    sess.run(b)

    train_error_list = []
    train_accuracy_list = []
    valid_error_list = []
    valid_accuracy_list = []

    for step in range(0,5000):
        batch_idx = step%numBatches
        trainDataBatch = trainData[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)]
        trainTargetBatch = trainTarget[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)]
        sess.run([train, W, b, y_pred], feed_dict={X: trainDataBatch, y_target: trainTargetBatch, l_rate: each_l_rate, W_lambda: each_W_lambda})
        if batch_idx == numBatches-1:
            train_error = sess.run(loss, feed_dict={X: trainDataBatch, y_target: trainTargetBatch, W_lambda: each_W_lambda})
            train_accuracy = sess.run(accuracy, feed_dict={X: trainDataBatch, y_target: trainTargetBatch})
            valid_error = sess.run(loss, feed_dict={X: validData, y_target: validTarget, W_lambda: each_W_lambda})
            valid_accuracy = sess.run(accuracy, feed_dict={X: validData, y_target: validTarget})
            train_error_list.append(train_error)
            train_accuracy_list.append(train_accuracy)
            valid_error_list.append(valid_error)
            valid_accuracy_list.append(valid_accuracy)
        if step%1000==0 and step > 0:
            print("Step: %d " % step)
            print("Training error: %f " % train_error_list[-1])
            print("Validation error: %f " % valid_error_list[-1])

    # plot image
    fig_loss = plt.figure()
    ax1 = fig_loss.add_subplot(111)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax1.set_ylabel('Cross Entropy Loss')
    ax2.set_ylabel('Train Accuracy')
    ax1.set_xlabel('Number of epochs')
    ax2.set_xlabel('Number of epochs')
    ax1.plot(train_error_list, label="Data: Training")
    ax1.plot(valid_error_list, label="Data: Validation")
    ax1.set_title("2.2.2 Cross-Entropy Loss:learning rate = 0.001, weight decay = 0.001") 
    ax1.grid(True)
    ax1.legend()
    ax2.plot(train_accuracy_list, label="Data: Training")
    ax2.plot(valid_accuracy_list, label="Data: Validation")
    ax2.set_title("2.2.2 Classification Accuracy:learning rate = 0.001, weight decay = 0.001") 
    ax2.grid(True)
    ax2.legend()
    plt.show()

    test_error = sess.run(loss, feed_dict={X: testData, y_target: testTarget, W_lambda: each_W_lambda})
    test_accuracy = sess.run(accuracy, feed_dict={X: testData, y_target: testTarget})
    print("Test error: %f " % test_error)
    print("Test accuracy: %f " % test_accuracy)
