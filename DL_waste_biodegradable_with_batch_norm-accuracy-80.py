IMG_S1,IMG_S2 = 100,100
X=[]
Y=[]
epsilon = 1e-3
def imgread_scale_resize(PATH,img):
    img_array = cv2.imread(os.path.join(PATH,img),cv2.IMREAD_COLOR)
    img_array = cv2.resize(img_array , (IMG_S1,IMG_S2))
    img_array = cv2.GaussianBlur(img_array,(3,3),0)
    return img_array

def createTrainingData():
    pathh = "/kaggle/input/waste-classification-data/dataset/DATASET"
    CATEGORIES = ["O","R"]
    types = ["TRAIN","TEST"]
    for typ in types:
        datadir = os.path.join(pathh,typ)
        for ele in CATEGORIES:
            a = 0
            PATH = os.path.join(datadir,ele)
            class_num = CATEGORIES.index(ele)
            for img in os.listdir(PATH):
                try:
                    img_array = imgread_scale_resize(PATH,img)
                    X.append(list(img_array))
                    Y.append(class_num)
                    print(a)
                except Exception as e:
                    pass
                a+=1
#             if(a>10000): break

createTrainingData()

X = np.array(X)
Y = np.array(Y)

X = X.reshape(X.shape[0],-1).T
Y = Y.T
X = X/255 #normalized

#Output config C: DEPth
def one_hottie(labels,C):
    sess = tf.Session()
    C = tf.constant(C) #four shapes
    One_hot_matrix = tf.one_hot(labels,C,axis=0)
    Y = sess.run(One_hot_matrix)
    sess.close()
    return Y
Y = one_hottie(Y,2)

#splitting
X_train, X_test, y_train, y_test = train_test_split(X.T, Y.T, test_size = 0.13, random_state = 0)
del X
del Y
X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T
#creating placeholders
def create_placeholders(n_x,n_y):
    """
    Creates two placeholder of shape (n_x,none) and (n_y,none)
    n_x : shape of 1 image vector - height * width * colors(RGB or grayscale etc)
    n_y : no. of classes
    """
    X = tf.placeholder(dtype = tf.float32, shape = (n_x,None),name = "X")
    Y = tf.placeholder(dtype = tf.float32, shape = (n_y,None),name = "Y")
    return X,Y

# SHAPE OF NET : LINEAR(Z1) => RELU(A1) => LINEAR(z2) => RELU(a2) => LINEAR(z3) => SOFTMAX(y)

#model functions
def initialize_parameters():
    """
        W1 : [n[1], shape of 1 image vector]
        b1 : [n[1], 1]
        W2 : [n[2], n[1]]
        b2 : [n[2], 1]
        W3 : [n[3], n[2]]
        b3 : [n[3], 1]
        W4 : [y[1], n[3]]
        b4 : [y[1], 1]
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3, W4, b4
    """
    W1 = tf.get_variable(name= "W1", shape= [40,100*100*3], initializer = tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable(name= "W2", shape= [30,40], initializer = tf.contrib.layers.xavier_initializer())
    W3 = tf.get_variable(name= "W3", shape= [20,30], initializer = tf.contrib.layers.xavier_initializer())
    W4 = tf.get_variable(name= "W4", shape= [2,20], initializer = tf.contrib.layers.xavier_initializer())

    beta1 = tf.Variable(tf.zeros([40,1]))
    beta2 = tf.Variable(tf.zeros([30,1]))
    beta3 = tf.Variable(tf.zeros([20,1]))
    beta4 = tf.Variable(tf.zeros([2,1]))

    scale1 = tf.Variable(tf.ones([40,1]))
    scale2 = tf.Variable(tf.ones([30,1]))
    scale3 = tf.Variable(tf.ones([20,1]))
    scale4 = tf.Variable(tf.ones([2,1]))

    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "W4": W4,
                  "beta1" : beta1,
                  "beta2" : beta2,
                  "beta3" : beta3,
                  "beta4" : beta4,
                  "scale1" : scale1,
                  "scale2" : scale2,
                  "scale3" : scale3,
                  "scale4" : scale4,}
    return parameters

#forward prop
def forward_propagation(X, parameters):
    """
    Arguments:
    X : placeholder of shape (input size, number of examples)
    parameters : dict of W and b

    Returns:
    Z4 -- the output of the last LINEAR unit
    """

    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']

    beta1 = parameters['beta1']
    beta2 = parameters['beta2']
    beta3 = parameters['beta3']
    beta4 = parameters['beta4']

    scale1 = parameters['scale1']
    scale2 = parameters['scale2']
    scale3 = parameters['scale3']
    scale4 = parameters['scale4']

    Z1 = tf.matmul(W1,X)
    batch_mean1, batch_var1 = tf.nn.moments(Z1,[0])
    Z1 = tf.nn.batch_normalization(Z1,batch_mean1,batch_var1,beta1,scale1,epsilon)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(W2,A1)

    batch_mean2, batch_var2 = tf.nn.moments(Z2,[0])
    Z2 = tf.nn.batch_normalization(Z2,batch_mean2,batch_var2,beta2,scale2,epsilon)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(W3,A2)

    batch_mean3, batch_var3 = tf.nn.moments(Z3,[0])
    Z3 = tf.nn.batch_normalization(Z3,batch_mean3,batch_var3,beta3,scale3,epsilon)
    A3 = tf.nn.relu(Z3)
    Z4 = tf.matmul(W4,A3)

    batch_mean4, batch_var4 = tf.nn.moments(Z4,[0])
    Z4 = tf.nn.batch_normalization(Z4,batch_mean4,batch_var4,beta4,scale4,epsilon)
    return Z4  #Linear Z4

def compute_cost(Z4, Y):
    """
    Computes the cost

    Arguments:
    Z4 : Linear unit output of forward prop ...of shape (6, number of examples)
    Y : True vales placeholder, same shape as Z4

    Returns:
    cost - Tensor of the cost function
    """

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z4)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits ,labels = labels))

    return cost

def random_mini_batches(X_train, y_train, minibatch_size, seed):
    m = X_train.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    num_complete_minibatches = floor(m/minibatch_size) # number of mini batches of size minibatch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X_train[:, k * minibatch_size : k * minibatch_size + minibatch_size]
        mini_batch_Y = y_train[:, k * minibatch_size : k * minibatch_size + minibatch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < minibatch_size)
    if m % minibatch_size != 0:
        mini_batch_X = X_train[:, num_complete_minibatches * minibatch_size : m]
        mini_batch_Y = y_train[:, num_complete_minibatches * minibatch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def model(X_train, y_train, X_test, y_test, learning_rate = 0.001,learning_rate_decay = False,
          num_epochs = 1000, minibatch_size = 100, print_cost = True):
    """
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost

    X, Y = create_placeholders(n_x,n_y)
    parameters = initialize_parameters()

    Z4 = forward_propagation(X,parameters)

    cost = compute_cost(Z4,Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, y_train, minibatch_size, seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X, Y:minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches

            if print_cost == True and epoch % 25 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        correct_prediction = tf.equal(tf.argmax(Z4), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: y_test}))

        return parameters

parameters = model(X_train, y_train, X_test, y_test,num_epochs=1500,minibatch_size = 100,learning_rate = 0.0001)
