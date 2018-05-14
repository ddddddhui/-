import tensorflow as tf
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

n_input = 20
n_classes = 2
weight_decay = 0.0001
dropout = 0.75

def standScale(x_train, x_test):
    prep = preprocessing.StandardScaler().fit(x_train)
    x_train = prep.transform(x_train)
    x_test = prep.transform(x_test)

    return x_train, x_test

def readTrain():
    filename3 = "../dataSet/train_data/trainData.csv"
    train_df = pd.read_csv(filename3,header=None)
    info = train_df.values
    x_data = info[:, :-1]
    y_data = info[:, -1:]

    enc = preprocessing.OneHotEncoder()
    y_label = enc.fit_transform(y_data).toarray()

    train_x, test_x, train_y, test_y = train_test_split(x_data, y_label, test_size=0.1)  # 选择10%的数据作为测试集
    return train_x,test_x,train_y,test_y

#fetch batch info data
def fetch_next_batch(input_data,batch_size,index):
    if (index+1) * batch_size <= len(input_data):
        b_x = input_data[index*batch_size:(index+1)*batch_size, :-2]
        b_y = input_data[index*batch_size:(index+1)*batch_size, -2:]
    else:
        b_x = input_data[index*batch_size:, :-2]
        b_y = input_data[index*batch_size:, -2:]
    return b_x, b_y
#Train Data

def learning_rate_adjust(epoch_num):
    if epoch_num < 100:
        lr = 0.18
    elif epoch_num < 2000:
        lr = 0.16
    elif epoch_num < 5000:
        lr = 0.15
    else:
        lr = 0.14
    return lr

##权重初始化
def init_weight(shape, st_dev):
    weight = tf.Variable(tf.random_normal(shape,stddev=st_dev),name='weight')
    return weight


def zero_weight(shape):
    weight = tf.Variable(tf.zeros(shape),name='weight')
    return weight

##偏置初始化
def init_bias(shape):
    bias = tf.Variable(tf.zeros(shape), name='bias')
    return bias

##隐含层前向传播
def layer_compute(input_layer,weights,bias):
    full_layer = tf.add(tf.matmul(input_layer,weights), bias)
    return full_layer


##激活函数
def activation(layer_in):
    return tf.nn.relu(layer_in)


##训练
def training(train_x,test_x,train_y,test_y):
    train_x,x_test = standScale(train_x,test_x) #标准化训练数据和测试数据集
    #
    input_data = np.concatenate((train_x,train_y), axis=1)

    # place Holder x, y, learning_rate
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, n_input], name='input_x')
        y = tf.placeholder(tf.float32, [None, n_classes], name='input_y')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    ##keep probability
    # with tf.name_scope('keep_prob'):
    #     keep_prob = tf.placeholder(tf.float32)

    ###Fully connect layer with 16 nodes
    with tf.name_scope('full_connect'):
        w1 = init_weight(shape=[n_input,25], st_dev=0.1)
        b1 = init_bias(shape=[25])
        output = layer_compute(x,w1,b1)
        output = activation(output)

    ##mlp 1 with 8 nodes
    with tf.name_scope('mlp_1'):
        w2 = zero_weight(shape=[25,2])
        b2 = init_bias(shape=[2])
        output = layer_compute(output,w2,b2)
        # output = activation(output)
    #
    # ###mlp 2 with 2 nodes
    # with tf.name_scope('mlp_2'):
    #     w3 = init_weight(shape=[8,20], st_dev=0.1)
    #     b3 = init_bias(shape=[20])
    #     output = layer_compute(output,w3,b3)
    #     output = activation(output)

    # with tf.name_scope("fc"):
    #     w4 = init_weight(shape=[8,2], st_dev=0.1)
    #     b4 = init_bias([2])
    #     output = layer_compute(output, w4, b4)
    #     output = activation(output)

    ###soft max
    with tf.name_scope('softmax'):
        output = tf.nn.softmax(output)

    ##drop out
    # with tf.name_scope('dropout'):
    #     out = tf.nn.dropout(output,keep_prob)

    ##l2 loss function
    with tf.name_scope('l2_loss'):
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

    ##define loss funct
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y),name='loss')  # 计算交叉熵

    ###optimizer
    with tf.name_scope('optimizer'):
        optm = tf.train.GradientDescentOptimizer(learning_rate,name='optimizer').minimize(loss)  # 梯度下降优化器
        # optm = tf.train.AdamOptimizer(learning_rate, name='adam_opt').minimize(loss + l2* weight_decay)

    with tf.name_scope('prediction'):
        prediction = tf.arg_max(output,1,name='prediction')

    with tf.name_scope('correction'):
        corr = tf.equal(prediction, tf.arg_max(y, 1),name='correction')  # 比较预测label和实际标签矩阵

    with tf.name_scope('accuracy'):
        accr = tf.reduce_mean(tf.cast(corr, tf.float32), name='accuracy')  # 转化正确率为float数值

    # 初始化TensorFlow

    epochs = 6000
    batch_size = 250
    display_step = 10

    max_acc = 0

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)  # 初始化模型
    saver = tf.train.Saver(max_to_keep=1)
    avg_loss_list = []
    test_acc_list = []
    count = 0
    for epoch in range(epochs+1):
        lr = learning_rate_adjust(epoch)

        avg_loss = 0
        total_batch = int(len(input_data) / batch_size) + 1
        for i in range(total_batch):
            batch_xs, batch_ys = fetch_next_batch(input_data, batch_size, i)
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, learning_rate:lr})
            avg_loss += sess.run(loss, feed_dict={x: batch_xs, y: batch_ys}) / total_batch

        # Display
        if epoch % display_step == 0:
            train_acc = sess.run(accr, feed_dict={x: train_x, y: train_y})

            saver.save(sess, './new_model/model.ckpt')
            test_acc = sess.run(accr, feed_dict={x: x_test, y: test_y})

            if test_acc > max_acc:
                max_acc = test_acc
                count = 0
            else: count += 1

            print("Epoch: %03d/%03d cost: %.9f Train acc: %.3f Test acc: %.3f"
                  % (epoch, epochs, avg_loss, train_acc, test_acc))
            avg_loss_list.append(avg_loss)
            test_acc_list.append(test_acc)

            if count >= 30:
                print("Current max acc is : %.3f", max_acc)
                break

    print("Now training done")
    sess.close()
    fig = plt.figure()
    ax = plt.subplot()
    l1, = ax.plot(avg_loss_list,c='b')
    l2, = ax.plot(test_acc_list,c='r')
    ax.set_xlabel("EPOCH NUM")
    ax.legend(handles=[l1,l2,], labels=['avg_loss', 'test_acc'], loc='best')
    plt.show()
    # fig.savefig("./loss_acc_1.png")


def model_start(sn_group):
    train_x, test_x, train_y, test_y = readTrain()
    # print(len(train_x))
    # print(len(test_x))

    isTrain = True
    if isTrain:
        training(train_x, test_x, train_y, test_y)
    else:
        x_test = np.array(sn_group)
        _, tmp = standScale(train_x, x_test)  # 模拟测试
        saver = tf.train.import_meta_graph("./new_model/model.ckpt.meta")
        with tf.Session() as sess:
            modelFile = tf.train.latest_checkpoint("./new_model/")

            saver.restore(sess, modelFile)
            graph = tf.get_default_graph()
            prediction = graph.get_tensor_by_name("prediction/prediction:0")

            pre_label = sess.run(prediction, feed_dict={"input/input_x:0": tmp})
            print(pre_label)
            return pre_label


model_start([])