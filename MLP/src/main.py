import tensorflow as tf
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

def standScale(x_train, x_test):
    prep = preprocessing.StandardScaler().fit(x_train)
    x_train = prep.transform(x_train)
    x_test = prep.transform(x_test)

    return x_train, x_test

def readTrain():
    filename1 = "../dataSet/train_data/facade0_all_new.csv"
    filename2 = "../dataSet/train_data/facade1_all_new.csv"
    filename3 = "../dataSet/train_data/trainData.csv"
    train_df = pd.read_csv(filename3,header=None)
    info = train_df.values
    x_data = info[:, :-1]
    y_data = info[:, -1:]

    # df0 = pd.read_csv(filename1, header=None)
    # df1 = pd.read_csv(filename2, header=None)
    # info1 = df1.values
    # info0 = df0.values
    #
    # X_0 = info0[:, :-1]
    # Y_0 = info0[:, -1:]
    # X = info1[:, :-1]
    # Y = info1[:, -1:]
    #
    # X_all = np.concatenate((X_0, X), axis=0)
    # Y_all = np.concatenate((Y_0, Y), axis=0)

    enc = preprocessing.OneHotEncoder()
    y_label = enc.fit_transform(y_data).toarray()

    train_x, test_x, train_y, test_y = train_test_split(x_data, y_label, test_size=0.1)  # 选择10%的数据作为测试集
    return train_x,test_x,train_y,test_y

#隐含层节点+偏置+softMax
def multilayer_perceptron(_X, _weights, _biases):
    layer_1 = tf.nn.relu(
        tf.add(tf.matmul(_X, _weights['w1']), _biases['b1'])
    ) ## hidden layer

    out = tf.matmul(layer_1,_weights['out'])

    out = out + _biases['out']

    out = tf.nn.softmax(out)
    return out

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
        lr = 0.15
    elif epoch_num < 1000:
        lr = 0.12
    else:
        lr = 0.08
    return lr

def training(train_x,test_x,train_y,test_y):
    n_hidden_1 = 16
    n_input = 20
    n_classes = 2
    weight_decay = 0.0001
    dropout = 0.75

    train_x,x_test = standScale(train_x,test_x) #标准化训练数据和测试数据集
    #
    input_data = np.concatenate((train_x,train_y), axis=1)

    # place Holder
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, n_input], name='input_x')
        y = tf.placeholder(tf.float32, [None, n_classes], name='input_y')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32)
    # 标准差为0.1的初始化权重
    with tf.variable_scope('weights', reuse=True):
        weights = {
            'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.1),name='w1'),
            'out': tf.Variable(tf.zeros([n_hidden_1, n_classes]),name='out1')
        }

    # bias
    with tf.variable_scope('biases',reuse=True):
        biases = {
            'b1': tf.Variable(tf.zeros([n_hidden_1]),name='b1'),
            'out': tf.Variable(tf.zeros([n_classes]),name='out2')
        }

    with tf.name_scope('l2_loss'):
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

    with tf.name_scope('prediction'):
        out = multilayer_perceptron(x, weights, biases)  # 预测softMax结果矩阵

    with tf.name_scope('dropout'):
        out = tf.nn.dropout(out,keep_prob)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y),name='loss')  # 计算交叉熵

    with tf.name_scope('optimizer'):
        optm = tf.train.GradientDescentOptimizer(learning_rate,name='optimizer').minimize(loss + l2*weight_decay)  # 梯度下降优化器

    with tf.name_scope('output'):
        pr_label = tf.arg_max(out, 1,name='prediction')
    with tf.name_scope('correction'):
        corr = tf.equal(pr_label, tf.arg_max(y, 1),name='correction')  # 比较预测label和实际标签矩阵
    with tf.name_scope('accuracy'):
        accr = tf.reduce_mean(tf.cast(corr, tf.float32), name='accuracy')  # 转化正确率为float数值

    # 初始化TensorFlow

    epochs = 4000
    batch_size = 1000
    display_step = 10

    max_acc = 0

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)  # 初始化模型
    saver = tf.train.Saver(max_to_keep=1)
    avg_loss_list = []
    test_acc_list = []
    for epoch in range(epochs+1):
        lr = learning_rate_adjust(epoch)

        avg_loss = 0
        total_batch = int(len(input_data) / batch_size) + 1
        for i in range(total_batch):
            batch_xs, batch_ys = fetch_next_batch(input_data, batch_size, i)
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, learning_rate:lr, keep_prob:dropout})
            avg_loss += sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keep_prob:dropout}) / total_batch
            # if i % 10 == 0:
            #     test_acc = sess.run(accr, feed_dict={x: test_x, y: test_y})
            #     print(test_acc)
        # Display
        if epoch % display_step == 0:
            train_acc = sess.run(accr, feed_dict={x: train_x, y: train_y,keep_prob:dropout})
            # if train_acc > max_acc:
            #     max_acc = train_acc
            saver.save(sess, './model/model3/model.ckpt')
            test_acc = sess.run(accr, feed_dict={x: x_test, y: test_y,keep_prob:1.0})
            # print("Train acc: %.3f, Test acc: %.3f" % (train_acc,test_acc))
            # test_acc = sess.run(pred, feed_dict={x: test_x, y: test_y})
            # print(test_acc)
            print("Epoch: %03d/%03d cost: %.9f Train acc: %.3f Test acc: %.3f"
                  % (epoch, epochs, avg_loss, train_acc, test_acc))
            avg_loss_list.append(avg_loss)
            test_acc_list.append(test_acc)
    print("Now training done")
    sess.close()
    fig = plt.figure()
    ax = plt.subplot()
    l1, = ax.plot(avg_loss_list,c='b')
    l2, = ax.plot(test_acc_list,c= 'r')
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
        saver = tf.train.import_meta_graph("./model/model2/model.ckpt.meta")
        with tf.Session() as sess:
            modelFile = tf.train.latest_checkpoint("./model/model1/")

            saver.restore(sess, modelFile)
            graph = tf.get_default_graph()
            prediction = graph.get_tensor_by_name("output/prediction:0")

            pre_label = sess.run(prediction, feed_dict={"input/input_x:0": tmp})
            print(pre_label)
            return pre_label
            # if pre_label[0] == 1:
            #     # 计算线段交点 sn_group[.,.,.,.]
            #
            #     # 存入数据库
            #     print("This group can form a frame, insert it into database")


# sn_group1 = [257970.74,257970.74,273888.69,278038.69,90.0,250570.74,250570.74,273888.69,278038.69,90.0,
#              250220.74,258370.74,278038.69,278038.69,0.0, 250170.74,258370.74,273888.69,273888.69,0.0]
# sn_go = [258370.74,285670.71,279188.69,279188.69,0.0,266370.74,266370.74,274188.69,279188.69,90.0,
#          258570.74,283570.74,274188.69,274188.69,0.0,264570.73,264570.73,274188.69,279188.69,90.0]
#
# sn_go1 = [266370.74,266370.74,274188.69,279188.69,90.0,264570.73,264570.73,274188.69,279188.69,90.0,
#           258370.74,285670.71,279188.69,279188.69,0.0,258570.74,283570.74,274188.69,274188.69,0.0]
#
# test_group = [[264570.73, 264570.73, 274188.69, 279188.69, 90.0, 266370.74, 266370.74, 274688.69, 275888.69, 90.0, 264770.73, 266370.74, 274688.69, 274688.69, 0.0, 264770.73, 266370.74, 275888.69, 275888.69, 0.0],
#               [264570.73, 264570.73, 274188.69, 279188.69, 90.0, 264820.73, 264820.73, 274738.69, 275838.69, 90.0, 264770.73, 266370.74, 274688.69, 274688.69, 0.0, 264770.73, 266370.74, 275888.69, 275888.69, 0.0],
#               [264570.73, 264570.73, 274188.69, 279188.69, 90.0, 266320.74, 266320.74, 274738.69, 275838.69, 90.0, 264770.73, 266370.74, 274688.69, 274688.69, 0.0, 264770.73, 266370.74, 275888.69, 275888.69, 0.0],
#               [264570.73, 264570.73, 274188.69, 279188.69, 90.0, 266370.74, 266370.74, 274188.69, 279188.69, 90.0, 264770.73, 266370.74, 274688.69, 274688.69, 0.0, 264770.73, 266370.74, 275888.69, 275888.69, 0.0]]

# model_start(test_group)
# model_start([sn_go])

model_start([])