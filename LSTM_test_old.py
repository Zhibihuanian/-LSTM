import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import math

def LSTMtest(data):

    n1 = len(data[0]) - 1 #因为最后一位为label
    n2 = len(data)
    print(n1, n2)

    # 设置常量
    input_size = n1  # 输入神经元个数
    rnn_unit = 10    # LSTM单元(一层神经网络)中的中神经元的个数
    lstm_layers = 7  # LSTM单元个数
    output_size = 1  # 输出神经元个数（预测值）
    lr = 0.0006      # 学习率

    train_end_index = math.floor(n2*0.9)  # 向下取整
    print('train_end_index', train_end_index)
    # 前90%数据作为训练集，后10%作为测试集
    # 获取训练集
    # time_step 时间步，batch_size 每一批次训练多少个样例
    def get_train_data(batch_size=60, time_step=20, train_begin=0, train_end=train_end_index):
        batch_index = []
        data_train = data[train_begin:train_end]
        normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化
        train_x, train_y = [], []  # 训练集
        for i in range(len(normalized_train_data) - time_step):
            if i % batch_size == 0:
                # 开始位置
                batch_index.append(i)
                # 一次取time_step行数据
            # x存储输入维度（不包括label） :X(最后一个不取）
            # 标准化(归一化）
            x = normalized_train_data[i:i + time_step, :n1]
            # y存储label
            y = normalized_train_data[i:i + time_step, n1, np.newaxis]
            # np.newaxis分别是在行或列上增加维度
            train_x.append(x.tolist())
            train_y.append(y.tolist())
        # 结束位置
        batch_index.append((len(normalized_train_data) - time_step))
        print('batch_index', batch_index)
        # print('train_x', train_x)
        # print('train_y', train_y)
        return batch_index, train_x, train_y

    # 获取测试集
    def get_test_data(time_step=20, test_begin=train_end_index+1):
        data_test = data[test_begin:]
        mean = np.mean(data_test, axis=0)
        std = np.std(data_test, axis=0)  # 矩阵标准差
        # 标准化(归一化）
        normalized_test_data = (data_test - np.mean(data_test, axis=0)) / np.std(data_test, axis=0)
        # " // "表示整数除法。有size个sample
        test_size = (len(normalized_test_data) + time_step - 1) // time_step
        print('test_size$$$$$$$$$$$$$$', test_size)
        test_x, test_y = [], []
        for i in range(test_size - 1):
            x = normalized_test_data[i * time_step:(i + 1) * time_step, :n1]
            y = normalized_test_data[i * time_step:(i + 1) * time_step, n1]
            test_x.append(x.tolist())
            test_y.extend(y)
        test_x.append((normalized_test_data[(i + 1) * time_step:, :n1]).tolist())
        test_y.extend((normalized_test_data[(i + 1) * time_step:, n1]).tolist())
        return mean, std, test_x, test_y

    # ——————————————————定义神经网络变量——————————————————
    # 输入层、输出层权重、偏置、dropout参数
    # 随机产生 w,b
    weights = {
        'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
        'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
    }
    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
    }
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # dropout 防止过拟合

    # ——————————————————定义神经网络——————————————————
    def lstmCell():
        # basicLstm单元
        # tf.nn.rnn_cell.BasicLSTMCell(self, num_units, forget_bias=1.0,
        # tate_is_tuple=True, activation=None, reuse=None, name=None)
        # num_units:int类型，LSTM单元(一层神经网络)中的中神经元的个数，和前馈神经网络中隐含层神经元个数意思相同
        # forget_bias:float类型，偏置增加了忘记门。从CudnnLSTM训练的检查点(checkpoin)恢复时，必须手动设置为0.0。
        # state_is_tuple:如果为True，则接受和返回的状态是c_state和m_state的2-tuple；如果为False，则他们沿着列轴连接。后一种即将被弃用。
        # （LSTM会保留两个state，也就是主线的state(c_state),和分线的state(m_state)，会包含在元组（tuple）里边
        # state_is_tuple=True就是判定生成的是否为一个元组）
        #   初始化的 c 和 a 都是zero_state 也就是都为list[]的zero，这是参数state_is_tuple的情况下
        #   初始state,全部为0，慢慢的累加记忆
        # activation:内部状态的激活函数。默认为tanh
        # reuse:布尔类型，描述是否在现有范围中重用变量。如果不为True，并且现有范围已经具有给定变量，则会引发错误。
        # name:String类型，层的名称。具有相同名称的层将共享权重，但为了避免错误，在这种情况下需要reuse=True.
        #

        basicLstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit, forget_bias=1.0, state_is_tuple=True)
        # dropout 未使用
        drop = tf.nn.rnn_cell.DropoutWrapper(basicLstm, output_keep_prob=keep_prob)
        return basicLstm



    def lstm(X):  # 参数：输入网络批次数目
        batch_size = tf.shape(X)[0]
        time_step = tf.shape(X)[1]
        w_in = weights['in']
        b_in = biases['in']

        # 忘记门（输入门）
        # 因为要进行矩阵乘法,所以reshape
        # 需要将tensor转成2维进行计算
        input = tf.reshape(X, [-1, input_size])
        input_rnn = tf.matmul(input, w_in) + b_in
        # 将tensor转成3维，计算后的结果作为忘记门的输入
        input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])
        print('input_rnn', input_rnn)
        # 更新门
        # 构建多层的lstm
        cell = tf.nn.rnn_cell.MultiRNNCell([lstmCell() for i in range(lstm_layers)])
        init_state = cell.zero_state(batch_size, dtype=tf.float32)

        # 输出门
        w_out = weights['out']
        b_out = biases['out']
        # output_rnn是最后一层每个step的输出,final_states是每一层的最后那个step的输出
        output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
        output = tf.reshape(output_rnn, [-1, rnn_unit])
        # 输出值，同时作为下一层输入门的输入
        pred = tf.matmul(output, w_out) + b_out
        return pred, final_states

    # ————————————————训练模型————————————————————

    def train_lstm(batch_size=60, time_step=20, train_begin=0, train_end=train_end_index):
        # 于是就有了tf.placeholder，
        # 我们每次可以将 一个minibatch传入到x = tf.placeholder(tf.float32,[None,32])上，
        # 下一次传入的x都替换掉上一次传入的x，
        # 这样就对于所有传入的minibatch x就只会产生一个op，
        # 不会产生其他多余的op，进而减少了graph的开销。

        X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
        Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
        batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
        # 用tf.variable_scope来定义重复利用,LSTM会经常用到
        with tf.variable_scope("sec_lstm"):
            pred, state_ = lstm(X) # pred输出值，state_是每一层的最后那个step的输出
        print('pred,state_', pred, state_)

        # 损失函数
        # [-1]——列表从后往前数第一列，即pred为预测值，Y为真实值(Label)
        #tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值
        loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
        # 误差loss反向传播——均方误差损失
        # 本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。
        # Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳.
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

        with tf.Session() as sess:
            # 初始化
            sess.run(tf.global_variables_initializer())
            theloss = []
            # 迭代次数
            for i in range(200):
                for step in range(len(batch_index) - 1):
                    # sess.run(b, feed_dict = replace_dict)
                    state_, loss_ = sess.run([train_op, loss],
                                             feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                        Y: train_y[batch_index[step]:batch_index[step + 1]],
                                                        keep_prob: 0.5})
                    #  使用feed_dict完成矩阵乘法 处理多输入
                    #  feed_dict的作用是给使用placeholder创建出来的tensor赋值


                    #  [batch_index[step]: batch_index[step + 1]]这个区间的X与Y
                    #  keep_prob的意思是：留下的神经元的概率，如果keep_prob为0的话， 就是让所有的神经元都失活。
                print("Number of iterations:", i, " loss:", loss_)
                theloss.append(loss_)
            print("model_save: ", saver.save(sess, 'model_save2\\modle.ckpt'))
            print("The train has finished")
        return theloss

    theloss = train_lstm()

    # ————————————————预测模型————————————————————
    def prediction(time_step=20):

        X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
        mean, std, test_x, test_y = get_test_data(time_step)
        # 用tf.variable_scope来定义重复利用,LSTM会经常用到
        with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
            pred, state_ = lstm(X)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            # 参数恢复（读取已存在模型）
            module_file = tf.train.latest_checkpoint('model_save2')
            saver.restore(sess, module_file)
            test_predict = []
            for step in range(len(test_x) - 1):
                predict = sess.run(pred, feed_dict={X: [test_x[step]], keep_prob: 1})
                predict = predict.reshape((-1))
                test_predict.extend(predict)  # 把predict的内容添加到列表

            # 相对误差=（测量值-计算值）/计算值×100%
            test_y = np.array(test_y) * std[n1] + mean[n1]
            test_predict = np.array(test_predict) * std[n1] + mean[n1]
            acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])
            print("预测的相对误差:", acc)

            print(theloss)
            plt.figure()
            plt.plot(list(range(len(theloss))), theloss, color='b', )
            plt.xlabel('times', fontsize=14)
            plt.ylabel('loss valuet', fontsize=14)
            plt.title('loss-----blue', fontsize=10)
            plt.show()
            # 以折线图表示预测结果
            plt.figure()
            plt.plot(list(range(len(test_predict))), test_predict, color='b', )
            plt.plot(list(range(len(test_y))), test_y, color='r')
            plt.xlabel('time value/day', fontsize=14)
            plt.ylabel('close value/point', fontsize=14)
            plt.title('predict-----blue,real-----red', fontsize=10)
            plt.show()



    prediction()