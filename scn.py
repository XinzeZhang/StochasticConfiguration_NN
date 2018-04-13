#coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import time
import scipy.io as sio
from sklearn.linear_model import Ridge
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ignore warning of futurewarning


##这里定义一些全局的变量
L_max =100
Tmax = 100
Lambdas = [0.5, 1, 5, 10, 30, 50, 100, 150, 200, 250]
Lambdas_len = np.size(Lambdas)
r =  [ 0.9, 0.99, 0.999,0.9999, 0.99999, 0.999999]
r_len =np.size(r)
##############

# 开启Eager Execution
tfe.enable_eager_execution()
# 使用TensorFlow自带的MNIST数据集，第一次会自动下载，会花费一定时间
#mnist = input_data.read_data_sets("/data/mnist", one_hot=True)

# load_data = sio.loadmat('Demo_Iris.mat')
load_data = sio.loadmat('Demo_Data.mat')
train_data = tf.cast(tf.contrib.eager.Variable(load_data['X']),tf.float32).gpu()
train_label = tf.cast(tf.contrib.eager.Variable(load_data['T']),tf.float32).gpu()

test_data = tf.cast(tf.contrib.eager.Variable(load_data['X2']),tf.float32).gpu()
test_label = tf.cast(tf.contrib.eager.Variable(load_data['T2']),tf.float32).gpu()
# 展示信息的间隔
verbose_interval = 500

def constant_variable_weight(shape,stddev,is_var=1):
    if is_var == 1:#等于1取变量，等于0取常量固定
        var = tf.contrib.eager.Variable(tf.truncated_normal(shape,stddev=stddev))
    else:
        var = tf.constant(np.array(np.random.normal(0,stddev,shape),np.float32))
    return var

def constant_variable_biases(shape,value,is_var=1):
    if is_var == 1:
        var = tf.contrib.eager.Variable(tf.constant(value,shape=shape))
    else:
        var = tf.constant(value=value,shape=shape)
    return var

with tf.device("/gpu:0"):
    # 第一层网络的参数，输入为28*28=784维，隐藏层150维
    W0 = constant_variable_weight(shape=[1, 1],stddev=0.1,is_var=0)
    b0 = constant_variable_biases(shape=[1],value=0.1,is_var=0)
    # 第二层网络的参数，一共有10类数字，因此输出为10维
    W1 = constant_variable_weight(shape=[1, 1],stddev=0.1)
    b1 = constant_variable_biases(shape=[1],value=0.1)

    # 构建多层神经网络
    def mlp(step, x, y, E0,is_train = True):
        global W0,b0,W1
        global W0_new,b0_new,find,temp1_array,temp2_array,best_t
        find =0

        m = E0.shape[1] #在mnist的图像分类中应该是等于10的
        for i_Lambdas in range(Lambdas_len):

            Lambda = Lambdas[i_Lambdas]
            W0_new = tf.constant(tf.random_uniform(shape=[1, Tmax], minval=-Lambda, maxval=Lambda)) #从-lambda到lambda随机产生T_max个w0
            b0_new = tf.constant(tf.random_uniform(shape=[Tmax], minval=-Lambda, maxval=Lambda))
            temp1_array =[]
            temp2_array = []
            HT = tf.matmul(x,W0_new) +b0_new
            HT = tf.sigmoid(HT)
            for t in range(Tmax):  # 遍历随机产生的Tmax个变量
                H_t = HT[:, t]
                H_t = tf.reshape(H_t, (-1, 1))

                for i_m in range(m):
                    eq = E0[:, i_m]
                    eq = tf.reshape(eq, (-1, 1))
                    # print("H_t ={}".format(H_t.numpy())+"eq = {}".format(eq.numpy)+"r_L = {}".format(r_L))
                    temp1 = tf.square(tf.matmul(tf.transpose(eq), H_t)) / tf.matmul(tf.transpose(H_t), H_t)
                    temp2 = tf.matmul(tf.transpose(eq), eq)
                    temp1_array.append(temp1)
                    temp2_array.append(temp2)
                    # print("temp = {}".format(temp.numpy())+"r_L = {}".format(r_L)+"temp1 = {}".format(temp1.numpy())+"temp2 = {}".format(temp2.numpy()))
                # if mfind == 1:  # 说明temp的值都大于0
                #     find = 1


            for i_r in range(r_len):
                r_L = r[i_r]
                max = - 1
                for t in range(Tmax):
                    temp = temp1_array[t] - (1 - r_L) * temp2_array[t]
                    temp = temp.numpy()
                    if temp >=0 and temp >max:
                        max = temp
                        best_t = t
                        find =1
                if find:
                    break
            if find:
                W0 = tf.concat([W0, tf.reshape(W0_new[:, best_t], (-1, 1))], 1)
                temp = tf.constant(b0_new[best_t].numpy(), shape=(1))
                b0 = tf.concat([b0, temp], 0)
                break
        if find==0:
            print("End searching")
            return 0


        return 1


    start = time.clock()
    # 执行3000步
    step =1
    loss =10
    tol =0.001
    pltx = np.zeros(L_max)
    plty = np.zeros(L_max)
    E = train_label
    regr = Ridge(alpha=0.000001)
    while (step<=100 )and(loss >tol):
        # 生成128个数据，batch_data是图像像素数据，batch_label是图像label信息
        #batch_data, batch_label = mnist.train.next_batch(128)
        # 梯度下降优化网络参数
        if mlp(step,train_data,train_label,E) ==0:
            break
        # 如果找到了可以添加的隐藏层结点，则用新的来进行计算W1
        hidden = tf.matmul(train_data, W0) + b0
        # print(E-train_label)
        hidden = tf.nn.sigmoid(hidden)
        # print(hidden)
        # 使用最小二乘的方法来对输出权重进行求解

        # W1 = tf.matmul(np.linalg.pinv(hidden), train_label)
        # if step == 11:
        #     sio.savemat('temp.mat',{'hidden':hidden,'train_label':train_label,'W1':W1})
        # temp1 = tf.ones((600,600))*5+ tf.matmul(hidden,tf.transpose(hidden))
        # temp2 = tf.matrix_inverse(temp1)
        # W1 = tf.matmul(tf.transpose(hidden),temp2)
        # W1 = tf.matmul(W1,train_label)

        regr.fit(hidden, train_label)

        logits = regr.predict(hidden)
        loss = tf.sqrt(tf.reduce_mean(tf.square(logits - train_label)))#计算均方根
        E = logits - train_label
        temp = tf.equal(tf.argmax(logits, 1), tf.argmax(train_label, 1))
        acc = tf.reduce_mean(tf.cast(temp,tf.float32))
        print("step ={}".format(step)+"trainning loss = {}".format(loss.numpy())+"  acc = {}".format(acc.numpy()) )
        pltx[step-1] = step
        plty[step-1] = loss.numpy()
        step = step+1

    hidden = tf.matmul(test_data, W0) + b0
    hidden = tf.nn.sigmoid(hidden)
    logits = regr.predict(hidden)
    loss = tf.sqrt(tf.reduce_mean(tf.square(logits - test_label)))  # 计算均方根
    print("test loss = {}".format(loss.numpy()) + "  acc = {}".format(acc.numpy()))
    print("cost_time: %.6f" % (time.clock() - start))

    temp = tf.equal(tf.argmax(logits, 1), tf.argmax(test_label, 1))
    acc = tf.reduce_mean(tf.cast(temp, tf.float32))


    plt.figure(figsize=(8, 4))
    plt.plot(pltx[0:step-1], plty[0:step-1], label="train_loss", color="red", linewidth=2)
    plt.show()
