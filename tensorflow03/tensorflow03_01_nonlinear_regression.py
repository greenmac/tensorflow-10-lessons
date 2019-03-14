# https://www.youtube.com/watch?v=k3O0VCHxw10&index=3&list=PLjSwXXbVlK6IHzhLOMpwHHLjYmINRstrk
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成200個隨機點
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data)+ noise

# 定義兩個placeholder
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定義神經網路中間層
Weights_L1 = tf.Variable(tf.random.normal([1, 10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1) # 激勵函數, 雙取正切函數

# 定義神經網路輸出層
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2) # 激勵函數, 雙取正切函數

# 二次代價函數
loss = tf.reduce_mean(tf.square(y-prediction))
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 變量初始化
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step, feed_dict={x:x_data, y:y_data})

    # 獲得預測值
    prediction_value = sess.run(prediction, feed_dict={x:x_data})
    # 畫圖
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()