# https://www.youtube.com/watch?v=KOic-GozMTo&list=PLjSwXXbVlK6IHzhLOMpwHHLjYmINRstrk&index=2
import tensorflow as tf
import numpy as np

# 使用numpy生成100個隨機點
# rand函数根据给定维度生成[0,1]之间的数据
x_data = np.random.rand(100)
y_data = x_data*0.1 + 0.2

# 構造一個線性模型
# 初始值不管多少都會接近訓練結果
# b = tf.Variable(0.)
# k = tf.Variable(0.)
b = tf.Variable(1.1)
k = tf.Variable(0.5)
y = k*x_data + b

# 二次代價函數
loss = tf.reduce_mean(tf.square(y_data - y)) # reduce_mean求平均值
# 定義一個梯度下降法來進行訓練的優化器
optimizaer = tf.train.GradientDescentOptimizer(0.2)
# 最小化代價函數, loss越小越接近真實值
train = optimizaer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20 ==0:
            print(step, sess.run([k, b]))