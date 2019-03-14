# https://www.youtube.com/watch?v=k3O0VCHxw10&index=3&list=PLjSwXXbVlK6IHzhLOMpwHHLjYmINRstrk
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 載入數據集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # one_hot某一位是1, 其他是0的格式

# 每個批次的大小
batch_size = 100
# 計算一共有多個梯次
n_batch = mnist.train.num_examples // batch_size

# 定義兩個placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10]) # 0~9數字,所以輸出是10個

# 創建一個簡單的神經網路
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W)+b)

# 二次代價函數
loss = tf.reduce_mean(tf.square(y-prediction))
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化變量
init = tf.global_variables_initializer()

# 存放在一個布爾型列表中
correct_prediciton = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1)) # argmax返回一維張量中最大的值所在的位置, 相同是True, 不同是False, 這裡是求最大的值為第幾個位置, 相當於哪個標籤
# 求準確率
accuracy = tf.reduce_mean(tf.cast(correct_prediciton, tf.float32)) # 轉換格式 True=1, False=0

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(201):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) # 記住上次給的最後的位置
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})

        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels}) # 餵進去測試集裡的圖片, 測試集裡的標籤
        print("Iter " + str(epoch) + "Testing Accuracy " + str(acc))