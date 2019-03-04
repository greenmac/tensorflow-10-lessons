# https://www.youtube.com/watch?v=KOic-GozMTo&list=PLjSwXXbVlK6IHzhLOMpwHHLjYmINRstrk&index=2
import tensorflow as tf

x = tf.Variable([1, 2])
a = tf.constant([3, 3])
# 增加一個減法op
sub = tf.subtract(x, a)
# 增加一個加法op
add = tf.add(x, sub)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # print(sess.run(sub))
    # print(sess.run(add))

# 創建一個變量初始化為0
state = tf.Variable(0, name='counter')
# 創建一個op, 作用是使state加1
new_value = tf.add(state, 1)
# 賦值op
update = tf.assign(state, new_value)
# 變量初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    print(sess.run(init))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))