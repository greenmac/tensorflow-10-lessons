# https://www.youtube.com/watch?v=KOic-GozMTo&list=PLjSwXXbVlK6IHzhLOMpwHHLjYmINRstrk&index=2
import tensorflow as tf

# 創建一個常數op
m1 = tf.constant([[3, 3]])
# 創建一個常數op
m2 = tf.constant([[2], [3]])
# 創建一個矩陣乘法op, 把m1和m2傳入
product = tf.matmul(m1, m2)

# 定義一個會話, 啟動默認圖
with tf.Session() as sess:
    # 調用sess的run方法來執行矩陣乘法op
    # run(product)觸發了圖中3個op
    result = sess.run(product)
    print(result)
