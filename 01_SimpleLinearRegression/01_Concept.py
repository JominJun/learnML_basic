# Simple Linear Regression | 간단 선형 회귀
# https://www.boostcourse.org/ai212/lecture/41844/

import tensorflow as tf

# input & output
x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]

# y = Wx + b
W = tf.Variable(2.9) # 임의의 Initial 값을 세팅함
b = tf.Variable(0.5)

# 가설 함수 | y = Wx + b
hypothesis = W * x_data + b

# 비용 함수 | (hypothesis - y)^2의 평균
cost = tf.reduce_mean(tf.square(hypothesis - y_data))