# Gradient Descent Implemention

import tensorflow as tf

tf.compat.v1.set_random_seed(0) # 반복해도 같은 결과를 얻을 수 있도록 랜덤 값을 설정함

X = [1., 2., 3., 4.]
Y = [1., 3., 5., 7.,]

W = tf.Variable([tf.random.normal([1], -100., 100.)])

for step in range(300+1):
  hypothesis = W * X
  cost = tf.reduce_mean(tf.square(hypothesis - Y))

  alpha = 0.01 # learning rate
  gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X) - Y, X))
  descent = W - tf.multiply(alpha, gradient)
  W.assign(descent)

  if not step % 10:
    print("{:5} | {:10.4f} | {:10.6f}".format(step, cost.numpy(), W.numpy()[0]))