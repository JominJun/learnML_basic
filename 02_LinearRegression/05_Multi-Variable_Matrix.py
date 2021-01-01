# Matrix를 이용한 Multi-Variable Linear Regression

import numpy as np
import tensorflow as tf

data = np.array([
  # X1, X2, X3, Y
  [73., 80., 75., 152.],
  [93., 88., 93., 185.],
  [89., 91., 90., 180.],
  [73., 66., 70., 142.]
], dtype=np.float32)

# slice data
X = data[:, :-1] # X1, X2, X3
Y = data[:, [-1]]

W = tf.Variable(tf.random.normal([3, 1]))
b = tf.Variable(tf.random.normal([1]))

learning_rate = 0.000001

# hypothesis, prediction function
def predict(X):
  return tf.matmul(X, W) + b # Matrix Multiply

n_epochs = 2000

for i in range(n_epochs+1):
  with tf.GradientTape() as tape:
    cost = tf.reduce_mean(tf.square(predict(X) - Y))

  W_grad, b_grad = tape.gradient(cost, [W, b])

  W.assign_sub(learning_rate * W_grad)
  b.assign_sub(learning_rate * b_grad)

  if not i % 100:
    print("{:5} | {:10.4f}".format(i, cost.numpy()))