# Tensorflow Implemention of Learning rate, Data prerpocessing, Overfitting & Solutions

import numpy as np
import tensorflow as tf

tf.random.set_seed(777)

xy = np.array([
  [828.659973, 833.450012, 908100, 828.349976, 831.659973],
  [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
  [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
  [816, 820.958984, 1008100, 815.48999, 819.23999],
  [819.359985, 823, 1188100, 818.469971, 818.97998],
  [819, 823, 1198100, 816, 820.450012],
  [811.700012, 815.25, 1098100, 809.780029, 813.669983],
  [809.51001, 816.659973, 1398100, 804.539978, 809.559998]
])

# Data preprocessing - Normalization
def normalization(data):
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)

  return numerator / denominator

xy = normalization(xy)

x_train = xy[:, 0:-1]
y_train = xy[:, [-1]]

# tf.data.api를 이용하여 반복적인 학습을 수행
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))

W = tf.Variable(tf.random.normal((4, 1)), dtype=tf.float32)
b = tf.Variable(tf.random.normal((1,)), dtype=tf.float32)

def linearReg_fn(features):
  hypothesis = tf.matmul(features, W) + b
  return hypothesis

def l2_loss(loss, beta=0.01): # loss term
  W_reg = tf.nn.l2_loss(W) # Output = sum(t**2) / 2
  loss = tf.reduce_mean(loss + W_reg * beta)

  return loss

def loss_fn(hypothesis, features, labels, flag=False):
  cost = tf.reduce_mean(tf.square(hypothesis - labels))

  if flag:
    cost = l2_loss(cost)
  
  return cost

is_decay = True
starter_learning_rate = 0.1

if is_decay:
  learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=starter_learning_rate, decay_steps=50, decay_rate=0.96, staircase=True) # 50회마다 0.96을 곱함
  optimizer = tf.keras.optimizers.SGD(learning_rate)
else:
  optimizer = tf.keras.optimizers.SGD(learning_rate=starter_learning_rate)

def grad(hypothesis, features, labels, l2_flag):
  with tf.GradientTape() as tape:
    loss_value = loss_fn(linearReg_fn(features), features, labels, l2_flag)

  return tape.gradient(loss_value, [W, b]), loss_value

def fit(epochs, verbose):
  for step in range(epochs):
    for features, labels in dataset:
      features = tf.cast(features, tf.float32)
      labels = tf.cast(labels, tf.float32)

      grads, loss_value = grad(linearReg_fn(features), features, labels, False)
        
      optimizer.apply_gradients(grads_and_vars=zip(grads, [W, b]))

      if (not step) | (not ((step+1) % verbose)):
        print("Iter: {}, Loss: {:.4f}".format(step+1, loss_value))

if __name__ == "__main__":
  fit(1000, 10)