# Logistic Regression | 로지스틱 회귀
# => 값을 0, 1로 딱딱 나누려는 경우에 사용
# => Linear Regression -> Sigmoid (0과 1 사이의 값으로 구별) -> Boundary Decision (Threshold: 임계값)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x_train = [[1., 2.], [2., 3.], [3., 1.], [4., 3.], [5., 3.], [6., 2.]]
y_train = [[0.], [0.], [0.], [1.], [1.], [1.]]

x_test = [[5., 2.]]
y_test = [[1.]]

x1 = [x[0] for x in x_train] # 1, 2, 3 ...
x2 = [x[1] for x in x_train] # 2, 3, 1 ...

colors = [int(y[0] % 3) for y in y_train]

plt.scatter(x1, x2, c=colors, marker='^')
plt.scatter(x_test[0][0], x_test[0][1], c="red")

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

W = tf.Variable(tf.zeros([2, 1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

# sigmoid(x) = 1 / (1 + e^(-x)) : x는 Wx + b (Linear Regression)
def logistic_regression(features):
  # tf.exp = 지수(exponential)
  hypothesis = tf.divide(1., 1. + tf.exp(-tf.matmul(features, W) + b))
  return hypothesis

# cost(h(x), y) = -ylog(h(x)) - (1-y)log(1-h(x))
def loss_fn(hypothesis, labels):
  cost = -tf.reduce_mean(labels * tf.math.log(hypothesis) + (1 - labels) * tf.math.log(1 - hypothesis))
  return cost

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

def accuracy_fn(hypothesis, labels):
  predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32) # tf.cast = 형변환
  accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32)) # predicted와 labels의 일치 여부를 확인하여 정확도를 구함
  return accuracy

# 미분값을 저장한 테이프를 반환
def grad(features, labels):
  with tf.GradientTape() as tape:
    hypothesis = logistic_regression(features) # sigmoid func
    loss_value = loss_fn(hypothesis, labels) # cost func
  
  return tape.gradient(loss_value, [W, b])

EPOCHS = 1000

for step in range(EPOCHS+1):
  for features, labels in iter(dataset.batch(len(x_train))): # x_train 개수만큼 묶기
    hypothesis = logistic_regression(features)
    grads = grad(features, labels) # x: features, y: labels

    optimizer.apply_gradients(grads_and_vars=zip(grads, [W, b])) # gradient 수정

    if not step % 100:
      print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(hypothesis, labels)))

# TEST
test_acc = accuracy_fn(logistic_regression(x_test), y_test)

print("Test Result = {}".format(tf.cast(logistic_regression(x_test) > 0.5, dtype=tf.int32)))
print("Testset Accuracy: {:.4f}".format(test_acc))