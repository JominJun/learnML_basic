# Multinomal Classification | 다항 분류
# Softmax Regression

# Softmax function | 확률 값으로 변환하는 과정
# S(yi) = e^(yi) / Σ_j(e^yi)

# Logistic cost VS Cross entropy | 결국에는 같다
# C:(H(x), y) = ylog(H(X)) - (1-y)log(1 - H(x))
# D:(S, L) = -Σ_i(L_i*log(S_i))

# Softmax Classifier의 Cost 함수
# Loss = 1/n * Σ_i(D(S(Wx_i+b), L_i))

# Softmax Classifier의 Gradient descent STEP
# STEP = -α ΔLoss(W1, W2)

# Implemention
import numpy as np
import tensorflow as tf

x_data = [
  [1, 2, 1, 1],
  [2, 1, 3, 2],
  [3, 1, 3, 4],
  [4, 1, 5, 5],
  [1, 7, 5, 5],
  [1, 2, 5, 6],
  [1, 6, 6, 6],
  [1, 7, 7, 7]
]

y_data = [
  [0, 0, 1],
  [0, 0, 1],
  [0, 0, 1],
  [0, 1, 0],
  [0, 1, 0],
  [0, 1, 0],
  [1, 0, 0],
  [1, 0, 0]
]

# Convert into numpy and float format
x_data = np.asarray(x_data, dtype=np.float32)
y_data = np.asarray(y_data, dtype=np.float32)

nb_classes = 3 # 분류 종류의 개수 (y_data의 열의 개수)

# Weight and bias setting
W = tf.Variable(tf.random.normal([4, nb_classes]), name="weight")
b = tf.Variable(tf.random.normal([nb_classes]), name="bias")

variables = [W, b]

# Softmax
def hypothesis(X):
  return tf.nn.softmax(tf.matmul(x_data, W) + b)

# Cross entropy cost/loss
# = tf.nn.softmax_cross_entropy_with_logits_v2 (tf 내장함수)
def cost_fn(X, Y):
  logits = hypothesis(X)
  cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.compat.v1.log(logits), axis=1)) # axis=1은 행단위 계산

  return cost

# Gradient Descent
def grad_fn(X, Y):
  with tf.GradientTape() as tape:
    cost = cost_fn(X, Y)
    grads = tape.gradient(cost, variables)

    return grads

# Train
def fit(X, Y, epochs=2000, verbose=100): # verbose : 출력 주기
  optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)

  for i in range(epochs):
    grads = grad_fn(X, Y)
    optimizer.apply_gradients(zip(grads, variables))

    if (not i) | (not ((i+1)%verbose)):
      print("Loss at epoch %d: %f" % (i+1, cost_fn(X, Y).numpy()))

if __name__ == "__main__":
  fit(x_data, y_data)

  # Prediction
  a = hypothesis(x_data)

  print(a)

  # argmax(_, 0) : 행 방향으로 가장 큰 값을 가지는 열의 인덱스를 반환
  # argmax(_, 1) : 열 방향으로 가장 큰 값을 가지는 행의 인덱스를 반환

  print(tf.argmax(a, 1)) # 학습을 통한 예측 값
  print(tf.argmax(y_data, 1)) # 실제 원하는 값