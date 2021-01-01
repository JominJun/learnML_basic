# Fancy Softmax Classifier
# Animal Classification with softmax_cross_entropy_with_logits
# Predicting animal type based on various features

# ONE-HOT & Reshape
# * If the input indices rank N, the output will have rank N+1.
# * The new axis is created at dimension axis (default: the new axis is appended at the end)

# How to Solve?
# => tf.reshape(Y_one_hot, [-1, number_classes]) # -1은 전체를 대상으로 함을 뜻함

# Example (number_classes = 7)
# raw: ([0], [3]) => (n, 1)
# after ONE-HOT: ([[[1, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 1, 0, 0, 0]]]) => (n, 1, 7)
# after Reshape: ([[1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]) => (n, 7)

# Implemention
import numpy as np
import tensorflow as tf

# Load Dataset
xy = np.loadtxt("C:\Programming\Python\AI\learnML_basic\data-04-zoo.csv", delimiter=",", dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# 오류 해결을 위해
# Error Msg | InvalidArgumentError: Value for attr 'TI' of float is not in the list of allowed values: uint8, int32, int64; NodeDef: {{node OneHot}}; Op<name=OneHot; signature=indices:TI, depth:int32, on_value:T, off_value:T -> output:T; attr=axis:int,default=-1; attr=T:type; attr=TI:type,default=DT_INT64,allowed=[DT_UINT8, DT_INT32, DT_INT64]> [Op:OneHot]
y_data = tf.dtypes.cast(y_data, tf.int32)

nb_classes = 7

# Make Y data as onehot shape
Y_one_hot = tf.one_hot(list(y_data), nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

# Softmax Classifier
# weight & bias setting
W = tf.Variable(tf.random.normal([16, nb_classes]), name="weight")
b = tf.Variable(tf.random.normal([nb_classes]), name="bias")

variables = [W, b]

# tf.nn.softmax computes softmax activations
def logit_fn(X):
  return tf.matmul(X, W) + b

def hypothesis(X):
  return tf.nn.softmax(logit_fn(X))

def cost_fn(X, Y):
  logits = logit_fn(X)
  cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)

  cost = tf.reduce_mean(cost_i)
  return cost

def grad_fn(X, Y):
  with tf.GradientTape() as tape:
    loss = cost_fn(X, Y)
    grads = tape.gradient(loss, variables)
    
    return grads

def prediction(X, Y):
  pred = tf.argmax(hypothesis(X), 1)
  correct_prediction = tf.equal(pred, tf.argmax(Y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  return accuracy

def fit(X, Y, epochs=1000, verbose=100):
  optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)

  for i in range(epochs):
    grads = grad_fn(X, Y)
    optimizer.apply_gradients(zip(grads, variables))

    if (not i) | (not ((i+1) % verbose)):
      acc = prediction(X, Y).numpy()
      loss = tf.reduce_sum(cost_fn(X, Y)).numpy()

      print("Loss & Acc at {} epoch {}, {}".format(i+1, loss, acc))

# Run
if __name__ == "__main__":
  fit(x_data, Y_one_hot)