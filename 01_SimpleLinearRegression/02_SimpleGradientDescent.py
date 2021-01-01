# Gradient descent | 경사 하강법
# x_data와 y_data를 가지고 '경사를 줄이며' W값과 b값을 찾아가는 과정

# Convex function | global minimum과 local minimum이 일치하는 경우
# 위 경우에만 Gradient descent를 활용하는 게 좋음

import tensorflow as tf

# 초기 설정
x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]

W = tf.Variable(2.9)
b = tf.Variable(0.5)

learning_rate = 0.01 # 미분 값을 구할 주기를 정함

# 100번 반복
for i in range(1000+1):
  # 구현
  with tf.GradientTape() as tape: # tf.Gradient()를 이용하여 tape에 저장
    hypothesis = W * x_data + b
    cost = tf.reduce_mean(tf.square(hypothesis - y_data))

  # W Gradinet, b Gradient
  W_grad, b_grad = tape.gradient(cost, [W, b]) # cost 함수에 대하여 W, b의 미분 값(Gradient)

  W.assign_sub(learning_rate * W_grad) # W -= learning_rate * W_grad
  b.assign_sub(learning_rate * b_grad) # b -= learning_rate * b_grad

  # 진행 상태 출력
  if not i % 10:
    print("{:5}|{:10.4}|{:10.4}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))