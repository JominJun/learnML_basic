# =={ Learning Rate }==

# 1. Good & Bad
# High learning rate is called Overshooting
# Normal learning rate is 0.01
# 3e-4(0.0003) is the best learning rate for Adam, hands down (andrej karpathy)
# Learning Rate 설정이 잘못되면 Overfitting(Bad)이 발생할 수 있음

# 2. Learning Rate Decay : Annealing the learning rate
# Loss 값이 줄어들다가 어느 순간이 되면 더 이상 줄어들지 않을 때가 있다.
# 이 때, learning rate를 조절하여 Loss 값이 더 줄어들 수 있게 할 수 있다.

# [Methods]
# - 01. Step Decay
# : Step 별로 특정 Epoch만큼 Learning rate 값을 조절하는 방식
# : N epoch or validation loss

# - 02. Exponential Decay
# : α = α0 * e^(-kt)

# - 03. 1/t Decay
# : Epoch 분(分)의 1로 조절하는 방식
# : α = α0 / (1+kt)

# [Tensorflow Code]
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase) # 시작 값, 전체 step, step 주기, 변화 값, staricase

# [Other Methods' Tensorflow Code]
# tf.train.exponential_decay  | 지수
# tf.train.inverse_time_decay | 1/t Decay
# tf.train.natural_exp_decay  | 자연상수(e)
# tf.train.piecewise_constant | 연속된 함수를 일정한 조각으로 나누는 기법

# * tf.train.polynomial_decay
# polynomial(다항의) ↔ exponential(지수의)
# big O-notation 에서의 함수 개형을 보면 O(n^2) < O(2^n)
# polynomial 함수가 exponential 함수보다 완만하게 learning rate 값을 조절함

# [Python Code]
# def exponential_decay(epoch):
#   starter_rate = 0.01
#   k = 0.96
#   exp_rate = starter_rate * exp(-k*t)
#   return exp_rate


# =={ Data Preprocessing }==

# Noisy Data. 즉, 쓸모없는 데이터들을 잘 처리해주는 것이 가장 중요!

# 1. Feature Scaling
# 대부분의 값이 한 쪽에 몰려있는데 일부 값들이 다른 쪽에 위치할 때, 해당 값들의 분포까지 잘 나타내게 하기 위한 방법
# ex) 단위가 서로 다른 값에 대하여 => 174(cm) VS 88(kg)

# - 01. Standardization (Mean distance) : 표준화 기법 (주로 사용)
# : X_new = (x-μ)/σ
# : X값을 평균으로 뺀 것을 표준편차로 나눈다 = 평균에서 얼마나 떨어져 있는가?

# - 02. Normalization (0~1) : 정규화 기법
# : X_new = (X - X_min) / (X_max - X_min)
# : 실제 X값을 최솟값으로 뺀 것을 최댓값과 최솟값의 차로 나눈 것

# [Python Code(numpy)]
# Standardization = (data - np.mean(data)) / sqrt(np.sum(data - np.mean(data))^2) / np.count(data)
# Normalization = (data - np.min(data, 0)) / (np.max(data, 0) - np.min(data, 0))