# Cost function in pure Python
# 순수 파이썬으로 간략화된 코스트 함수 구현하기 (b 제외)

import numpy as np

X = np.array([1, 2, 3])
Y = np.array([1, 2, 3])

def cost_func(W, X, Y):
  c = 0

  for i in range(len(X)):
    c += (W * X[i] - Y[i]) ** 2

  return c / len(X)

for feed_W in np.linspace(-3, 5, num=15): # -3부터 5까지의 숫자를 15개로 나눔
  curr_cost = cost_func(feed_W, X, Y)
  print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))