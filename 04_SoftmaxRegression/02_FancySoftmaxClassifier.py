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

# 