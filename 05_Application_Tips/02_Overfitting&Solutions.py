# =={ Concepts }==

# Overfitting
# 학습이 과하게 되어 High variance가 발생하며 Test 값을 넣었을 때에 Validation 때와 다르게 만족하지 못하는 값이 나오는 경우

# Underfitting
# 학습이 덜 되어 Hight bias가 발생


# =={ Solutions }==

# 1. Feature Normalization

# - 01. Get more data
# : more data will actually make a difference, (helps to fix high variance)

# - 02. Smaller set of features
# : dimensionality reduction(PCA) (fixes high variance) : 차원을 줄여서 각각의 데이터가 가지고 있는 특징들을 더 강화

#   [Scikit-learn Code]
#   from sklearn.decomposition import PCA
#   pca = decomposition.PCA(n_components=3)
#   pca.fit(X)
#   X = pca.transform(X)

# - 03. Add additional features : For Underfitting
# : hypothesis is too simple, make hypothesis more specific (fixes high bias) : 적절한 값을 찾아야 함. 너무 많아도 문제고 적어도 문제


# 2. Regularization
# : add term to loss (특정 값을 추가하여 정규화)

# λ-- : fixes high bias (Underfitting)
# λ++ : fixes high variance (Overfitting)

# Linear regression with regularization
# Cost Function = ((hypothesis - y)의 평균) + (λ/2m) * Σ_j=1(to m) (θ_j)^2 : 이 식은 이해가 안되지만 쨌든 부족한 녀석들은 채워주고 많은 녀석들은 그대로 두는 듯 하다

# [Tensorflow Code]
# L2_loss = tf.nn.l2_loss(W) : output = sum(t**2) / 2


# 3. Other Solutions for Overfitting

# - 01. More Data (Data Augmentation)
# -- Color Jittering      # 색상의 다양화 (이미지의 경우)
# -- Horizontal Flips     # 뒤집기
# -- Random Crops/Scales  # 적절한 크기로 자르거나 조절하기

# - 02. Dropout (0.5 is common) | 9강 참고
# - 03. Batch Normalization     | 9강 참고