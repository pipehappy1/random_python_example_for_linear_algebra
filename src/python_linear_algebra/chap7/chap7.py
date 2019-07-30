import numpy as np

# 首先，我们需要定义数据集和学习率
# Size of the points dataset.
m = 20
# Points x-coordinate and dummy value (x0, x1).
X0 = np.ones((m, 1))
X1 = np.arange(1, m+1).reshape(m, 1)
X = np.hstack((X0, X1))
# Points y-coordinate
y = np.array([
	3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
	11, 13, 13, 16, 17, 18, 17, 19, 21
]).reshape(m, 1)
# The Learning Rate alpha.
alpha = 0.01

# 接下来我们以矩阵向量的形式定义代价函数和代价函数的梯度
def error_function(theta, X, y):
	'''Error function J definition.'''
	diff = np.dot(X, theta) - y
	return (1./2*m) * np.dot(np.transpose(diff), diff)
def gradient_function(theta, X, y):
	'''Gradient of the function J definition.'''
	diff = np.dot(X, theta) - y
	return (1./m) * np.dot(np.transpose(X), diff)

# 最后就是算法的核心部分，梯度下降迭代计算
def gradient_descent(X, y, alpha):
	'''Perform gradient descent.'''
	theta = np.array([1, 1]).reshape(2, 1)
	gradient = gradient_function(theta, X, y)
	while not np.all(np.absolute(gradient) <= 1e-5):
		theta = theta - alpha * gradient
		gradient = gradient_function(theta, X, y)
	return theta

# 当梯度小于1e–5时，说明已经进入了比较平滑的状态，
# 这时候再继续迭代效果不大，所以可退出循环！
optimal = gradient_descent(X, y, alpha)
print('optimal:', optimal)
print('error function:', error_function(optimal, X, y)[0,0])
