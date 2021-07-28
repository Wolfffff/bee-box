import numpy as np

def euclidean(array_x, array_y):
	n = array_x.shape[0]
	ret = 0.
	for i in range(n):
		ret += (array_x[i]-array_y[i])**2
	return np.sqrt(ret)

print(euclidean(np.array([[0, 0], [0, 0]]), np.array([[1, 1], [2, 2]])))
print(euclidean(np.array([[-1, 1], [1, 1]]), np.array([[1, 2], [1, 2]])))