import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

normal = np.random.multivariate_normal
# Number of samples
nSamples = 500
# (unit) variance:
s2 = 1
# below, we provide the coordinates of the mean as
# a first argument, and then the covariance matrix
# we generate nexamples examples for each category
sgx0 = normal([0.,0.], [[s2, 0.], [0.,s2]], nSamples)
sgx1 = normal([2.,2.], [[s2, 0.], [0.,s2]], nSamples)
# setting the labels for each category
sgy0 = np.zeros((nSamples,))
sgy1 = np.ones((nSamples,))
#xyz
sgx = np.concatenate((sgx0, sgx1))
sgy = np.concatenate((sgy0, sgy1))

# define parameters
# bias:
b = 0
# x1 weight:
w1 = 1
# x2 weight:
w2 = 1

def sigmoid_2d(x1, x2):
    # z is a linear function of x1 and x2
    z = w1*x1 + w2*x2 + b
    return 1 / (1+np.exp(-z))

xmin, xmax, npoints = (-6,6,51)
linx1 = np.linspace(xmin,xmax,npoints)
# no need for a new array, we just reuse the one we have with another name:
linx2 = linx1

gridx1, gridx2 = np.meshgrid(np.linspace(xmin,xmax,npoints), np.linspace(xmin,xmax,npoints))
# print(gridx1.shape, gridx2.shape)
# print('gridx1:')
# print(gridx1)
# print('gridx2')
# print(gridx2)
#
# z = sigmoid_2d(gridx1, gridx2)
# plt.pcolor(gridx1, gridx2, z)
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.colorbar()
# plt.show()

clf = LogisticRegression(solver='lbfgs')  #clf: classifier
clf.fit(sgx, sgy)
print(gridx1.shape, gridx2.shape)

grid = np.c_[gridx1.flatten(), gridx2.flatten()]
prob = clf.predict_proba(grid)
plt.pcolor(gridx1,gridx2,prob[:,1].reshape(npoints,npoints))
plt.colorbar()
plt.scatter(sgx0[:,0], sgx0[:,1], alpha=0.5)
plt.scatter(sgx1[:,0], sgx1[:,1], alpha=0.5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
