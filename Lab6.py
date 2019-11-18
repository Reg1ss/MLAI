import numpy as np
import matplotlib.pyplot as plt
import pods

# set prior variance on w
alpha = 4.
# set the order of the polynomial basis set
order = 5
# set the noise variance
sigma2 = 0.01

# data = pods.datasets.olympic_marathon_men()
# x = data['X']
# y = data['Y']
data = np.loadtxt('Datasets/olympic_marathon_men.csv', delimiter=',')
x = data[:, 0, None]
y = data[:, 1, None]
num_data = x.shape[0]
num_pred_data = 100 # how many points to use for plotting predictions
x_pred = np.linspace(1890, 2016, num_pred_data)[:, None] # input locations for predictions
# print(x_pred)

def polynomial(x, degree, loc, scale):
    degrees = np.arange(degree+1)
    return ((x-loc)/scale)**degrees

loc = 1950.
scale = 1.
degree = 5.
Phi_pred = polynomial(x_pred, degree=degree, loc=loc, scale=scale)
Phi = polynomial(x, degree=degree, loc=loc, scale=scale)

w_vec = np.random.normal(size=200)
print('w sample mean is ', w_vec.mean())
print('w sample variance is ', w_vec.var())

phi = 7
f_vec = phi*w_vec
print('True mean should be phi*0 = 0.')
print('True variance should be phi*phi*1 = ', phi*phi)
print('f sample mean is ', f_vec.mean())
print('f sample variance is ', f_vec.var())

mu = 4 # mean of the distribution
alpha = 2 # variance of the distribution
w_vec = np.random.normal(size=200)*np.sqrt(alpha) + mu  #np.sqrt() extraction of a root
print('w sample mean is ', w_vec.mean())
print('w sample variance is ', w_vec.var())

z_vec = np.random.normal(size=1000) # by convention, in statistics, z is often used to denote samples from the standard normal
w_vec = z_vec*np.sqrt(alpha) + mu
# plot normalized histogram of w, and then normalized histogram of z on top
# plt.hist(w_vec, bins=30, density=True)
# plt.hist(z_vec, bins=30, density=True)
# plt.legend(('$w$', '$z$'))
# plt.show()

K = int(degree) + 1
z_vec = np.random.normal(size=K)
w_sample = z_vec*np.sqrt(alpha)
print(w_sample)

# f_sample = np.dot(Phi_pred,w_sample)
# plt.plot(x_pred.flatten(), f_sample.flatten(), 'r-')
# plt.show()

scale = 100.
Phi_pred = polynomial(x_pred, degree=degree, loc=loc, scale=scale)
Phi = polynomial(x, degree=degree, loc=loc, scale=scale)
f_sample = np.dot(Phi_pred,w_sample)
# plt.plot(x_pred.flatten(), f_sample.flatten(), 'r-')
# plt.show()

num_samples = 10
K = int(degree)+1
for i in range(num_samples):
    z_vec = np.random.normal(size=K)
    w_sample = z_vec*np.sqrt(alpha)
    f_sample = np.dot(Phi_pred,w_sample)
    plt.plot(x_pred.flatten(), f_sample.flatten())  #flatten降维到1维
plt.show()