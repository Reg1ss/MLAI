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

K = int(degree) + 1 #+1 for the design metrix
z_vec = np.random.normal(size=K)
w_sample = z_vec*np.sqrt(alpha)
# print(w_sample)
#
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
    # plt.plot(x_pred.flatten(), f_sample.flatten())  #flatten降维到1维
# plt.show()

#Question 1
sigma2 = 0.01
w_cov = np.linalg.inv(1/sigma2*np.dot(Phi.T,Phi)+1/alpha*np.eye(K))
w_mean = np.dot(w_cov,np.dot(Phi.T,y))*1/sigma2
# print(w_cov)
# print(w_mean)

w_sample = np.random.multivariate_normal(w_mean.flatten(), w_cov)   #multivariate gaussian distribution
# print(w_sample.shape)
f_sample = np.dot(Phi_pred,w_sample)
# plt.plot(x_pred.flatten(), f_sample.flatten(), 'r-')
# plt.plot(x, y, 'rx') # plot data to show fit.
# plt.show()

for i in range(num_samples):
    w_sample = np.random.multivariate_normal(w_mean.flatten(), w_cov)
    f_sample = np.dot(Phi_pred,w_sample)
#     plt.plot(x_pred.flatten(), f_sample.flatten())
# plt.plot(x, y, 'rx') # plot data to show fit.
# plt.show()

K = 10 # how many Gaussians to add.
num_samples = 1000 # how many samples to have in y_vec
mus = np.linspace(0, 5, K) # mean values generated linearly spaced between 0 and 5
sigmas = np.linspace(0.5, 2, K) # sigmas generated linearly spaced between 0.5 and 2
y_vec = np.zeros(num_samples)
total = 0
for mu, sigma in zip(mus, sigmas):  #zip package the elements into tuple and return a list of the tuple
    z_vec = np.random.normal(size=num_samples) # z is from standard normal
    y_vec += z_vec*sigma + mu # add to y z*sigma + mu
    total += mu
# print(total)

# now y_vec is the sum of each scaled and off set z.
print('Sample mean is ', y_vec.mean(), ' and sample variance is ', y_vec.var())
print('True mean should be ', mus.sum())
print('True variance should be ', (sigmas**2).sum(), ' standard deviation ', np.sqrt((sigmas**2).sum()))

# plt.hist(y_vec, bins=30, normed=True)
# plt.legend('$y$')
# plt.show()

# Question 2
# compute mean under posterior density
f_pred_mean = np.dot(Phi_pred,w_mean)

# plot the predictions

# compute mean at the training data and sum of squares error
f_mean = np.dot(Phi,w_mean)
sum_squares = np.sum((f_mean-y)**2)
print('The error is: ', sum_squares)

#Question 3
# print(Phi[0,:].T @ w_cov @ Phi[0,:])
# Compute variance at function values
f_pred_var = (Phi_pred @ w_cov @ Phi_pred.T).diagonal()
f_pred_std = np.sqrt(f_pred_var)
# print(f_pred_std)

# plot the mean and error bars at 2 standard deviations above and below the mean
plt.plot(x_pred,f_pred_mean,'black')
plt.plot(x,y,'rx')
plt.errorbar(x_pred,f_pred_mean,yerr=f_pred_std)    #errorbarx：数据点的水平位置 y：数据点的垂直位置 yerr：y轴方向的数据点的误差计算方法 xerr：x轴方向的数据点的误差计算方法
plt.show()
