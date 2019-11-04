import numpy as np # import numpy for the arrays.
import pylab as plt
import pods


# def quadratic(x,n):
#     """Take in a vector of input values and return the design matrix associated
#     with the basis functions."""
#     return np.hstack([np.ones((n, 1)), x, x**2])
#
# # ensure plots appear in the notebook.
# # first let's generate some inputs
# n = 100
# x = np.zeros((n, 1))  # create a data set of zeros
# x[:, 0] = np.linspace(-1, 1, n) # fill it with values between -1 and 1
#
# Phi = quadratic(x,100)
#
# fig, ax = plt.subplots(figsize=(12,4))
# ax.set_ylim([-1.2, 1.2]) # set y limits to ensure basis functions show.
# ax.plot(x[:,0], Phi[:, 0], 'r-', label = '$\phi_1(x)=1$')
# ax.plot(x[:,0], Phi[:, 1], 'g-', label = '$\phi_2(x) = x$')
# ax.plot(x[:,0], Phi[:, 2], 'b-', label = '$\phi_3(x) = x^2$')
# ax.legend(loc='lower right')
# ax.set_title('Quadratic Basis Functions')
# plt.show()

#fitting to data
data = pods.datasets.olympic_marathon_men()
y = data['Y']
x = data['X']
y -= y.mean()
y /= y.std()

def polynomial(x, num_basis=4, data_limits=[-1., 1.]):
    "Polynomial basis"
    centre = data_limits[0]/2. + data_limits[1]/2.
    span = data_limits[1] - data_limits[0]
    z = x - centre
    z = 2*z/span
    Phi = np.zeros((x.shape[0], num_basis))
    for i in range(num_basis):
        Phi[:, i:i+1] = z**i
    return Phi


def radial(x, num_basis=4, data_limits=[-1., 1.]):
    "Radial basis constructed using exponentiated quadratic form."
    if num_basis > 1:
        centres = np.linspace(data_limits[0], data_limits[1], num_basis)
        width = (centres[1] - centres[0]) / 2.
    else:
        centres = np.asarray([data_limits[0] / 2. + data_limits[1] / 2.])
        width = (data_limits[1] - data_limits[0]) / 2.

    Phi = np.zeros((x.shape[0], num_basis))
    for i in range(num_basis):
        Phi[:, i:i + 1] = np.exp(-0.5 * ((x - centres[i]) / width) ** 2)
    return Phi

def fourier(x, num_basis=4, data_limits=[-1., 1.]):
    "Fourier basis"
    tau = 2*np.pi
    span = float(data_limits[1]-data_limits[0])
    Phi = np.zeros((x.shape[0], num_basis))
    for i in range(num_basis):
        count = float((i+1)//2)
        frequency = count/span
        if i % 2:
            Phi[:, i:i+1] = np.sin(tau*frequency*x)
        else:
            Phi[:, i:i+1] = np.cos(tau*frequency*x)
    return Phi

# Phi = polynomial(x,9,[1896.,2012.])
Phi = radial(x,4,[1896.,2012.])
# Phi = fourier(x,7,[1896.,2012.])
# print(Phi.shape)
# print(np.dot(Phi.T,Phi))
w = np.linalg.solve(np.dot(Phi.T,Phi),np.dot(Phi.T,y))
# print(w.shape)
error = 0
prediction = np.dot(Phi,w)
error = ((y - prediction)**2).sum()
print("Objective error is: ",error)
print(w)
plt.plot(x,y,'rx')
plt.plot(x, prediction, 'g-')
plt.show()

#QR Decomposition
# Phi = polynomial(x,9,[1896.,2012.])
# # Phi = radial(x,8,[1896.,2012.])
# # Phi = fourier(x,7,[1896.,2012.])
# # print(Phi.shape)
# # print(np.dot(Phi.T,Phi))
# Q,R = np.linalg.qr(Phi)
# w = np.linalg.solve(R,np.dot(Q.T,y))



