import numpy as np
import pods
import pylab as plt

data = pods.datasets.olympic_marathon_men()
x = data['X']
y = data['Y']
plt.plot(x, y, 'rx')
# plt.show()

#Question 1
def linear(x, num_basis=2):
    Phi = np.zeros((x.shape[0], num_basis))
    for i in range(num_basis):
        Phi[:, i:i+1] = x**i
    return Phi

def prediction(w, x, linear):
    Phi = linear(x)
    f = np.dot(Phi,w)
    return f

def objective(w, x, y, linear):
    e = np.dot((y - prediction(w,x,linear)).T,y - prediction(w,x,linear))
    return e.sum()

def fit(x, y, linear):
    w = np.linalg.solve(np.dot(linear(x).T,linear(x)),np.dot(linear(x).T,y))
    return w

x_for_predict = np.zeros((2020-1890+1,1))
for i in range(x_for_predict.shape[0]):
    x_for_predict[i,0] = i+1890

w = fit(x,y,linear)
prediction_for_Q1 = prediction(w,x_for_predict,linear)
# print('training error for linear model:')
# print(objective(w,x,y,linear))
# plt.plot(x_for_predict,prediction_for_Q1,'b-')
# plt.show()

#Question 2
def quadratic(x, num_basis=3):
    Phi = np.zeros((x.shape[0], num_basis))
    for i in range(num_basis):
        Phi[:, i:i + 1] = x ** i
    return Phi

w = fit(x,y,quadratic)
prediction_for_Q2 = prediction(w,x_for_predict,quadratic)
# print('training error for quadratic model:')
# print(objective(w,x,y,quadratic))
# plt.plot(x_for_predict,prediction_for_Q2,'g-')
# plt.show()

#Question 3
# print(np.where(x==1980)[0][0])    #get the index of a particular element
x_for_training = x[:np.where(x==1980)[0][0]+1]
x_for_validation = x[np.where(x==1980)[0][0]+1:]
y_for_training = y[:np.where(x==1980)[0][0]+1]
y_for_validation = y[np.where(x==1980)[0][0]+1:]
# print(x_for_validation)

#using linear model
w = fit(x_for_training,y_for_training,linear)
prediction_for_Q3_linear = prediction(w,x_for_training,linear)
print('validation error for linear model:')
print(objective(w,x_for_validation,y_for_validation,linear))

#using quadratic model
w = fit(x_for_training,y_for_training,quadratic)
prediction_for_Q3_quadratic = prediction(w,x_for_training,quadratic)
print('validation error for quadratic model:')
print(objective(w,x_for_validation,y_for_validation,quadratic))

#Question 4
# def polynomial(x, degree, offset, scale):
#     degrees = np.arange(degree+1)
#     return ((x-offset)/scale)**degrees

def polynomials(*x):
    # print(x)
    Phi = np.zeros((x[0].shape[0], x[1]))
    for i in range(x[1]):
        Phi[:, i:i + 1] = ((x[0]-x[2])/x[3]) ** i
    return Phi

def prediction(w, polynomials, *x):
    Phi = polynomials(*x)
    f = np.dot(Phi,w)
    return f

def objective(w,y, polynomials, *x):
    e = np.dot((y - prediction(w,polynomials,*x)).T,y - prediction(w,polynomials,*x))
    return e.sum()

def fit(y, polynomials,*x):
    w = np.linalg.solve(np.dot(polynomials(*x).T,polynomials(*x)),np.dot(polynomials(*x).T,y))
    return w

objective_error_on_training_set = []
objective_error_on_validation_set = []
for i in range(1,18):
    arg = (x_for_training,i,1956,120)
    w = fit( y_for_training, polynomials,*arg)
    objective_error_on_training_set.append(objective(w,y_for_training,polynomials,*arg))
    arg = (x_for_validation, i, 1956, 120)
    objective_error_on_validation_set.append(objective(w, y_for_validation, polynomials, *arg))

index_training_set = objective_error_on_training_set.index(min(objective_error_on_training_set))
index_validation_set = objective_error_on_validation_set.index(min(objective_error_on_validation_set))

print("On training set:")
print("degree: ",index_training_set+1)
print("objective error: ",min(objective_error_on_training_set))

print("On validation set:")
print("degree: ",index_validation_set+1)
print("objective error: ",min(objective_error_on_validation_set))