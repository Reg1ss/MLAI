import pods
import pylab as plt
import numpy as np
import pandas as pd

# pods.notebook.display_google_book(id='spcAAAAAMAAJ', page=72)
# data = pods.datasets.olympic_marathon_men()
#
# x = data['X']
# y = data['Y']
# # print(x)
# print(y)
#
# # plt.plot(x, y, 'rx')
# # plt.xlabel('year')
# # plt.ylabel('pace in min/km')
# # plt.show()
# m = -0.4
# c = 80
#
# c = (y - m*x).mean()
# print(c)
# m = ((y - c)*x).sum()/(x**2).sum()
# print(m)
#
# x_test = np.linspace(1890, 2020, 130)[:,None]   #,None is same as numpy.newaxis
# f_test = m*x_test + c

# print(x_test)
# print(f_test)

# plt.plot(x_test, f_test, 'b-')
# plt.plot(x, y, 'rx')
# plt.show()

# #Interations with coordinate descent
# maxIteration = pow(10,6);
# diff_last_round = 0;
# converge = False
# for i in np.arange(maxIteration):
#     if(converge):
#         break
#     m = ((y - c)*x).sum()/(x*x).sum()
#     c = (y-m*x).sum()/y.shape[0]
#     diff = 0;
#     for j in range(y.shape[0]):
#         diff_per_element = (y[j] - m*x[j] - c)*(y[j] - m*x[j] - c)
#         diff += diff_per_element
#         # print(y[j])
#         # print(x[j])
#         # print("c: ",c)
#         # print("m: ",m)
#     if(i==0):
#         diff_last_round = diff
#         print("The Iteration ", i, " Error is: ", diff)
#         continue;
#     if((diff_last_round - diff)<pow(10,-4)):
#         converge = True
#     diff_last_round = diff
#     if(i%10==0):
#         print("The Iteration ",i," Error is: ",diff)
#
#
# print("Converged. Final m is: ",m," Final c is: ",c)
# f_test = m*x_test + c
# plt.plot(x_test, f_test, 'b-')
# plt.plot(x, y, 'rx')
# plt.show()

# #Directly solve
# # define the vector w
# w = np.zeros(shape=(2, 1))
# w[0] = c
# w[1] = m
#
# X = np.hstack((np.ones_like(x), x))
# print(X)
#
# f = np.dot(X, w)
# resid = (y-f)
# E = np.dot(resid.T, resid) # matrix multiplication on a single vector is equivalent to a dot product.
# print("Error function is:", E)
#
# w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
# print(w)
#
# m = w[1]; c=w[0]
# f_test = m*x_test + c
# print(m)
# print(c)
# plt.plot(x_test, f_test, 'b-')
# plt.plot(x, y, 'rx')
# plt.show()

data = pods.datasets.movie_body_count()
movies = data['Y']

print(', '.join(movies.columns))

select_features = ['Year', 'Body_Count', 'Length_Minutes']
X = movies.loc[:, select_features]
X['Eins'] = 1 # add a column for the offset
y = movies[['IMDB_Rating']]

w = pd.DataFrame(data=np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y)),  # solve linear regression here
                 index = X.columns,  # columns of X become rows of w
                 columns=['regression_coefficient']) # the column of X is the value of regression coefficient
print(w)
(y - np.dot(X, w)).hist()
print(w)
plt.show()
import scipy as sp
Q, R = np.linalg.qr(X)
w = sp.linalg.solve_triangular(R, np.dot(Q.T, y))
w = pd.DataFrame(w, index=X.columns)
print(w)