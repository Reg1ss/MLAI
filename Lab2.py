import numpy as np
import pods
import pandas as pd
import sys
import zipfile
import random

# pods.util.download_url("http://files.grouplens.org/datasets/movielens/ml-latest-small.zip")
# zip_console = zipfile.ZipFile('ml-latest-small.zip','r')
# for name in zip_console.namelist():
#     zip_console.extract(name,'./')


YourStudentID = 166  # Include here the last three digits of your UCard number
nUsersInExample = 10 # The maximum number of Users we're going to analyse at one time

ratings = pd.read_csv("./ml-latest-small/ratings.csv")
"""
ratings is a DataFrame with four columns: userId, movieId, rating and tags. We
first want to identify how many unique users there are. We can use the unique 
method in pandas
"""
indexes_unique_users = ratings['userId'].unique()   #get the unique data as a set. Return a vector
n_users = indexes_unique_users.shape[0]     #user numbers
""" 
We randomly select 'nUsers' users with their ratings. We first fix the seed
of the random generator to make sure that we always get the same 'nUsers'
"""
np.random.seed(YourStudentID)   #if seed is the same then every time we can get the same random set
indexes_users = np.random.permutation(n_users)      #disorder the array and return new array(The difference with shuffle)
my_batch_users = indexes_users[0:nUsersInExample]
"""
We will use now the list of 'my_batch_users' to create a matrix Y. 
"""
# We need to make a list of the movies that these users have watched
list_movies_each_user = [[] for _ in range(nUsersInExample)]
list_ratings_each_user = [[] for _ in range(nUsersInExample)]
# Movies
list_movies = ratings['movieId'][ratings['userId'] == my_batch_users[0]].values #return a ndarray
# print(type(list_movies))
list_movies_each_user[0] = list_movies
# Ratings
list_ratings = ratings['rating'][ratings['userId'] == my_batch_users[0]].values
list_ratings_each_user[0] = list_ratings
# Users
n_each_user = list_movies.shape[0]
# print(list_movies.shape)
list_users = my_batch_users[0]*np.ones((1, n_each_user))

for i in range(1, nUsersInExample):
    # Movies
    local_list_per_user_movies = ratings['movieId'][ratings['userId'] == my_batch_users[i]].values
    list_movies_each_user[i] = local_list_per_user_movies
    list_movies = np.append(list_movies,local_list_per_user_movies)
    # Ratings
    local_list_per_user_ratings = ratings['rating'][ratings['userId'] == my_batch_users[i]].values
    list_ratings_each_user[i] = local_list_per_user_ratings
    list_ratings = np.append(list_ratings, local_list_per_user_ratings)
    # Users
    n_each_user = local_list_per_user_movies.shape[0]
    local_rep_user =  my_batch_users[i]*np.ones((1, n_each_user))
    list_users = np.append(list_users, local_rep_user)

# Let us first see how many unique movies have been rated
indexes_unique_movies = np.unique(list_movies)
n_movies = indexes_unique_movies.shape[0]
# As it is expected no all users have rated all movies. We will build a matrix Y
# with NaN inputs and fill according to the data for each user
temp = np.empty((n_movies,nUsersInExample,))
temp[:] = np.nan
Y_with_NaNs = pd.DataFrame(temp)
for i in range(nUsersInExample):
    local_movies = list_movies_each_user[i]
    # .in1d compares two vectors and return 'mask' value which is a set of boolean values. If 'revert' parameter is False,
    # then the value of the location of the same elements is True. Otherwise, it's reversed.
    ixs = np.in1d(indexes_unique_movies, local_movies, invert=False)
    Y_with_NaNs.loc[ixs, i] = list_ratings_each_user[i]     #.loc meas locate the position of [a,b] a,b can be vectors

Y_with_NaNs.index = indexes_unique_movies.tolist()      #.tolist will convert the matrices and arrays to list
Y_with_NaNs.columns = my_batch_users.tolist()

p_list_ratings = np.concatenate(list_ratings_each_user).ravel()
p_list_ratings_original = p_list_ratings.tolist()
mean_ratings_train = np.mean(p_list_ratings)
p_list_ratings =  p_list_ratings - mean_ratings_train # remove the mean
p_list_movies = np.concatenate(list_movies_each_user).ravel().tolist()      #.concatenate: joint arrays; .ravel:  make matrices become a vector
p_list_users = list_users.tolist()
Y = pd.DataFrame({'users': p_list_users, 'movies': p_list_movies, 'ratingsorig': p_list_ratings_original,'ratings':p_list_ratings.tolist()})


q = 2 # the dimension of our map of the 'library'
learn_rate = 0.01
# *0.001 or *0.01 or *0.1 have no big difference because their squared value(U*V) are all far smaller than the rating(when calculating 'diff')
U = pd.DataFrame(np.random.normal(size=(nUsersInExample, q))*0.001, index=my_batch_users)
V = pd.DataFrame(np.random.normal(size=(n_movies, q))*0.001, index=indexes_unique_movies)

# #Display Dataframe
# print(Y_with_NaNs)
print(Y)
print(U)
print(V)

#Training with Batch gradient descent
def objective_gradient(Y, U, V):
    gU = pd.DataFrame(np.zeros((U.shape)), index=U.index)
    gV = pd.DataFrame(np.zeros((V.shape)), index=V.index)
    obj = 0.
    nrows = Y.shape[0]
    for i in range(nrows):
        row = Y.iloc[i]
        user = row['users']
        film = row['movies']
        rating = row['ratings']
        prediction = np.dot(U.loc[user], V.loc[film]) # vTu
        diff = prediction - rating # vTu - y
        obj += diff*diff
        gU.loc[user] += 2*diff*V.loc[film]
        gV.loc[film] += 2*diff*U.loc[user]
    # If we should use the mean gradient?(Although it will converge very slowly)
    # gU = gU/nrows
    # gV = gV/nrows
    return obj, gU, gV

#Training with stochastic gradient descent
def objective_gradient_with_SGD(Y, U, V):
    gU = pd.DataFrame(np.zeros(U.shape), index=U.index)
    gV = pd.DataFrame(np.zeros(V.shape), index=V.index)
    nrows = Y.shape[0]
    rowNum = random.randint(0,nrows-1)
    row = Y.iloc[rowNum]
    user = row['users']
    film = row['movies']
    rating = row['ratings']
    prediction = np.dot(U.loc[user], V.loc[film])
    diff = prediction - rating
    obj = diff * diff * nrows
    gU_gradient_change = 2 * diff * V.loc[film]
    gV_gradient_change = 2 * diff * U.loc[user]
    for i in range(nrows):
        row_counter = Y.iloc[i]
        user_counter = row_counter['users']
        film_counter = row_counter['movies']
        gU.loc[user_counter] += gU_gradient_change
        gV.loc[film_counter] += gV_gradient_change
    return obj, gU, gV

#Training
print("Training")
iterations = 1000
switcher = 1    # 0 for BGD and 1 for SGD
for i in range(iterations):
    if(switcher==0):
        obj, gU, gV = objective_gradient(Y, U, V)
    else:
        obj, gU, gV = objective_gradient_with_SGD(Y, U, V)
    print("Iteration", i, "Objective function: ", obj)
    U -= learn_rate*gU
    V -= learn_rate*gV

def predict(Y,U,V):
    nrows = Y.shape[0]
    for i in range(nrows):
        row = Y.iloc[i]
        user = row['users']
        film = row['movies']
        rating = row['ratings']
        prediction = np.dot(U.loc[user],V.loc[film])
        absoluteError = abs(prediction - rating)
        Y.loc[i,'prediction'] = prediction      #add a new column
        Y.loc[i,'absError'] = absoluteError     #add a new column
    return Y

y_with_prediction = predict(Y,U,V)
print(y_with_prediction)



























# #numpy array
# a = np.array([[1,2,3,4],
#              [4,5,6,7],
#              [7,8,9,10]])
# b = np.array([])
# print(a.shape[1]) #[0]rows [1]columns
# x_select = a[1, 0:2] #parameters[row,column:column]
# print(x_select)

# #在指定的范围内返回均匀间隔的数字
# np.linspace(start=-2, stop=2, num=50, endpoint=True, retstep=False, dtype=None)

