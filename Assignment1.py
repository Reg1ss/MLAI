import pods
import pandas as pd
import numpy as np
import pylab as plt

# pods.util.download_url('https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip')
import zipfile
# zip = zipfile.ZipFile('./AirQualityUCI.zip', 'r')
# for name in zip.namelist():
#     zip.extract(name, '.')

air_quality = pd.read_excel('./air_quality_dataSet/AirQualityUCI.xlsx', usecols=range(2, 15))
# print(air_quality.sample(2))
print(air_quality.describe())

#question 3- split data
#random seed
MyStudentID = 22846
np.random.seed(MyStudentID)
#number of data & number of attributes
n_data = air_quality.shape[0]
n_columns = air_quality.shape[1]
#disorder the data
indexes_data = np.random.permutation(n_data)
#get the number of training_set & test_set
n_training_set = int(n_data*0.7)
n_test_set = n_data - n_training_set
temp = np.empty((n_training_set,n_columns))
# index_training_set = []
# index_test_set = []
# data attributes
column_data = air_quality.columns
#get training set
for i in range(n_training_set):
    # index_training_set.append(indexes_data[i])
    temp[i][:] = air_quality.iloc[indexes_data[i]]
    # print(temp[i])
training_set = pd.DataFrame(temp)
# training_set.index = index_training_set
training_set.columns = column_data
#get test set
temp = np.empty((n_test_set,n_columns))
for i in range(n_training_set,n_data):
    # index_test_set.append(indexes_data[i])
    temp[i-n_training_set][:] = air_quality.iloc[indexes_data[i]]
test_set = pd.DataFrame(temp)
# test_set.index = index_test_set
test_set.columns = column_data
# print(training_set.loc[8387])
# print(test_set.iloc[n_test_set-1])

#Question4- Missing value
drop_criterion = int(0.2*n_training_set)
#preprocess training set
#drop columns with more than 20% missing values
for j in column_data:
    counter_missing_per_data = 0;
    if(j=='CO(GT)'):
        continue
    for i in range(n_training_set):
        if(training_set[j][i]==-200):
            counter_missing_per_data += 1
        if(counter_missing_per_data>=drop_criterion):
            training_set = training_set.drop(j,axis=1)
            drop_column = j
            print("Drop column: ",drop_column)
            break

#update column data
column_data = training_set.columns

#drop rows with missing labels
for i in range(n_training_set):
    if (training_set["CO(GT)"][i] == -200):
        training_set = training_set.drop([i])

#update training set number and index
n_training_set = training_set.shape[0]
training_set.index = range(n_training_set)

#get fixed value for every column's missing value
fixed_value = {}
for j in column_data:
    counter = 0
    sum = 0
    for i in range(n_training_set):
        if(training_set[j][i]!=-200):
            sum += training_set[j][i]
            counter += 1
    fixed_value[j]=sum/counter

#fix Nan value
for i in range(n_training_set):
    for j in column_data:
        if(training_set[j][i]==-200):
            training_set[j][i]=fixed_value[j]


#question5- Normalisation of training set except the objective variable
mean_value_per_column = {}
std_value_per_column = {}
for j in column_data:
    if(j=="CO(GT)"):
        continue
    # get mean value and std for every column in training set
    mean_value_per_column[j] = training_set[j].mean()
    std_value_per_column[j] = training_set[j].std()
    # normalize training set
    training_set[j] = (training_set[j] - mean_value_per_column[j]) / std_value_per_column[j]
# print(training_set.std())

#question5- get dataframes
#get objective variable
y = training_set.loc[:,column_data[0]]
#get feature  variables
#get design metrix
first_column = [1 for i in range(n_training_set)]
training_set.insert(1,"W0",first_column)
#update column data
column_data = training_set.columns
X = training_set.loc[:,column_data[1]:column_data[12]]


#question6- training with closed form
#get a set of values for alpha
alpha = np.logspace(-3,2,20)
#split training data in to training set and validation set
n_training = int(n_training_set*0.7)
n_validation = n_training_set - n_training
#get random indexes
indexes_data_for_X_and_y = np.random.permutation(n_training_set)
#split X
#get X_training
temp = np.empty((n_training,n_columns-1))
for i in range(n_training):
    temp[i][:] = X.iloc[indexes_data_for_X_and_y[i]]
X_training = pd.DataFrame(temp)
#get X_validation
temp = np.empty((n_validation,n_columns-1))
for i in range(n_training,n_training_set):
    temp[i-n_training][:] = X.iloc[indexes_data_for_X_and_y[i]]
X_validation = pd.DataFrame(temp)
#split y
#get y_training
temp = np.empty((n_training,1))
for i in range(n_training):
    temp[i][:] = y.iloc[indexes_data_for_X_and_y[i]]
y_training = pd.DataFrame(temp)
#get y_validation
temp = np.empty((n_validation,1))
for i in range(n_training,n_training_set):
    temp[i-n_training][:] = y.iloc[indexes_data_for_X_and_y[i]]
y_validation = pd.DataFrame(temp)

#training
temp_w = np.empty((len(alpha),n_columns-1))
for i in range(len(alpha)):
    for j in range(n_columns-1):
        temp_w[i][j] = (np.linalg.solve(2*np.dot(X_training.T, X_training) + n_training*alpha[i]*np.identity(n_columns-1), 2*np.dot(X_training.T, y_training)))[j]
        # print("temp_w ",i," and ",j,"is:",temp_w[i][j])
# w is a metrix contains parameter vectors with different alpha values
w = pd.DataFrame(temp_w)
mse_for_alpha = []
#validate the best alpha value on validation set
for i in range(len(alpha)):
    mse_for_alpha.append((np.dot(y_validation[0] - np.dot(w.iloc[i],X_validation.T), y_validation[0] - np.dot(w.iloc[i],X_validation.T).T))/n_validation)
alpha_min_mse_index = mse_for_alpha.index(min(mse_for_alpha))
alpha_best = alpha[alpha_min_mse_index]
print("MSE on validation set:")
print(mse_for_alpha[alpha_min_mse_index])
# print(alpha_min_mse_index)
print("The best alpha is: ",alpha_best)

#question7 -vaditation with closed form
#preprocess test set
#drop the same column in test set
test_set = test_set.drop(drop_column,axis=1)

#drop rows with missing labels
for i in range(n_test_set):
    if (test_set["CO(GT)"][i] == -200):
        test_set = test_set.drop([i])

#update test set number and index
n_test_set = test_set.shape[0]
test_set.index = range(n_test_set)

#fix Nan value
column_data = test_set.columns
for i in range(n_test_set):
    for j in column_data:
        if(test_set[j][i]==-200):
            test_set[j][i]=fixed_value[j]

# #Check there is no missing value
# for i in range(n_training_set):
#     for j in column_data:
#         if(training_set[j][i]==-200):
#             print("Still have missing values!")
# for i in range(n_test_set):
#     for j in column_data:
#         if(test_set[j][i]==-200):
#             print("Still have missing values!")

#get design metrix for test set
first_column = [1 for i in range(n_test_set)]   #create a list in line
test_set.insert(1,"W0",first_column)
column_data = test_set.columns

#Normalisation of test set
for j in column_data:
    # get mean value and std for every column in training set
    if (j == "CO(GT)" or j == "W0"):
        continue
    # normalize training set
    test_set[j] = (test_set[j] - mean_value_per_column[j]) / std_value_per_column[j]
# print(test_set["PT08.S1(CO)"])
# print(test_set.std())

#training with all training data
temp_w = np.empty((len(alpha),n_columns-1))
for i in range(len(alpha)):
    for j in range(n_columns-1):
        temp_w[i][j] = (np.linalg.solve(2*np.dot(X.T, X) + n_training_set*alpha[i]*np.identity(n_columns-1), 2*np.dot(X.T, y)))[j]
w = pd.DataFrame(temp_w)
# mse_for_alpha = []
# for i in range(len(alpha)):
#     mse_for_alpha.append((np.dot(y_validation[0] - np.dot(w.iloc[i],X_validation.T), y_validation[0] - np.dot(w.iloc[i],X_validation.T).T))/n_validation)
# alpha_min_mse_index = mse_for_alpha.index(min(mse_for_alpha))
# alpha_best = alpha[alpha_min_mse_index]
# print(alpha_best)

#mse on test set
#get objective variable of test set
y_test_series = test_set.loc[:,column_data[0]]
y_test = pd.DataFrame(y_test_series)
#get feature variables of test set
X_test = test_set.loc[:,column_data[1]:column_data[12]]
#get mse
mse_test_set = ((np.dot(y_test[column_data[0]] - np.dot(w.iloc[alpha_min_mse_index],X_test.T), y_test[column_data[0]] - np.dot(w.iloc[alpha_min_mse_index],X_test.T).T))/n_test_set)
print("w of the closed form solution is: ")
print(w.iloc[alpha_min_mse_index])
print("MSE on test set: ",mse_test_set)
#get absolute error
(y_test[column_data[0]] - np.dot(X_test, w.iloc[alpha_min_mse_index])).hist(bins=1000)
# plt.show()

# question 8-minibatch gradient descent

# def compute_obj(y,X,w,alpha):
#     obj = 0
#     for i in range(X.shape[0]):
#         diff = (y.iloc[i] - np.dot(w[0],X.iloc[i].T))*(y.iloc[i] - np.dot(w[0],X.iloc[i].T)) + alpha/2*np.dot(w[0],w[0].T)
#         obj += diff
#     return obj

def compute_mse(y,X,w):
    mse = ((np.dot(y[column_data[0]] - np.dot(w[0],X.T), y[column_data[0]] - np.dot(w[0],X.T).T))/y.shape[0])
    return mse

def objective_gradient(y_value,x,w,alpha,n):
    diff = y_value - np.dot(x.T,w[0])
    gw = alpha * w[0] - (2*diff*x)/n
    return gw

def BGD(y, X, w, alpha, learning_rate, n_batch, max_iter = 500):
    # converge = False
    n_total_data = y.shape[0]
    indicator = 0;
    iteration_counter = 0;
    batch_gw = 0
    while(True):
        # if converge:
        #     break
        if(iteration_counter==max_iter):
            break
        for i in range(n_total_data):
            if (iteration_counter == max_iter):
                break
            batch_gw += objective_gradient(y[0][i],X.iloc[i],w,alpha,n_total_data)/n_batch
            indicator += 1
            if(indicator%n_batch==0):
                w -= learning_rate * batch_gw
                batch_gw = 0
                iteration_counter += 1
                print("Update: ",iteration_counter)
    return w

def BGD_1(y, X, w, alpha, learning_rate, n_batch, max_iter = 500):
    # converge = False
    n_total_data = y.shape[0]
    iteration_counter = 0;
    batch_gw = 0
    last_batch_start = int((y.shape[0]//n_batch) * n_batch)
    for j in range(max_iter):
        # if converge:
        #     break
        if(iteration_counter==max_iter):
            break
        for i in range(n_total_data):
            if (i == last_batch_start-1):
                batch_gw = 0
                for k in range(n_total_data - last_batch_start):
                    batch_gw += objective_gradient(y[0][i],X.iloc[i],w,alpha,n_total_data)/(n_total_data-last_batch_start)
                w -= learning_rate * batch_gw
                break
            batch_gw += objective_gradient(y[0][i],X.iloc[i],w,alpha,n_total_data)/n_batch
            if(i%n_batch==0):
                w -= learning_rate * batch_gw
                batch_gw = 0
                print("Update: ")

        iteration_counter += 1
        print("iteration",iteration_counter)
    return w

def objective_gradient_batch(y_batch,X_batch,w,alpha,n):
    gw = alpha * w-2/n*(np.dot(X_batch.T,y_batch)-np.dot(np.dot(X_batch.T,X_batch),w))
    return gw

def objective_gradient_batch_with_python_matrix(y_batch,X_batch,w,alpha,n):
    gw = alpha * w-2/n*(X_batch.T.values @ y_batch.values - X_batch.T.values @ X_batch.values @ w.values)
    gw = pd.DataFrame(gw)
    return gw

def BGD_batch(y, X, w, alpha, learning_rate, n_batch, max_iter = 500):
    # converge = False
    n_total_data = y.shape[0]
    total_batch = int(y.shape[0] // n_batch)
    iteration_counter = 0;
    batch_gw = 0
    while(True):
        # if converge:
        #     break
        if(iteration_counter>=max_iter):
            break
        for i in range(total_batch):
            if (iteration_counter == max_iter):
                break
            y_batch = y[i*n_batch : (i+1)*n_batch-1]
            X_batch = X[i*n_batch:(i+1)*n_batch-1]
            batch_gw = objective_gradient_batch(y_batch,X_batch,w,alpha,n_total_data) #/n_batch
            w -= learning_rate * batch_gw
            iteration_counter += 1
            # last few data as a batch
            if(i==total_batch-1):
                y_batch = y[(total_batch)*n_batch : n_total_data]
                X_batch = X[(total_batch)*n_batch : n_total_data]
                batch_gw = objective_gradient_batch(y_batch, X_batch, w, alpha, y_batch.shape[0]) #/X_batch.shape[0]
                w -= learning_rate * batch_gw
                iteration_counter += 1
                #if the last batch is the final iteration
                if(iteration_counter>=max_iter):
                    break
    return w

w_list = []
mse_list = []
hyper_parameter_list = []
n_alpha = 5
n_learning_rate = 5
n_batch = 5
alpha = np.logspace(-3,1,n_alpha)
learning_rate = np.logspace(-2,-1,n_learning_rate)
batch = np.linspace(128,512,n_batch)
#use training set to train
for i in range(n_alpha):
    for j in range(n_learning_rate):
        for k in range(n_batch):
            print("alpha:", i + 1, " learning_rate:", j + 1, " batch iteration: ", k + 1)
            w = pd.DataFrame(np.random.normal(size=(n_columns - 1, 1)) * 0.00, index=np.arange(n_columns - 1))
            w = BGD_batch(y_training,X_training,w,alpha[i],learning_rate[j],int(batch[k]),max_iter=500)
            w_list.append(w)
            mse_list.append(compute_mse(y_test,X_test,w))
            hyper_parameter_list.append([alpha[i],learning_rate[j],batch[k]])
            # print(hyper_parameter_list)
hyper_parameter_index = mse_list.index(min(mse_list))
print("The result with the training data:")
print("The lowest mse is: ",min(mse_list))
print("The best α, η and B are: ",hyper_parameter_list[hyper_parameter_index])

#use all training data to train
best_alpha = hyper_parameter_list[hyper_parameter_index][0]
best_learning_rate = hyper_parameter_list[hyper_parameter_index][1]
best_batch = hyper_parameter_list[hyper_parameter_index][2]
w = pd.DataFrame(np.random.normal(size=(n_columns - 1, 1)) * 0.00, index=np.arange(n_columns - 1))
y = y.to_frame()#or y doesn't have column index
w = BGD_batch(y,X,w,best_alpha,best_learning_rate,int(best_batch),max_iter=500)
print(" ")
print("The result with all the training data:")
print("w of the closed form solution is: ")
print(w)
print("The mse on the test set is: ",compute_mse(y_test,X_test,w))




