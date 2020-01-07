import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,roc_curve,auc,confusion_matrix
import torch.nn as nn

#question 1
scale = 0.3
#add nosie
def nosiy(img):
    img = img + scale * torch.randn(3,32,32)
    img = np.clip(img,-1,1)
    return img

#trasnform: preprocess the dataset
torch.manual_seed(1722846)
transform = transforms.Compose(     #compose several processes to one
    [transforms.ToTensor(),     #transform a [0,255] PIL(python image library).image to tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),    #normalize the value to [-1,1], first():mean, second():std  x = (x - mean(x))/stddev(x)
     transforms.Lambda(lambda img: nosiy(img))])

#CIFAR10: 60,000 imgs, 50,000 for training, 10,000 for test
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#get datasets with noise
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
# print(trainset[0][0]) #0 for img, 1 for label

#get original datasets
transform_original = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset_original = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_original)
testset_original = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_original)

#get catdog subsets with noise
#get CatDog_traindata
index=0
CatDog_indices_trainset = []
for traindata in trainset:
    if(traindata[1]==3 or traindata[1]==5):
        CatDog_indices_trainset.append(index)
    index += 1
CatDog_traindata = torch.utils.data.Subset(trainset, CatDog_indices_trainset)
#get CatDog_testdata
index=0
CatDog_indices_testset = []
for testdata in testset:
    if(testdata[1]==3 or testdata[1]==5):
        CatDog_indices_testset.append(index)
    index += 1
CatDog_testdata = torch.utils.data.Subset(testset, CatDog_indices_testset)

# get catdog subset without noise
# get CatDog_traindata_original
# index=0
# CatDog_indices_trainset_original = []
# for traindata_original in trainset_original:
#     if(traindata_original[1]==3 or traindata_original[1]==5):
#         CatDog_indices_trainset_original.append(index)
#     index += 1
# CatDog_traindata_original = torch.utils.data.Subset(trainset_original, CatDog_indices_trainset_original)
# #get CatDog_testdata_original
# index=0
# CatDog_indices_testset_original = []
# for testdata_original in testset_original:
#     if(testdata_original[1]==3 or testdata_original[1]==5):
#         CatDog_indices_testset_original.append(index)
#     index += 1
# CatDog_testdata_original = torch.utils.data.Subset(testset_original, CatDog_indices_testset_original)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# for i in range(10):
#     imshow(torchvision.utils.make_grid([CatDog_traindata_original[i][0],CatDog_traindata[i][0]]))   #torchvision.utils.make_grid: connet pics as grid


# imshow(trainset[0][0])
# imshow(trainset_original[0][0])

#Question 2
#get X and Y from pytorch dataset
def get_X_and_y(dataset):
    X = np.zeros(shape=(len(dataset),3,32,32))
    Y = np.zeros(shape=(len(dataset), ))
    for i in range(len(dataset)):
        X[i] = dataset[i][0].numpy()
        Y[i] = dataset[i][1]    #dataset[i][1] is an int
    return  X,Y

#label function for a bar graph
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.3, 1.01*height, '%0.2f' % float(height))

# #a)
# #get X and Y from CatDog dataset without noise
# X_trainset_original,y_trainset_original = get_X_and_y(CatDog_traindata_original)
# # print(Y_trainset_original.shape)
# no_components = 150     #max 3072
# #Expand 4D X_trainset matrix into 2D
# X_trainset_original_reshape = X_trainset_original.reshape((X_trainset_original.shape[0],3*32*32))
#
# #show PCA components ratio
# pca_CIFAR10_original = PCA(n_components=no_components, svd_solver='full', whiten=True).fit(X_trainset_original_reshape)
# components_ratio = (np.cumsum(pca_CIFAR10_original.explained_variance_ratio_))
# least_compunent = 0
# for i in range(components_ratio.shape[0]):
#     if(components_ratio[i]>0.8):
#         print("The first ",i," components has contributed over 80% variance")
#         least_compunent = i
#         break
# k_values = np.linspace(least_compunent,150,7)
# #get new features after PCA for trainset
# X_train_CIFAR10_original_intmde = []
# for i in range(len(k_values)):
#     k_values[i] = int(k_values[i])
#     X_train_CIFAR10_original_intmde.append(PCA(n_components=int(k_values[i]), svd_solver='randomized', whiten=True).fit_transform(X_trainset_original_reshape))
# #A: Because the first 31 components has contributed over 80% variance. So we choose 7 number from 31 to 150 in the same distance.
# #   Finally we choose 7 K values: 31, 50, 70, 90, 110 ,130, 150
#
# #2b)
# #A: Because our features in the datasets are continuous and it is very hard to separate them into discrete classes. So we choose
# #   Gaussian NB.
# #2c)
# #get X and y for testset
# X_testset_original, y_testset_original = get_X_and_y(CatDog_testdata_original)
# X_testset_original_reshape = X_testset_original.reshape((X_testset_original.shape[0],3*32*32))
# #Gaussian NB
# model = GaussianNB()    #automatically set the prior
# model.fit(X_trainset_original_reshape, y_trainset_original);
# Y_testset_original_predict = []
# y_testset_original_predict_raw = model.predict(X_testset_original_reshape)
# Y_testset_original_predict.append(y_testset_original_predict_raw)
# # print(y_testset_original_predict)
# correct_ratio = []
# correct_ratio.append(accuracy_score(y_testset_original,y_testset_original_predict_raw))
# print("Correct ratio for original features is: ", correct_ratio[0])
#
# #get new features after PCA for test set
# X_test_CIFAR10_original_intmde = []
# for i in range(len(k_values)):
#     k_values[i] = int(k_values[i])
#     X_test_CIFAR10_original_intmde.append(PCA(n_components=int(k_values[i]), svd_solver='randomized', whiten=True).fit_transform(X_testset_original_reshape)) #new features after PCA
#
# for i in range(7):
#     model = GaussianNB()    #automatically set the prior
#     model.fit(X_train_CIFAR10_original_intmde[i], y_trainset_original);
#     y_testset_original_predict_k = model.predict(X_test_CIFAR10_original_intmde[i])
#     Y_testset_original_predict.append(y_testset_original_predict_k)
#     correct_ratio.append(accuracy_score(y_testset_original,y_testset_original_predict_k))
#     print("Correct ratio for model with K value = ", k_values[i] ," is: ", correct_ratio[i+1])
#
# bar_x = ['Original']
# for value in k_values:
#     bar_x.append(value)
# accuracy_bar = plt.bar(range(len(bar_x)),correct_ratio, tick_label = bar_x)
# plt.xlabel('8 NBs')
# plt.ylabel('Correct ratio')
# autolabel(accuracy_bar)
# plt.show()
#
# #roc
# line_colors = ['b','g','r','c','m','y','gold','lightpink']
# color_index = 0
# aucs = []
# for y_predict in Y_testset_original_predict:
#     fpr,tpr,thresholds = roc_curve(y_testset_original,y_predict,pos_label=5)
#     roc_auc = auc(fpr,tpr)
#     aucs.append(roc_auc)
#     if(color_index==1):
#         plt.plot(fpr, tpr, color=line_colors[color_index],
#                  lw=2, label='Original features %s (area = %0.2f)' % (color_index+1,roc_auc))
#     else:
#         plt.plot(fpr, tpr, color=line_colors[color_index],
#                  lw=2, label='K = %s %s (area = %0.2f)' % (k_values[color_index-1],color_index + 1, roc_auc))
#     color_index+=1
#     plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic')
#     plt.legend(loc="lower right")
# plt.show()
#
# #2e)
# aucs_bar = plt.bar(range(len(bar_x)),aucs, tick_label = bar_x)
# plt.xlabel('8 NBs')
# plt.ylabel('AUC')
# autolabel(aucs_bar)
# plt.show()


#Question 3
# #3a)
# #get X and Y from CatDog dataset with noise
# X_trainset,y_trainset = get_X_and_y(CatDog_traindata)
# # print(Y_trainset_original.shape)
# no_components = 500     #max 3072
# #Expand 4D X_trainset matrix into 2D
# X_trainset_reshape = X_trainset.reshape((X_trainset.shape[0],3*32*32))
# #show PCA components ratio
# pca_CIFAR10 = PCA(n_components=no_components, svd_solver='randomized', whiten=True).fit(X_trainset_reshape)
# components_ratio = (np.cumsum(pca_CIFAR10.explained_variance_ratio_))
# # print(components_ratio)
# # least_compunent = 0
# # for i in range(components_ratio.shape[0]):
# #     if(components_ratio[i]>0.7):
# #         print("The first ",i," components has contributed over 70% variance")
# #         least_compunent = i
# #         break
# k_values = np.linspace(least_compunent,150,7)
# #get new features after PCA for trainset
# X_train_CIFAR10_intmde = []
# for i in range(len(k_values)):
#     k_values[i] = int(k_values[i])
#     X_train_CIFAR10_intmde.append(PCA(n_components=int(k_values[i]), svd_solver='randomized', whiten=True).fit_transform(X_trainset_reshape))
# X_testset, y_testset = get_X_and_y(CatDog_testdata)
# X_testset_reshape = X_testset.reshape((X_testset.shape[0],3*32*32))
# #Gaussian NB
# model = GaussianNB()    #automatically set the prior
# model.fit(X_trainset_reshape, y_trainset);
# #get X and y for testset
# Y_testset_predict = []
# y_testset_predict_raw = model.predict(X_testset_reshape)
# Y_testset_predict.append(y_testset_predict_raw)
# # print(y_testset_original_predict)
# correct_ratio = []
# correct_ratio.append(accuracy_score(y_testset,y_testset_predict_raw))
# print("Correct ratio for original features is: ", correct_ratio[0])
# #get new features after PCA for test set
# X_test_CIFAR10_intmde = []
# for i in range(len(k_values)):
#     k_values[i] = int(k_values[i])
#     X_test_CIFAR10_intmde.append(PCA(n_components=int(k_values[i]), svd_solver='randomized', whiten=True).fit_transform(X_testset_reshape)) #new features after PCA
# for i in range(7):
#     model = GaussianNB()    #automatically set the prior
#     model.fit(X_train_CIFAR10_intmde[i], y_trainset);
#     y_testset_predict_k = model.predict(X_test_CIFAR10_intmde[i])
#     Y_testset_predict.append(y_testset_predict_k)
#     correct_ratio.append(accuracy_score(y_testset,y_testset_predict_k))
#     print("Correct ratio for model with K value = ", k_values[i] ," is: ", correct_ratio[i+1])
# bar_x = ['Original']
# for value in k_values:
#     bar_x.append(value)
# accuracy_bar = plt.bar(range(len(bar_x)),correct_ratio, tick_label = bar_x)
# plt.xlabel('8 NBs')
# plt.ylabel('Correct ratio')
# autolabel(accuracy_bar)
# plt.show()


#3b)
#get X and Y from whole dataset without noise
X_trainset_original,y_trainset_original = get_X_and_y(trainset_original)
X_testset_original, y_testset_original = get_X_and_y(CatDog_testdata)
X_trainset_original_reshape = X_trainset_original.reshape((X_trainset_original.shape[0],3*32*32))
pca_CIFAR10 = PCA(n_components=150, svd_solver='randomized', whiten=True).fit(X_trainset_original_reshape)
components_ratio = (np.cumsum(pca_CIFAR10.explained_variance_ratio_))
print(components_ratio)
least_compunent = 0
for i in range(components_ratio.shape[0]):
    if(components_ratio[i]>0.7):
        print("The first ",i," components has contributed over 70% variance")
        least_compunent = i
        break
#A: we choose 3 values that contributes the variance above 50%
k_values = np.linspace(1,least_compunent,3)
#get new features after PCA for trainset
X_trainset_original_CIFAR10_intmde = []
for i in range(len(k_values)):
    k_values[i] = int(k_values[i])
    X_trainset_original_CIFAR10_intmde.append(PCA(n_components=int(k_values[i]), svd_solver='randomized', whiten=True).fit_transform(X_trainset_original_reshape))

# model = GaussianNB()    #automatically set the prior
# model.fit(X_trainset_reshape, y_trainset);
# #get X and y for testset
# Y_testset_predict = []
# y_testset_predict_raw = model.predict(X_testset_reshape)
# Y_testset_predict.append(y_testset_predict_raw)
# # print(y_testset_original_predict)
# correct_ratio = []
# correct_ratio.append(accuracy_score(y_testset,y_testset_predict_raw))
# print("Correct ratio for original features is: ", correct_ratio[0])
















# #Question 4
# import torch.nn as nn
# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             # 1 input image channel, 16 output channel, 3x3 square convolution
#             nn.Conv2d(3, 16, 3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, 3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 7)
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, 7),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid()  #to range [0, 1]
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         x = 2 * x - 1
#         return x
#
# myAE=Autoencoder()
#
# #Hyperparameters for training
# batch_size=64
# learning_rate=0.001
# max_epochs=6
#
# #Set the random seed for reproducibility
# #Choose mean square error loss
# criterion = nn.MSELoss()
# #Choose the Adam optimiser
# optimizer = torch.optim.Adam(myAE.parameters(), lr=learning_rate, weight_decay=1e-5)
# #Specify how the data will be loaded in batches (with random shffling)
# train_loader_original = torch.utils.data.DataLoader(trainset_original, batch_size=batch_size, shuffle=True)
# #Storage
# outputs = []
# #Start training
# for epoch in range(max_epochs):
#     loss_total = 0
#     loss_count = 0
#     for data in train_loader_original:
#         img, label = data
#         optimizer.zero_grad()
#         img_noise = np.clip(img + 0.2 * torch.randn(3,32,32),-1,1)
#         recon = myAE(img_noise)
#         loss = criterion(recon, img)
#         loss.backward()
#         optimizer.step()
#         loss_total += loss
#         loss_count += 1
#     if (epoch % 1) == 0:
#         print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss_total/loss_count)))
#     outputs.append((epoch, img, recon))
# # print(loss.grad_fn)
# # print(loss.grad_fn.next_functions[0][0])
#
# numImgs = 4
# for k in range(0, max_epochs,1):
#     plt.figure(figsize=(numImgs, 2))
#     imgs = outputs[k][1].detach().numpy()
#     recon = outputs[k][2].detach().numpy()
#     for i, item in enumerate(imgs):
#         if i >= numImgs: break
#         plt.subplot(2, numImgs, i + 1)
#         item = item / 2 +0.5
# #         print(item)
#         plt.imshow(np.transpose(item, (1, 2, 0)))
#
#     for i, item in enumerate(recon):
#         if i >= numImgs: break
#         plt.subplot(2, numImgs,numImgs+ i + 1)
#         item = item / 2 +0.5
#         plt.imshow(np.transpose(item, (1, 2, 0)))
#
# #4c)
# test_loader_original = torch.utils.data.DataLoader(testset_original, batch_size=1, shuffle=True)
# testset_losses = []
# for data in test_loader_original:
#     img, label = data
#     img_noise = np.clip(img + 0.2 * torch.randn(3, 32, 32), -1, 1)
#     recon = myAE(img_noise)
#     loss = criterion(recon, img)
#     testset_losses.append(float(loss))
#
# #4d)
# batch_size=64
# learning_rate=0.001
# max_epochs=6
#
# loss_batch_size = []
# for batch_size in [32,64,128]:
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(myAE.parameters(), lr=learning_rate, weight_decay=1e-5)
#     train_loader_original = torch.utils.data.DataLoader(trainset_original, batch_size=batch_size, shuffle=True)
#     #Start training
#     print('batch size = ',batch_size)
#     for epoch in range(max_epochs):
#         loss_total = 0
#         loss_count = 0
#         for data in train_loader_original:
#             img, label = data
#             optimizer.zero_grad()
#             img_noise = np.clip(img + 0.2 * torch.randn(3,32,32),-1,1)
#             recon = myAE(img_noise)
#             loss = criterion(recon, img)
#             loss.backward()
#             optimizer.step()
#             loss_total += loss
#             loss_count += 1
#     loss_batch_size.append(loss_total/loss_count)














# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,   #data sampling apparatus
#                                           shuffle=True, num_workers=2)  #num_workers:how many progresses to use
#
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)


# print('Top 15 eigenvalues: \n', pca_CIFAR10_original.explained_variance_[:15])
#
# cumulative_variance = np.cumsum(pca_CIFAR10_original.explained_variance_[:15])
# print('\nCumulative variance values:\n', cumulative_variance)
#
# # Plot cumulative variance
# plt.plot(range(1, cumulative_variance.shape[0] + 1), cumulative_variance)
# plt.title('Cumulative variance of principal components')
# plt.xlabel('Principle component number')
# plt.show()
# pca_object = PCA(n_components=100, svd_solver='randomized', whiten=True) #components number
# x_train_CIFAR10_original_intmde = pca_object.fit_transform(X_trainset_original_reshape)  #new features after PCA
# x_train_CIFAR10_original_approx = pca_object.inverse_transform(x_train_CIFAR10_original_intmde)
# reconstruction_error = np.sum(np.square(X_trainset_original_reshape - x_train_CIFAR10_original_approx)) / (
#         X_trainset_original_reshape.shape[0] * X_trainset_original_reshape.shape[1])
# print("\nMean squared error:%f" % (reconstruction_error))



