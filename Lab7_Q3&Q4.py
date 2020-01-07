import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torchvision import datasets

# Note: You need to reshape the data from images to vectors in order to use PCA
# Your code with comments and output

mnist_data = datasets.MNIST('data', train=True, download=True)
x_train_mnist = mnist_data.data.numpy()

# Determine the dimensions of images, number of features in the data and the number of different digits. 150 = number of prinicpal components.
h_mnist, w_mnist = x_train_mnist.shape[1:]
print((x_train_mnist.shape))
# print(h_mnist)
# print(w_mnist)
x_train_mnist_reshape = x_train_mnist.reshape((x_train_mnist.shape[0], h_mnist * w_mnist))  #Expand the high dimension metrix into low dimension metrix
y_train_mnist = mnist_data.targets.numpy()
# print(y_train_mnist.shape)
no_features = x_train_mnist_reshape.shape[1]
no_classes = np.unique(y_train_mnist).shape[0]
# print(no_features)
# print(no_classes)
# print(x_train_mnist_reshape.shape)
no_components = 150     #how many principle components we want to save

# fit pca to the training data

pca_mnist = PCA(n_components=no_components, svd_solver='randomized',    #svd_solver: ‘auto’, ‘full’, ‘arpack’, ‘randomized’ last 2 for big data
                whiten=True).fit(x_train_mnist_reshape)

# top 30 eigenvalues
print('Top 30 eigenvalues: \n', pca_mnist.explained_variance_[:30])

# Cumulative variance
cumulative_variance = np.cumsum(pca_mnist.explained_variance_[:30])
print('\nCumulative variance values:\n', cumulative_variance)

# Plot cumulative variance
plt.plot(range(1, cumulative_variance.shape[0] + 1), cumulative_variance)
plt.title('Cumulative variance of principal components')
plt.xlabel('Principle component number')
plt.show()

eigenfaces_mnist = pca_mnist.components_.reshape((no_components, h_mnist, w_mnist))

eigenface_titles_mnist = ["eigenface %d" % i for i in range(eigenfaces_mnist.shape[0])]


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()

# Plot top 10 eigenfaces
plot_gallery(eigenfaces_mnist, eigenface_titles_mnist, h_mnist, w_mnist, n_row=2, n_col=5)

# Reconstruct data using top 10 components

pca_object = PCA(n_components=10, svd_solver='randomized', whiten=True) #components number
x_train_mnist_intmde = pca_object.fit_transform(x_train_mnist_reshape)  #features after PCA
x_train_mnist_approx = pca_object.inverse_transform(x_train_mnist_intmde)

# Reconstruction error calculation

reconstruction_error = np.sum(np.square(x_train_mnist_reshape - x_train_mnist_approx)) / (
            x_train_mnist_reshape.shape[0] * x_train_mnist_reshape.shape[1])
print("\nMean squared error:%f" % (reconstruction_error))

# Plot 10 reconstructed images

reconstruction_titles = ["True image %d " % i for i in range(y_train_mnist.shape[0])]
true_titles = ["Reconstruction of image %d " % i for i in range(y_train_mnist.shape[0])]


def plot_gallery_2(images1, images2, titles1, titles2, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(1, n_row * n_col + 1):
        plt.subplot(n_row, n_col, 2 * i - 1)
        plt.imshow(images1[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles1[i], size=12)
        plt.xticks(())
        plt.yticks(())
        plt.subplot(n_row, n_col, 2 * i)
        plt.imshow(images2[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles2[i], size=12)
        plt.xticks(())
        plt.yticks(())
        # plt.show()
        if 2 * i == n_row * n_col or 2 * i - 1 == (n_row * n_col - 1):
            break


plot_gallery_2(x_train_mnist_reshape, x_train_mnist_approx, reconstruction_titles, true_titles, h_mnist, w_mnist,
               n_row=5, n_col=4)



#Question 4
print("")
print('Digits 4 and 2 have been selected.\n')
# Include data with only digits 2 and 4
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
random_state = 170

x_train_mnist_2 = x_train_mnist[(y_train_mnist[:] == 2) | (y_train_mnist[:] == 4)]
y_train_mnist_2 = y_train_mnist[(y_train_mnist[:] == 2) | (y_train_mnist[:] == 4)]
h_mnist, w_mnist = x_train_mnist.shape[1:]
x_train_mnist_2_reshape = np.reshape(x_train_mnist_2, newshape = (x_train_mnist_2.shape[0], h_mnist*w_mnist))

# number of features, number of classes and number of principal components to calculate

no_features = x_train_mnist_2_reshape.shape[1]
no_classes = np.unique(y_train_mnist_2).shape[0]
no_components = 150

# fit PCA

pca_mnist_2 = PCA(n_components=no_components, svd_solver='randomized', whiten=True).fit(x_train_mnist_2_reshape)

# extract eigenfaces

eigenfaces_mnist_2 = pca_mnist_2.components_.reshape((no_components, h_mnist, w_mnist))

# eigenface titles

eigenface_titles_mnist_2 = ["eigenface %d" % i for i in range(1, eigenfaces_mnist_2.shape[0]+1)]

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

# Transform data to 2d for k means

pca_object_2 = PCA(n_components = 2, svd_solver='randomized', whiten=True)
x_train_mnist_2d = pca_object_2.fit_transform(x_train_mnist_2_reshape)
plt.figure(figsize = (10, 4))
plt.subplot(121)
plt.title("2d representation")
plt.scatter(x_train_mnist_2d[:, 0][y_train_mnist_2 == 2], x_train_mnist_2d[:, 1][y_train_mnist_2 == 2], c='r', label = '2')
plt.scatter(x_train_mnist_2d[:, 0][y_train_mnist_2 == 4], x_train_mnist_2d[:, 1][y_train_mnist_2 == 4], c='b', label = '4')
plt.legend()
plt.show()

# k means implementation
#
# kmeans = KMeans(n_clusters=2, random_state=random_state).fit(x_train_mnist_2d)
# plt.subplot(122)
# plt.scatter(x_train_mnist_2d[:, 0][kmeans.labels_ == 1], x_train_mnist_2d[:, 1][kmeans.labels_ == 1], c = 'r', label = '2')#
# plt.scatter(x_train_mnist_2d[:, 0][kmeans.labels_ == 0], x_train_mnist_2d[:, 1][kmeans.labels_ == 0], c = 'b', label = '4')#
# plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1])
# plt.title("K means fit")
# plt.legend()
# plt.show()
#
# # Plot top 10 eigenfaces
# print('\nPlotting top 10 eigenfaces..')
# print(len(eigenface_titles_mnist_2))
# plot_gallery(eigenfaces_mnist_2, eigenface_titles_mnist_2, h_mnist, w_mnist, n_row = 2, n_col = 5)
# plt.show()

