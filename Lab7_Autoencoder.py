import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import  matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np
import torchvision

# #Assignment2
# scale = 0.3
# #add nosie
# def nosiy(img):
#     img = img + scale * torch.randn(3,32,32)
#     img = np.clip(img,-1,1)
#     return img
# torch.manual_seed(1722846)
# transform = transforms.Compose(     #compose several processes to one
#     [transforms.ToTensor(),     #transform a [0,255] PIL(python image library).image to tensor
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),    #normalize the value to [-1,1], first():mean, second():std  x = (x - mean(x))/stddev(x)
#      transforms.Lambda(lambda img: nosiy(img))])
#
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# trainset = list(trainset)[:4096]


mnist_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()) #train: True for training set & False for test set
print(len(mnist_data))
mnist_data = list(mnist_data)[:4160]
print(len(mnist_data))

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # 1 input image channel, 16 output channel, 3x3 square convolution
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  #to range [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

torch.manual_seed(2020)
myAE=Autoencoder()
# print(myAE)

params = list(myAE.parameters())
print(len(params))  #result = 12 means the parameters in the 6 layers(weight & bias) are lined up in params
print(params[0].size())  # First Conv2d's .weight
print(params[1].size())  # First Conv2d's .bias
print(params[1])

#Hyperparameters for training
batch_size=64
learning_rate=0.05
max_epochs = 20

#Set the random seed for reproducibility
# torch.manual_seed(2020)
#Choose mean square error loss
criterion = nn.MSELoss()
#Choose the Adam optimiser
optimizer = torch.optim.Adam(myAE.parameters(), lr=learning_rate, weight_decay=1e-5)
#Specify how the data will be loaded in batches (with random shffling)
train_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True) #data sampling apparatus
#Storage
outputs = []

#Start training
counter = 0
loss1 = 0
loss3 = 0
for epoch in range(max_epochs):
    for data in train_loader:
        img, label = data
        # print(img.size())
        optimizer.zero_grad()
        recon = myAE(img)
        loss = criterion(recon, img)
        loss.backward()
        optimizer.step()
    if (epoch % 3) == 0:
        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
    outputs.append((epoch, img, recon),)
# print(loss.grad_fn)
# print(loss.grad_fn.next_functions[0][0])

# numImgs = 12;
# for k in range(0, max_epochs, 9):
#     plt.figure(figsize=(numImgs, 2))
#     imgs = outputs[k][1].detach().numpy()
#     recon = outputs[k][2].detach().numpy()
#     for i, item in enumerate(imgs):
#         if i >= numImgs: break
#         plt.subplot(2, numImgs, i + 1)
#         plt.imshow(item[0])
#         plt.show()
#
#     for i, item in enumerate(recon):
#         if i >= numImgs: break
#         plt.subplot(2, numImgs, numImgs + i + 1)
#         plt.imshow(item[0])
#         plt.show()

epochIndex=0;
img1Index=1;
img2Index=3;

imgs = outputs[epochIndex][1].detach().numpy()
x1 = outputs[epochIndex][1][img1Index,:,:,:];# first image
x2 = outputs[epochIndex][1][img2Index,:,:,:] # second image
x = torch.stack([x1,x2])     # stack them together so we only call `encoder` once
embedding = myAE.encoder(x)
e1 = embedding[0] # embedding of first image
e2 = embedding[1] # embedding of second image
print(e1.size())

embedding_values = []
for i in range(0, 10):
    e = e1 * (i/10) + e2 * (10-i)/10
    embedding_values.append(e)
embedding_values = torch.stack(embedding_values)

recons = myAE.decoder(embedding_values)
plt.figure(figsize=(10, 2))
for i, recon in enumerate(recons.detach().numpy()):
    plt.subplot(2,10,i+1)
    plt.imshow(recon[0])
plt.subplot(2,10,11)
plt.imshow(imgs[img2Index][0])
plt.show()
plt.subplot(2,10,20)
plt.imshow(imgs[img1Index][0])
plt.show()
