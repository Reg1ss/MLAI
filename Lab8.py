import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set() # for statistical data visualization
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import fetch_20newsgroups

X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)   #generate a clustering dataset
print(X.shape)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu');  #scatter picture

#plot
fig, ax = plt.subplots()    #several sub pictures
ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
ax.set_title('Naive Bayes Model', size=14)

xlim = (-8, 8)
ylim = (-15, 5)

xg = np.linspace(xlim[0], xlim[1], 60)
yg = np.linspace(ylim[0], ylim[1], 40)
xx, yy = np.meshgrid(xg, yg)
Xgrid = np.vstack([xx.ravel(), yy.ravel()]).T

for label, color in enumerate(['red', 'blue']):
    mask = (y == label)
    mu, std = X[mask].mean(0), X[mask].std(0)  # Estimate the mean and variance from the data
    P = np.exp(-0.5 * (Xgrid - mu) ** 2 / std ** 2).prod(1)
    Pm = np.ma.masked_array(P, P < 0.03)
    ax.pcolorfast(xg, yg, Pm.reshape(xx.shape), alpha=0.5,
                  cmap=color.title() + 's')
    ax.contour(xx, yy, P.reshape(xx.shape),
               levels=[0.01, 0.1, 0.5, 0.9],
               colors=color, alpha=0.2)

ax.set(xlim=xlim, ylim=ylim)

#GaussianNB
model = GaussianNB()    #automatically set the prior by the dataset
model.fit(X, y)

rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)  #the numbers are to control the x,y limits
ynew = model.predict(Xnew)
print((ynew.shape))

#multinomialNB