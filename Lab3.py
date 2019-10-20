import pods
import pylab as plt
import numpy as np

# pods.notebook.display_google_book(id='spcAAAAAMAAJ', page=72)
data = pods.datasets.olympic_marathon_men()
data = type(data)

x = data['X']
y = data['Y']
print(x)
print(y)

plt.plot(x, y, 'rx')
plt.xlabel('year')
plt.ylabel('pace in min/km')

m = -0.4
c = 80

c = (y - m*x).mean()
print(c)
m = ((y - c)*x).sum()/(x**2).sum()
print(m)

x_test = np.linspace(1890, 2020, 130)[:, None]
f_test = m*x_test + c

plt.plot(x_test, f_test, 'b-')
plt.plot(x, y, 'rx')

for i in np.arange(10):
    m = ((y - c)*x).sum()/(x*x).sum()
    c = (y-m*x).sum()/y.shape[0]
print(m)
print(c)