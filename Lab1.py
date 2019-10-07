import pods
import zipfile
import pandas as pd # import the pandas library into a namespace called pd
import pylab as plt # this imports the plotting library in python
import numpy as np
from imdb import IMDb


# print("This is the Jupyter notebook")
# print("It provides a platform for:")
# words = ['Open', 'Data', 'Science']
# from random import shuffle
# for i in range(3):
#     shuffle(words)
#     print(' '.join(words))

# pods.util.download_url('https://github.com/sjmgarnier/R-vs-Python/archive/master.zip')
# zip = zipfile.ZipFile('./master.zip', 'r')
# for name in zip.namelist():
#     zip.extract(name, '.')
film_deaths = pd.read_csv('./R-vs-Python-master/Deadliest movies scrape/code/film-death-counts-Python.csv')
print(film_deaths.describe())
# film_deaths.describe?
print(film_deaths['Year'])
print(film_deaths['Body_Count'])

plt.plot(film_deaths['Year'], film_deaths['Body_Count'], 'rx')
# plt.show()

film_deaths[film_deaths['Body_Count']>200]
film_deaths[film_deaths['Body_Count']>200].sort_values('Body_Count', ascending=False)
film_deaths['Body_Count'].hist(bins=20) # histogram the data with 20 bins.
plt.title('Histogram of Film Kill Count')
# plt.show()
deaths = (film_deaths.Body_Count>40).sum()  # number of positive outcomes (in sum True counts as 1, False counts as 0)
total_films = film_deaths.Body_Count.count()
print('total films:',total_films)
print('death:', deaths)
prob_death = float(deaths)/float(total_films)
print("Probability of deaths being greather than 40 is:", prob_death)

#Conditioning
for year in [2000, 2002]:
    deaths = (film_deaths.Body_Count[film_deaths.Year==year]>40).sum()
    total_films = (film_deaths.Year==year).sum()

    prob_death = float(deaths)/float(total_films)
    print("Probability of deaths being greather than 40 in year", year, "is:", prob_death)

Year = []
prob_death_Year = []
for year in range(1949,2010):
    if((film_deaths.Year==year).sum() != 0):
        deaths = (film_deaths.Body_Count[film_deaths.Year == year] > 40).sum()
        total_films = (film_deaths.Year == year).sum()
        prob_death = float(deaths) / float(total_films)
        Year.append(year)
        prob_death_Year.append(prob_death)


array_prob_death_Year = np.array(prob_death_Year)
# print(array_prob_death_Year)
plt.plot(Year,prob_death_Year)
# plt.show()

# Conditioning & Joint probability distribution
# p(y|t) & p(y,t)
p_t = (film_deaths.Year==2002).sum()/(film_deaths.Body_Count.count())
p_y = (film_deaths.Body_Count>40).sum()/(film_deaths.Body_Count.count())
p_y_given_t = (film_deaths.Body_Count[film_deaths.Year==2002]>40).sum()/(film_deaths.Year==2002).sum()
p_t_given_y = (film_deaths.Body_Count[film_deaths.Year==2002]>40).sum()/(film_deaths.Body_Count>40).sum()
p_y_and_t = (film_deaths.Body_Count[film_deaths.Year==2002]>40).sum()/film_deaths.Body_Count.count()

print(p_t,p_y,p_t_given_y,p_y_given_t,p_y_and_t)
print('p(y,t) = ',p_y_and_t,' --- p(t)p(y|t) = ', p_t*p_y_given_t, ' --- p(y)p(t|y) = ', p_y*p_t_given_y)
print('p(y,t) = ',p_y_and_t,' --- p(t)p(y) = ', p_t*p_y)

# Marginal distribution
Year = []
prob_death_Year = []
total_films = (film_deaths.Body_Count).count()
print(total_films)
for year in range(1949,2010):
    if((film_deaths.Year==year).sum() != 0):
        deaths = (film_deaths.Year==year).sum()
        prob_death = float(deaths) / float(total_films)
        Year.append(year)
        prob_death_Year.append(prob_death)
counter = 0
for year in Year:
    print(year,prob_death_Year[counter])
    counter += 1


# # IMDb
# ia = IMDb()

# for movie in ia.search_movie('Batman'):
#     print(movie)
