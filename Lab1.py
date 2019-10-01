import pods
import zipfile
import pandas as pd # import the pandas library into a namespace called pd

# print("This is the Jupyter notebook")
# print("It provides a platform for:")
# words = ['Open', 'Data', 'Science']
# from random import shuffle
# for i in range(3):
#     shuffle(words)
#     print(' '.join(words))

pods.util.download_url('https://github.com/sjmgarnier/R-vs-Python/archive/master.zip')
zip = zipfile.ZipFile('./master.zip', 'r')
for name in zip.namelist():
    zip.extract(name, '.')
film_deaths = pd.read_csv('./R-vs-Python-master/Deadliest movies scrape/code/film-death-counts-Python.csv')
film_deaths.describe()