'''
Jontavius Caston
DSC_550
October 4, 2019
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

'''Importing the Data and looking at the .head and a description of the dataset'''
pokemon = "C:/Users/lionb/Desktop/pokemon.xlsx"
data = pd.read_excel(pokemon)
print('Here are the first 10 entries of the dataset\n', data.head(10))
print('Here are the statistics of the dataset\n', data.describe(include='all'))
print('This is the number of entries in our dataset\n', data.shape)
print('\nHere is the number of unique elements\n', data.nunique)

'''Now we will check to see if there are any nulls in the dataset and then if any how many are there'''
print(data.isnull().values.any())
print(data.isnull().sum())

corr = data.corr()
print('Here is the correlation oof the data\n', corr)

'''A Histogram so that we can better see what we are dealing with'''
pokeHist = plt.hist(data['Type'], bins=18)
plt.xticks(rotation='vertical')
pokeX = plt.xlabel('Pokemon Types')
pokeY = plt.ylabel('Number of Pokemon Per Type')
plt.show()

'''Here is a scatter plot of our data'''
types = data['Type']
plt.scatter(data['Type'], data['Attack'], label='Attack', s=10, alpha=0.50)
plt.xticks(rotation='vertical')
plt.xlabel('Pokemon Type')
plt.ylabel('Attack Power')
plt.title('Pokemon Attack Power by Type')
plt.legend()
plt.show()

'''Here we will plot a heat map of the correlation'''
sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns, cmap='BuGn')
plt.show()

'''Now we will standardize our feature matrix'''
x = data[['Number', 'Total', 'HP', 'Attack', 'Defense', 'SpecialAttack', 'SpecialDefense', 'Speed']]
y = data[['Name', 'Type']]
yDf = y
fit = StandardScaler().fit_transform(x)
print('Here are the original number of features\n', fit)

'''Here we will create a PCA keeping 2 principle components'''
pca = PCA(n_components=2, whiten=True)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
print('Here are the reduced number of features\n', principalComponents)
print('Here is the PCA Data Frame\n', principalDf)

'''Now we will concatenate the PCA Data Frame with our removed labels to make our final Data Frame'''
finalDf = pd.concat([principalDf, yDf], axis=1)
print('Here is the final Data Frame complete with labels\n', finalDf)

'''Now we will practice a little model evaluation using a couple of different methods'''
#define x and y
feature_cols = data[['HP', 'Attack', 'Defense', 'Speed']]
x = feature_cols
y = data.Type
# split x and y
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2)
logreg = LogisticRegression()


# fit your 2 train variables to the Logistic Regression
logreg.fit(x_train, y_train)
y_pred_class = logreg.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred_class))

# Check classification accuracy of KNN with k = 5
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))

# creating a confusion matrix
print(metrics.confusion_matrix(y_test, y_pred_class))
