import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import svm
np.random.seed(42)
import pickle
#load data
df = pd.read_csv(r'ml\breast_cancer.csv')
#print sample 
#print(df.info())
#drop unwanted serios
df.drop(['Unnamed: 32', 'id'],  axis = 1, inplace=True) 
#print(df.head())
#map
df['diagnosis'] = df['diagnosis'].map({'M' : 1, 'B' : 0})
#print(df.head())
x = df.drop('diagnosis', axis=1)
print(x.columns)
y = df['diagnosis']
#corration
#print(df.corr())
#split test/train
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2)
#select model
model = svm.SVC()
#train model
model.fit(x_train, y_train)
#validate model
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))
# #save model 
f=r'ml\model_cancer.pkl'
pickle.dump(model,open(f,'wb'))