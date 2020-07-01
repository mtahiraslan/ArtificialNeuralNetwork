# -*- coding: utf-8 -*-
"""
Created on Sun May 12 14:42:07 2019

@author: M.Tahir
"""
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
#from sklearn.cross_validation import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
import keras
from keras.layers import Dense

import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
# Importing the dataset
dataset = pd.read_csv('./reklam_verisi.csv')

X = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values

label_encoder=LabelEncoder()
X[:, 1] = label_encoder.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] 
print(dataset)
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


ann=MLPClassifier(activation='logistic',max_iter=200, solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=1)
ann.fit(X_train,y_train)
print(" MLP Train Başarı: {:.3f}".format(ann.score(X_train,y_train)))
print("MLP Test Başarı: {:.3f}".format(ann.score(X_test,y_test)))

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

Optimizer=["Adam","Adamax","Nadam","SGD","RMSprop","Adagrad","Adadelta"]
basarilar=[]
matrisler=[]
best=0
atama=0
tut=0

for i in range(7):

    model = Sequential()

    model.add(Dense(output_dim = 2, init = 'uniform', activation = 'relu', input_dim =4))
    
    model.add(Dense(output_dim = 2, init = 'uniform', activation = 'relu'))
    
    model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

    model.compile(optimizer =Optimizer[i-1], loss = 'binary_crossentropy', metrics = ['accuracy'])

    model.fit(X_train, y_train, batch_size = 10, nb_epoch = 15)
    

    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.55) 
    print("")
    print("Tahmin Sonuçları")
    print("")
    print(y_pred)
    print("")
    print("------------------------------------------------------------------")
    
    cm = confusion_matrix(y_test, y_pred)
    matrisler.append(cm)
    scores=model.evaluate(X_test,y_test)
    
    accuracy=scores[1]*100
    best=accuracy
    if best>atama:
        atama=best
        tut=i    
    basarilar.append(accuracy)
    
    
print(" Relu-Relu-Sigmoid ")
print("")
print("------------------------------------------------------------------")
for i in range(7):    
    print("")
    print(Optimizer[i-1]+" Karmaşıklık Matrisi") 
    print("")
    print(matrisler[i-1])
    print("")
    print(Optimizer[i-1]+" Acc Sonucu:"+str(basarilar[i-1]))
    print("")
    
print("------------------------------------------------------------------")
print("")
print(" Relu-Relu-Sigmoid Katmanı İçin En İyi Acc Sonucu :"+Optimizer[tut])
print("")
print("ACC:"+str(atama))
print("")
print("------------------------------------------------------------------")
    

print("GridSearch İçin Parametrelerin Sonuç Değerlerinin Karşılaştırılması")
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

a=SVC(kernel='linear').fit(X_train,y_train)
print("Linear Sonucu",a.score(X_train,y_train))

a=SVC(kernel='rbf').fit(X_train,y_train)
print("Rbf Sonucu",a.score(X_train,y_train))

a=SVC(kernel='rbf',C=0.5).fit(X_train,y_train)
print("Rbf C=0.5 İçin Sonuç:",a.score(X_train,y_train))

a=SVC(kernel='rbf',C=0.1).fit(X_train,y_train)
print("Rbf C=0.1 İçin Sonuç:",a.score(X_train,y_train))

a=SVC(kernel='linear',C=0.5).fit(X_train,y_train)
print("Linear C=0.5 İçin Sonuç:",a.score(X_train,y_train))

a=SVC(kernel='linear',C=0.1).fit(X_train,y_train)
print("Linear C=0.1 İçin Sonuç:",a.score(X_train,y_train))

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy=accuracy_score(y_test,y_pred)
print("Başarı=",accuracy)

from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.01, 0.01,0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]}]

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
accuracy = grid_search.best_score_
print("GridSearch Acc=",accuracy)
print("En Başarılı GridSearch Parametreleri=",grid_search.best_params_)

classifier = SVC(kernel = 'rbf', gamma=0.7)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy=round(accuracy_score(y_test,y_pred),2)*100
print("En İyi Sonuçlara Göre Başarı Değeri:",accuracy)
