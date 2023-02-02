# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 16:15:29 2021

@author: DyningAida
"""

import pandas as pd
#import seaborn as sns, matplotlib.pyplot as plt, numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, f1_score, classification_report

df = pd.read_csv("mental_illness.csv", encoding='latin1')
# encode atribut
encode = LabelEncoder()
df['pendidikan'] = encode.fit_transform(df['pendidikan'])
df['usia'] = encode.fit_transform(df['usia'])
df['gender'] = encode.fit_transform(df['gender'])
# shuffle data
df = df.sample(frac=1)
# independent variable
x = df.drop(["mental_illness","pendidikan","memiliki_komputer","akses_internet_memadai",
                "tinggal_di_rumah_subsidi","gender","kerja_parttime"
                ,"bekerja_dan_sekolah","menerima_bantuan_sosial"], axis = 1)
x.head()
# dependent variable
y = df["mental_illness"]
y.head()
# split data train dan data test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 123)
# memanggil naive bayes classifier
nb = GaussianNB()
# menginputkan data training
nb_train = nb.fit(x_train, y_train)
# Menentukan hasil prediksi dari x_test
y_pred = nb_train.predict(x_test)
y_pred
#np.array(y_test)
# Menentukan probabilitas hasil prediksi
nb_train.predict_proba(x_test)
#membuat confusion matrix
#confusion_matrix(y_test, y_pred)
#def display_conf(y_test, y_pred):
#    sns.heatmap(confusion_matrix(y_test, y_pred),annot=True,linewidths=3,cbar=False)
#    plt.title('Confusion Matrix')
#    plt.ylabel('Actual')
#    plt.xlabel('Prediction')
#    plt.show()
# Memanggil fungsi untuk menampilkan visualisasi confusion matrix
#display_conf(y_test, y_pred)
#f1_score(y_test, y_pred, labels=np.unique(y_pred), average='weighted')
#print classification report
print(classification_report(y_test, y_pred))