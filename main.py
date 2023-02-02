from flask import Flask, app, render_template, request, redirect, url_for
import pandas as pd
#import seaborn as sns, matplotlib.pyplot as plt, numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)

data_diri = []
data_kondisii = []
tabel = ['memiliki_komputer','inet',
    'kerjasekolah','bantuan','tinggal_subsidi','gender']
tabel_kondisi = ['tinggal_dengan_ortu','pernah_gapyear','lama_gapyear',
                'pernah_dirawat','banyak_dirawat',
                'disable','usia','anxiety','depresi','obessive','mood_swing',
                'gangguan_kecemasan','compulsive','mudah_lelah','susah_konsentrasi']

@app.route("/",methods=['POST','GET'])
def index():
    if request.method == "POST":
        for i in tabel:
            data_diri.append(int(request.form[i]))
        return render_template('data_diri.html',data=data_diri)
    return render_template('form.html')

@app.route("/save_profile",methods=['POST'])
def save_profile():
    if request.method == "POST":
        # save data dimasukkan kesini
        return render_template('result.html')
    return "data gagal disimpan"

@app.route("/data_kondisi",methods=['POST','GET'])
def data_kondisi():
    if request.method == "POST":
        for a in tabel_kondisi:
            data_kondisii.append(int(request.form[a]))
        print(data_kondisii)
        return render_template('data_kondisi.html', data=data_kondisii)
    return render_template('form_kondisi.html')
    
@app.route("/save_kondisi",methods=['POST','GET'])
def save_kondisi():
    if request.method == "POST":
        # analisis juga dipanggil disini
        df = pd.read_csv("mental_illness.csv", sep=',', encoding='latin1')
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
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 123)
        # memanggil naive bayes classifier
        nb = GaussianNB()
        # menginputkan data training
        nb_train = nb.fit(x_train, y_train)
        # Menentukan hasil prediksi dari x_test
        y_pred = nb_train.predict([data_kondisii])
        print(y_pred)
        return render_template('analysis_result.html', y_pred=y_pred)
    return render_template('result.html')

def diagnose(x_test):
    df = pd.read_csv("mental_illness.csv", sep=',')
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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 123)
    # memanggil naive bayes classifier
    nb = GaussianNB()
    # menginputkan data training
    nb_train = nb.fit(x_train, y_train)
    # Menentukan hasil prediksi dari x_test
    y_pred = nb_train.predict(x_test)
    return y_pred

if __name__ == '__main__':
    app.run(debug=True)