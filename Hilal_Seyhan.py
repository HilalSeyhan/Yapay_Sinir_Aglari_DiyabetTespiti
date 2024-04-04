# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 18:11:28 2022

@author: Hilal
"""

#Kütüphaneleri Dahil Ettim
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Hata alındığı için eklendi
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Veri Setini tanımladım
data = pd.read_csv("diabetes.csv")

#Özellik Matrisinin Oluşturdum
X = data.iloc[:, 0:-1].values


#Bağımlı Kategorik Değişkenleri Oluşturma
Y = data.iloc[:, -1].values
print(X)
print(Y)

#Veri setini ayırdım
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Unsur ölçekleme gerçekleştirdim
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#YSA' yı başlatıldı
ann = tf.keras.models.Sequential()

#İlk gizli katman
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
#İkinci gizli katman
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

#ANN' i derleme
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

X_train = np.asarray(X_train).astype('float32')
Y_train = np.asarray(Y_train).astype(np.float32)

#ANN' i eğitim verileri ile eğittim
history = ann.fit(X_train, Y_train, batch_size=32, epochs=100)


#SubPlot Yapımı
df = data.copy()
fig, ax = plt.subplots()
ax.scatter(df.Pregnancies, df.SkinThickness, c=df.Outcome, cmap="viridis")
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('SubPlot')
plt.ylabel('Pregnancies')
plt.xlabel('SkinThickness')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#ScatterPlot Yapımı
df = data.copy()
sns.scatterplot(x="Pregnancies", y="SkinThickness", data=data)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('ScatterPlot')
plt.ylabel('SkinThickness')
plt.xlabel('Pregnancies')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#CountPlot Yapımı
df = data.copy()
sns.countplot(x='Pregnancies', data=data)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('CountPlot')
plt.ylabel('Count')
plt.xlabel('Pregnancies')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#BoxPlot yapımı
df = data.copy()
sns.boxplot(x = df["Pregnancies"])
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('BoxPlot')
plt.ylabel(' ')
plt.xlabel('Pregnancies')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Violin yapımı
sns.catplot(y = "BloodPressure", kind = "violin", data=data)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('Violin')
plt.ylabel('BloodPressure')
plt.xlabel(' ')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Lmplot Yapımı
df = data.copy()
sns.lmplot(x="Pregnancies", y="BMI", data=data)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('Lmplot')
plt.ylabel('BMI')
plt.xlabel('Pregnancies')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#ScatterPlot Matrisi (PairPlot) Yapımı
df = data.copy()
sns.pairplot(data)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('PairPlot')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()