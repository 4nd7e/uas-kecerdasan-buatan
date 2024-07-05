#Nama  : Andrean Wahyudi
#NIM   : 211011401126
#Kelas : 06TPLP021

# Implementasi Regresi Linear dengan Python

Tutorial ini mengikuti langkah-langkah dari video [Tutorial Machine Learning Linear Regression dengan Python](https://www.youtube.com/watch?v=3x4b0U36qRU).

## 1. Import Library yang Diperlukan
Kita akan menggunakan `pandas`, `numpy`, dan `sklearn` untuk analisis data dan model machine learning.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

## 2. Load Dataset
Di sini, kita akan menggunakan dataset yang sudah ada atau mengunggah dataset kita sendiri.

# Misalnya kita menggunakan dataset yang tersedia di sklearn
from sklearn.datasets import load_boston
boston = load_boston()

# Konversi ke DataFrame
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target
data.head()

## 3. Pisahkan Data untuk Training dan Testing
Kita akan membagi data menjadi dua bagian: satu untuk melatih model dan satu lagi untuk menguji akurasi model.

    X = data.drop('PRICE', axis=1)
y = data['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## 4. Melatih Model
Kita akan melatih model regresi linear menggunakan data training.

    model = LinearRegression()
model.fit(X_train, y_train)

## 5. Evaluasi Model
Terakhir, kita akan mengevaluasi performa model menggunakan data testing.

predictions = model.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()
