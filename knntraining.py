
# Import module yang akan digunakan

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Load dataset
dataset = pd.read_csv('csv/data_training.csv')

# Membagi data training dan testing
X = dataset.drop('kelas', axis=1)
Y = dataset['kelas']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)


# Inisiasi jumlah k pada kNN
knn3 = KNeighborsClassifier(n_neighbors=3)
knn5 = KNeighborsClassifier(n_neighbors=5)
knn7 = KNeighborsClassifier(n_neighbors=7)

# ................................................................
# Train model menggunakan data training k=3
knn3.fit(X_train, Y_train)

# Melakukan prediksi / klasifikasi
Y_pred = knn3.predict(X_test)

# Akurasi model
print("\nK=3, Accuracy:", round(metrics.accuracy_score(Y_test, Y_pred)*100, 1), "%")
# ................................................................

# Train model menggunakan data training k=5
knn5.fit(X_train, y=Y_train)

# Melakukan prediksi / klasifikasi
Y_pred = knn5.predict(X_test)

# Akurasi model
print("\nK=5 Accuracy:", round(metrics.accuracy_score(Y_test, Y_pred)*100, 1), "%")

# ................................................................
# Train model menggunakan data training k=7
knn7.fit(X_train, Y_train)

# Melakukan prediksi / klasifikasi
Y_pred = knn7.predict(X_test)

# Akurasi model
print("\nK=7 Accuracy:", round(
    metrics.accuracy_score(Y_test, Y_pred)*100, 1), "%\n")

# ................................................................


'''
# looping singkat


def kNN(k, X_train, Y_train, X_test, Y_test):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    print(f"\nK={k} Accuracy:", round(
        metrics.accuracy_score(Y_test, Y_pred)*100, 1), "%\n")


for i in range(3, 10, 2):
    hasil = kNN(i, X_train, Y_train, X_test, Y_test)
'''
