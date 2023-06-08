

# Import module yang akan digunakan
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Load dataset
dataset1 = pd.read_csv('csv/data_training.csv')
dataset2 = pd.read_csv('csv/data_testing.csv')

X_train = dataset1.drop('kelas', axis=1)
y_train = dataset1['kelas']

X_test = dataset2.drop('kelas', axis=1)
y_test = dataset2['kelas']


# Inisiasi jumlah k pada kNN
knn9 = KNeighborsClassifier(n_neighbors=7)

# Train model menggunakan data training
knn9.fit(X_train, y_train)

# Melakukan prediksi / klasifikasi
Y_pred = knn9.predict(X_test)

# Print hasil klasifikasi
for i in range(10):
    print("Hasil Klasifikasi", i+1, ":", Y_pred[i])


'''
# looping singkat

def kNN(k, X_train, y_train, X_test):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    Y_pred = knn.predict(X_test)
    for i in range(10):
        print("Hasil Klasifikasi", i+1, ":", Y_pred[i])


for i in range(3, 10, 2):
    print(f"Hasil Testing dengan K={i}:", "\n")
    hasil = kNN(i, X_train, y_train, X_test)
    print("\n")
'''
