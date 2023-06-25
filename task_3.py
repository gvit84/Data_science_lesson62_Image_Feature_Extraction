import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skimage.feature import hog
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Завантаження датасету CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Перетворення зображень в ознаки HOG
def extract_hog_features(images):
    hog_features = []
    for image in images:
        hog_feature = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1)  #channel_axis=-1
        hog_features.append(hog_feature)
    return np.array(hog_features)

X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

# Побудова класифікатора SVM
svm = Pipeline([('scaler', StandardScaler()), ('classifier', LinearSVC())])

# Навчання класифікатора
svm.fit(X_train_hog, y_train.ravel())

# Прогнозування на тестових даних
y_pred = svm.predict(X_test_hog)

# Обчислення метрик точності
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Виведення результатів
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
