import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import os


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


df = pd.read_excel("C:/Users/Diego/PycharmProjects/portfoliopython/Rede_Neural_IA/Gerador_de_AmostrasENGs.xlsx")


X = df.iloc[8:108, 5:11]
y = df.iloc[8:108, 12]


y = np.where(y == 1, 1, 0)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=40, random_state=42, stratify=y
)


print("\nContagem de classes:")
print("Treino:", np.unique(y_train, return_counts=True))
print("Teste:", np.unique(y_test, return_counts=True))


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=200,
    batch_size=8,
    verbose=0
)


loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
acc_percent = accuracy * 100


print(f"\nAcurácia final no conjunto de teste: {acc_percent:.2f}%")


plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Curva de Aprendizado')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()


max_acc = max(max(history.history['accuracy']), max(history.history['val_accuracy']))
x_pos = len(history.history['accuracy']) * 0.5
y_pos = max_acc * 0.85

plt.text(
    x_pos, y_pos,
    f"Acurácia final: {acc_percent:.2f}%",
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.8)
)


plt.show()