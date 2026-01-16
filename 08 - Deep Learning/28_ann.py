import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from keras import Sequential, layers

# Parte 0: Explicación del problema
# Se nos da la información de clientes de un banco, en el que ultimamente los clientes
# se han estado yendo del banco. Nuestro objetivo es idendificar los patrones de los
# clientes que se van y poder predecir si algún cliente se irá en el futuro.

# Parte 1: Perparar los datos

# Importar el conjunto de datos
df: pd.DataFrame = pd.read_csv("Churn_Modelling.csv")

# Crear las matrices de valores con las variables independientes y la dependiente
X: np.ndarray = df.iloc[:, 3:-1].values
y: np.ndarray = df.iloc[:, -1].values

# Convertir la variaable categorica "gender" a una columna numerica
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])  # [0, 1, ...]

# Convertir la variable categorica "geograhy" en variables dummy
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough"
)
X: np.ndarray = np.array(ct.fit_transform(X))  # [[0, 1, 0, ...], [1, 0, 0, ...], ...]

# Separar el conjunto de datos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Estandarizar las variables. Rango: (0, 1)
sc = StandardScaler()
X_train: np.ndarray = sc.fit_transform(X_train)  # Escalar el conjunto de entrenamiento
X_test: np.ndarray = sc.transform(X_test)

# Parte 2: Preparar la red neuronal artifucial

# Iniciar la red neuronal
ann = Sequential()

# El número de neuronas o unidades esta dada por el promedio de todas las
# variables de entrada (11) y la variable de salida (1)
# 11 + 1 = 12 -> 12/2 = 6

# Agregar capas ocultas a la red
ann.add(layers.Dense(units=6, activation="relu"))
ann.add(layers.Dense(units=6, activation="relu"))
# ¿Por qué dos capas ocultas?
# Dado que el problema se trata de entender los patrones de clientes que se salen del
# banco, no nos basta con una sola capa oculta, ya que pueden existir multiples factores
# que influyan esta desición. Es por eso que se le agrega una capa más para profundizar
# en el aprendizaje y que sea más sensible a patrones complejos.

# Agregar la capa de salida, en este caso nuestra variable dependiente es binaria (0,1),
# por lo que con una unidad basta
ann.add(layers.Dense(units=1, activation="sigmoid"))

ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Parte 3: Entrenar la red neuronal artificial
# 100 epochs es un buen número para que la red neuronal aprenda los patrones, pero no
# son demasiados como para sobreajudtar
ann.fit(X_train, y_train, batch_size=32, epochs=100)
