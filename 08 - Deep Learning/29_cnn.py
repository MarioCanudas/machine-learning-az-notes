from keras.utils import image_dataset_from_directory
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling

# Parte 0 : Explicación del problema
# Se nos da un conjunto de imageens de perror y gatos, nuestro objetivo es
# crear un modelo que clasifique una imagen entre gato y perro

# Parte 1: Preparar el conjunto de imagenes
training_set = image_dataset_from_directory(
    "dataset/training_set", label_mode="binary", batch_size=32, image_size=(64, 64)
)

test_set = image_dataset_from_directory(
    "dataset/test_set", label_mode="binary", batch_size=32, image_size=(64, 64)
)

# Parte 2: Construir la CNN
cnn = Sequential()

# Agregar capa de reescalado de imagenes, ya que, en la preparación de las imagenes
# unicamente fueron extraidas. Esto ayuda a que el entrenamiento sea más eficiente
cnn.add(Rescaling(1.0 / 255, input_shape=(64, 64, 3)))

# Agregar 3 capas conculacionales y capas de pooling, donde los filtros van aumentando
# conforme se avanza la red, para que el modelo profudice los patrones más complejos
# entre gatos y perros

# Capa 1
cnn.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
cnn.add(MaxPooling2D(pool_size=2, strides=2))

# Capa 2
cnn.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
cnn.add(MaxPooling2D(pool_size=2, strides=2))

# Capa 3
cnn.add(Conv2D(filters=128, kernel_size=3, activation="relu"))
cnn.add(MaxPooling2D(pool_size=2, strides=2))

# Agregar capa de aplanamiento de la imagen, que transforma la matriz en un vector
cnn.add(Flatten())

# Agregar capa full connected
cnn.add(Dense(units=256, activation="relu"))

# Agregar la capa de salida binaria (pero o gato)
cnn.add(Dense(units=1, activation="sigmoid"))

cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Parte 3: Entrenar la CNN (tardado)
cnn.fit(x=training_set, validation_data=test_set, epochs=25)
