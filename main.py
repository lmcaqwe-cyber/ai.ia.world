# Proyecto ai.ia.world
# Desarrollado por: Lenin Miguel Carranza Alcantar

# Importar bibliotecas necesarias
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Definir función para cargar datos
def cargar_datos(ruta_datos):
    datos = np.load(ruta_datos)
    return datos

# Definir función para entrenar modelo
def entrenar_modelo(datos):
    X = datos[:, :-1]
    y = datos[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    modelo = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    modelo.compile(optimizer='adam', loss='mean_squared_error')
    modelo.fit(X_train, y_train, epochs=100)
    return modelo

# Cargar datos y entrenar modelo
ruta_datos = 'datos.npy'
datos = cargar_datos(ruta_datos)
modelo = entrenar_modelo(datos)

# Guardar modelo entrenado
modelo.save('modelo_ai_ia_world.h5')