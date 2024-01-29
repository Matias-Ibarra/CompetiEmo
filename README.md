# CompetiEmo
Competición de Kaggle sobre detección de emociones.

[emociones](./img/emociones.jpg)

[Enlace](https://www.kaggle.com/competitions/emodecode-3000) a la competición en Kaggle. La carpeta de datos no forma parte del repositorio por una cuestión de memoria.

Descripción:

Reconocimiento de Expresiones Faciales

Se cree que un humano tiene al menos 7 emociones básicas, donde la tristeza y la felicidad son las que la gente suele mencionar más. Para el ser humano, es algo así como un trabajo fácil descubrir cómo se siente una persona por la expresión en su rostro. Pero esta vez, debes entrenar una Red Neuronal, utilizando las capas y parámetros que puedas considerar necesarios para obtener la mayor precisión, diferenciando entre personas tristes y felices.

Tienes razón, las personas pueden sentir más de una emoción al mismo tiempo, pero olvidemos eso por un segundo e intentemos simplificar el problema.

Las imágenes ya están en blanco y negro, tienen un tamaño de 48x48 y tienen los rostros recortados, lo que debería hacer todo un poco más fácil. Sin embargo, siéntete libre de hacer cualquier cambio en las imágenes, pero recuerda hacer el mismo tipo de tratamiento tanto para el conjunto de imágenes de entrenamiento como para el conjunto de imágenes de prueba, justo antes de hacer las predicciones.

Ahora tienes herramientas como OpenCV, NN y CNN para mejorar tu última competencia.

¿Estás feliz? ¿Estás triste? ¿Estás enojado o quieres matar a alguien? ¡Vamos a darle un toque de humor a esto!

```python

model = keras.Sequential([
    # 32 matrices de salida de 48x48
    # El kernel de 3x3 toma matrices de este tamaño dentro de la imágen y aplica el producto escalar.
    # Función de activación ReLu ---> max(0,x)
    keras.layers.Conv2D(32, (3, 3), padding='same', activation="relu", input_shape=(48, 48, 1)),
    keras.layers.Conv2D(64, (3, 3), padding='same', activation="relu"),
    # BatchNormalization mejora la taza de aprendizaje
    keras.layers.BatchNormalization(),
    # Con Maxpooling reducimos a la mitad y nos quedamos con los valores mas altos.
    # Nos quedan 64 imágenes de 24x24
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # Dropout hace que se ignoren ciertas neuronas al azar, entonces la red se vuelve menos sensible
    # a los pesos específicos de las neuronas. Generaliza mejor y es menos probable que se sobreadapte
    # a los datos de entrenamiento
    keras.layers.Dropout(0.25),
    
    keras.layers.Conv2D(128, (5, 5), padding='same', activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),

    # l2 penaliza los coeficientes y la pendiente de la línea irá más hacia 0, pero nunca será igual a 0.
    keras.layers.Conv2D(512, (3, 3), padding='same', activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(512, (3, 3), padding='same', activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),

    # Flatten() convierte las matrices multidimensionales en matrices unidimensionales aplanadas.
    keras.layers.Flatten(),
    # Dense() cambia las dimensiones del vector, en este caso 256 es el tamaño de la capa de salida.
    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    
    keras.layers.Dense(512, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),

    # En la ultima capa de salida se suele utilizar softmax.
    keras.layers.Dense(7, activation="softmax")
])

```

Matriz predicciones vs valor real:

[Matriz](./img/matriz.png)

Resultados de la competición ---> [Enlace](https://www.kaggle.com/competitions/emodecode-3000/leaderboard)


Integrantes: María Neches, Matias Ibarra
