# MUTEvolution
En este espacio se sube el código creado para el grupo 

Se debe agregar toda la documentación que ustedes consideren pertinente para la compresión de los modelos usados, la ejecución del código y los resultados obtenidos. 
Puden, si desean, agregar imágenes o resultados obtenidos. 

Recuerden que este readme es su puerta de entrada para su proyecto. 

Un ejemplo puede ser: 
# SignBuddy ( amigo señas en español)

Our project is a Sign-Text translator with AI using neuronal networks.

## Tabla de contenidos

1. [Nombre](#Nombre)
2. [Descripción](#descripción)
3. [Arquitectura](#Arquitectura)
4. [Proceso](#Proceso)
5. [Funcionalidades](#Funcionalidades)
6. [Estado del proyecto](#EstadoDelProyecto)
7. [Agradecimientos](#Agradecimientos)


* Nombre del proyecto
SignBuddy ( amigo señas en español)
* Breve descripción del proyecto -> Alguna imagen o gif que muestre el proyecto
Our project is a Sign-Text translator with AI using neuronal networks.
![alt text](image.png)
* Arquitectura del proyecto + imagen

* Proceso de desarrollo:
-we created the dataset
- for data preprocessing we used mediapipe to analyze the movements of the person in the video and collected the keypoints in numpy arrays.
![alt text](image-1.png) ![alt text](image-2.png)
-error control using "Excception" exceptions was used during the creation of some functions when they failed, but then we deleted the try/exceptions.
-¿Qué modelo de Machine Learning están usando?
we use deep learning instead machine learning. We use a sequential neural network with LTSM layers and EarlyStopping for callback. We also use tensorboard to see the history graph
![alt text](image-3.png)

-Estadísticos (Valores, gráficos, …)
![alt text](image-4.png)
![alt text](image-5.png)
-Métrica(s) de evaluación del modelo
![alt text](image-6.png)
![alt text](image-7.png)
* Funcionalidades extra:
we are implementing an avatar (the friend sign) in blender that copies the user's movement instead of viewing the user's camera with the landmarks on top of him