import cv2  # Biblioteca para procesamiento de imágenes y video
import numpy as np  # Biblioteca para manejo de arrays y operaciones matemáticas
import tensorflow as tf  # Biblioteca para aprendizaje profundo
from tensorflow import keras  # Módulo de Keras para manejar redes neuronales
from collections import deque  # Estructura de datos de cola doblemente enlazada
import time  # Módulo para gestionar el tiempo
import os  # Módulo para interactuar con el sistema de archivos
import socket  # Biblioteca para manejar la comunicación en red (sockets)
from cvzone.HandTrackingModule import HandDetector  # Detector de manos de la librería CVZone
from cvzone.PoseModule import PoseDetector  # Detector de poses corporales de CVZone

# Parámetros de la comunicación por red utilizando protocolo UDP
serverAddressPort = ("127.0.0.1", 8080)  # Dirección IP y puerto del servidor
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Se configura un socket de tipo UDP

# Cargar modelo de aprendizaje profundo previamente entrenado
actions = []  # Lista para almacenar las acciones posibles
sequence = []  # Lista para almacenar las secuencias de keypoints
sentence = []  # Lista para almacenar las acciones detectadas de forma continua
predictions = []  # Lista para almacenar las predicciones realizadas
threshold = 0.80  # Umbral de confianza para las predicciones
model = keras.models.load_model('lector_model(95acc-95val_acc-95test).keras')  # Se carga el modelo entrenado
NP_PATH = 'new_dataset/NP_PATH'  # Ruta donde están almacenadas las acciones

# Cargar las acciones desde el directorio de datos
for action in os.listdir(NP_PATH):  # Recorre el directorio para cargar las acciones
    actions.append(action)  # Añade cada acción encontrada a la lista

# Definir un color único para visualizar todas las acciones (formato BGR)
action_color = (255, 0, 0)  # Color azul en formato BGR (Azul, Verde, Rojo)

# Parámetros para el detector de manos y de poses corporales
hand_detector = HandDetector(maxHands=2, detectionCon=0.8)  # Detecta hasta 2 manos con 80% de confianza
pose_detector = PoseDetector(modelComplexity=2)  # Detector de poses con complejidad de modelo 2

# Configuración de las dimensiones de la ventana de video
width, height = 840, 920

# Función para extraer los keypoints de las manos y del cuerpo (pose)
def extract_keypoints(hands, pose_lms):
    # Extrae keypoints de la pose (33 puntos)
    if pose_lms:
        pose_keypoints = np.array([[kp[0], kp[1], kp[2], 1] for kp in pose_lms]).flatten()  # Convierte keypoints a un array plano
    else:
        pose_keypoints = np.zeros(33 * 4)  # Si no hay detección, rellena con ceros

    # Keypoints de la mano izquierda (21 puntos)
    lh = np.zeros(21 * 3)  # Si no hay detección, rellena con ceros
    if hands and len(hands) > 0:  # Si se detecta al menos una mano
        hand_left = hands[0]  # La primera mano detectada
        lh = np.array([[kp[0], kp[1], kp[2]] for kp in hand_left['lmList']]).flatten()  # Extrae keypoints y los aplana

    # Keypoints de la mano derecha (21 puntos)
    rh = np.zeros(21 * 3)  # Si no hay detección, rellena con ceros
    if hands and len(hands) > 1:  # Si se detectan dos manos
        hand_right = hands[1]  # La segunda mano detectada
        rh = np.array([[kp[0], kp[1], kp[2]] for kp in hand_right['lmList']]).flatten()  # Extrae keypoints y los aplana

    # Concatenar todos los keypoints (pose + mano izquierda + mano derecha)
    return np.concatenate([pose_keypoints, lh, rh])

# Función para visualizar las probabilidades de cada acción
def prob_viz(res, actions, image, color):
    output_image = image.copy()  # Copia la imagen original
    for num, prob in enumerate(res):  # Itera sobre las probabilidades
        # Dibuja una barra para cada acción indicando la probabilidad
        cv2.rectangle(output_image, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), color, -1)
        # Muestra el nombre de la acción sobre la barra
        cv2.putText(output_image, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_image  # Devuelve la imagen con las visualizaciones

# Captura de video desde la cámara
cap = cv2.VideoCapture(1)  # Usa la cámara predeterminada
cap.set(3, width)  # Define el ancho de la imagen de video
cap.set(4, height)  # Define el alto de la imagen de video

# Bucle principal para procesar el video en tiempo real
while cap.isOpened():
    success, image = cap.read()  # Captura un frame de la cámara
    if not success:
        break  # Si no hay imagen, sale del bucle
    
    # Detectar manos y pose en la imagen capturada
    hands, image_with_landmarks = hand_detector.findHands(image)  # Detección de manos
    pose_image = pose_detector.findPose(image_with_landmarks, draw=True)  # Detección de pose
    pose_lms, bboxInfo = pose_detector.findPosition(pose_image, bboxWithHands=False)  # Obtiene los puntos clave de la pose

    # Si se detectan manos
    if hands:
        keypoints = extract_keypoints(hands, pose_lms)  # Extrae los keypoints

        # Añadir los keypoints a la secuencia y mantener solo los últimos 34 frames
        sequence.append(keypoints)
        sequence = sequence[-34:]

        current_time = time.time()  # Guarda el tiempo actual

        if len(sequence) == 34:  # Si la secuencia tiene 34 frames
            input_data = np.expand_dims(np.array(sequence), axis=0)  # Convierte la secuencia a un array de entrada

            # Realiza la predicción utilizando el modelo
            res = model.predict(input_data)[0]
            predicted_action = actions[np.argmax(res)]  # Acción con mayor probabilidad
            print(predicted_action)
            predictions.append(np.argmax(res))  # Guarda la predicción

            sequence.clear()  # Limpia la secuencia después de la predicción

            # Mostrar la predicción más común en los últimos 10 frames si supera el umbral
            if len(predictions) >= 10:
                most_common = np.argmax(np.bincount(predictions[-10:]))  # Predicción más común
                if res[most_common] > threshold:
                    if len(sentence) > 0:
                        if most_common != actions.index(sentence[-1]):  # Evitar duplicados
                            sentence.append(actions[most_common])
                    else:
                        sentence.append(actions[most_common])

            # Enviar los puntos clave a través de la red
            keypoints_str = ','.join([f"{coord:.3f}" for coord in keypoints])  # Formatea los keypoints como una cadena
            sock.sendto(str.encode(f"({keypoints_str})"), serverAddressPort)  # Envía la cadena al servidor

            # Visualizar las probabilidades (opcional)
            # image_with_landmarks = prob_viz(res, actions, image_with_landmarks, action_color)
            last_prediction_time = current_time  # Guarda el tiempo de la última predicción

    else:
        # Si no se detectan manos, limpiar la secuencia
        sequence.clear()
        cv2.putText(image_with_landmarks, 'No hand detected', (10, 30),  # Muestra un mensaje si no hay manos detectadas
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Mostrar la imagen procesada con los resultados en tiempo real
    cv2.imshow('Real-time Prediction', image_with_landmarks)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos de la cámara y cerrar las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()
