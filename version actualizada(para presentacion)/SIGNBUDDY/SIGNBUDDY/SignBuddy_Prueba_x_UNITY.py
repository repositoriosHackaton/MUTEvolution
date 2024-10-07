
import cv2
import numpy as np
import mediapipe as mp
import tensorflow 
from tensorflow import keras
from collections import deque
import time
import os
import socket  # Importar la librería socket

# Parámetros de la comunicación
serverAddressPort = ("127.0.0.1", 8080)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)



actions = []
sequence = []
sentence = []
predictions = []
threshold = 0.20
mp_holistic = mp.solutions.holistic  # Modelo Holístico
mp_drawing = mp.solutions.drawing_utils  # Utilidades de dibujo
model = keras.models.load_model('lector_model(99acc-97val_acc).keras')
train_dir = 'new_dataset/train' 
NP_PATH = 'new_dataset/NP_PATH'

# Cargar acciones desde el directorio
for action in os.listdir(NP_PATH):
    actions.append(action)

def draw_landmarks(image, results_holistic):
    # Configuración para líneas más delgadas
    landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    connection_drawing_spec = mp_drawing.DrawingSpec(thickness=1)

    if results_holistic.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results_holistic.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_drawing_spec,
            connection_drawing_spec=connection_drawing_spec)
    
    if results_holistic.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results_holistic.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=landmark_drawing_spec,
            connection_drawing_spec=connection_drawing_spec)
    
    if results_holistic.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results_holistic.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=landmark_drawing_spec,
            connection_drawing_spec=connection_drawing_spec)
    
    return image

def extract_keypoints(results_holistic):
    # Extracción de keypoints de la pose
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results_holistic.pose_landmarks.landmark]).flatten() if results_holistic.pose_landmarks else np.zeros(33*4)
    
    # Extracción de keypoints de la mano izquierda
    lh = np.zeros(21*3)
    if results_holistic.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results_holistic.left_hand_landmarks.landmark]).flatten()
    
    # Extracción de keypoints de la mano derecha
    rh = np.zeros(21*3)
    if results_holistic.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results_holistic.right_hand_landmarks.landmark]).flatten()
    
    return np.concatenate([pose, lh, rh])

def prob_viz(res, actions, image, color):
    output_image = image.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_image, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), color, -1)
        cv2.putText(output_image, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_image

# Definir un solo color para todas las acciones (por ejemplo, azul)
action_color = (255, 0, 0)  # Formato BGR (Azul)

width, height = 720, 640

# Inicializar MediaPipe Holistic
with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,                 # 0 para más rápido, 2 para más preciso
    smooth_landmarks=True,
    min_detection_confidence=0.5,       # Aumentar el umbral para reducir falsos positivos
    min_tracking_confidence=0.5) as holistic:

    # Captura de video
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        # Convertir la imagen a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Procesar la imagen con MediaPipe Holistic
        results_holistic = holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        
        # Dibujar los landmarks en la imagen
        image_with_landmarks = draw_landmarks(image, results_holistic)
        
        # Verificar si se detecta la pose o manos
        if results_holistic.left_hand_landmarks or results_holistic.right_hand_landmarks:
            keypoints = extract_keypoints(results_holistic)
            
            # Añadir keypoints a la secuencia y mantener solo los últimos 33
            sequence.append(keypoints)
            sequence = sequence[-34:]
            current_time = time.time()
            
            if len(sequence) == 34:
                # Convertir la secuencia a un array numpy
                input_data = np.expand_dims(np.array(sequence), axis=0)  # Forma: (1, 33, 258)
                
                # Realizar predicción
                res = model.predict(input_data)[0]
                predicted_action = actions[np.argmax(res)]
                print(predicted_action)
                predictions.append(np.argmax(res))
                sequence.clear()  # Limpiar la secuencia cuando se hace una predicción
                
                # Lógica de visualización
                if len(predictions) >= 10:
                    most_common = np.argmax(np.bincount(predictions[-10:]))
                    if res[most_common] > threshold:
                        if len(sentence) > 0:
                            if most_common != actions.index(sentence[-1]):
                                sentence.append(actions[most_common])
                        else:
                            sentence.append(actions[most_common])
                
                # Enviar las posiciones de las manos como una cadena formateada
                keypoints_str = ','.join([f"{coord:.3f}" for coord in keypoints])  # Formato: "x1,y1,z1,x2,y2,z2,...,x258"
                sock.sendto(str.encode(f"({keypoints_str})"), serverAddressPort)
                
                # Visualizar probabilidades
                image_with_landmarks = prob_viz(res, actions, image_with_landmarks, action_color)
                last_prediction_time = current_time
            
            # Mostrar la predicción en la imagen
            cv2.rectangle(image_with_landmarks, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image_with_landmarks, ' '.join(sentence), (3, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # # Dibujar el bounding box para manos izquierdas
            # if results_holistic.left_hand_landmarks:
            #     h, w, _ = image.shape
            #     landmarks = results_holistic.left_hand_landmarks.landmark
            #     xmin = min([lm.x for lm in landmarks]) * w
            #     xmax = max([lm.x for lm in landmarks]) * w
            #     ymin = min([lm.y for lm in landmarks]) * h
            #     ymax = max([lm.y for lm in landmarks]) * h
            #     area = (xmax - xmin) * (ymax - ymin)
            #     if area > 1000:
            #         cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            
            # Dibujar el bounding box para manos derechas
            # if results_holistic.right_hand_landmarks:
            #     h, w, _ = image.shape
            #     landmarks = results_holistic.right_hand_landmarks.landmark
            #     xmin = min([lm.x for lm in landmarks]) * w
            #     xmax = max([lm.x for lm in landmarks]) * w
            #     ymin = min([lm.y for lm in landmarks]) * h
            #     ymax = max([lm.y for lm in landmarks]) * h
            #     area = (xmax - xmin) * (ymax - ymin)
            #     if area > 1000:
            #         cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        
        else:
            # No se detecta pose ni manos
            sequence.clear()  # Limpiar la secuencia cuando no se detecta nada
            cv2.putText(image_with_landmarks, 'No hand or pose detected', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Mostrar la imagen procesada
        cv2.imshow('Real-time Prediction', image_with_landmarks)
        
        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
