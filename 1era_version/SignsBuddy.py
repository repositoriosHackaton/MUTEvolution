import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
import tensorflow 
from collections import deque
import time, os, h5py
 


last_prediction_time = 0
prediction_interval = 1

actions = []
sequence = []
sentence = []
predictions = []
threshold = 0.5
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands()
pose = mp_pose.Pose()
model = keras.models.load_model('lector_model(99acc-97val_acc).keras')
train_dir = 'new_dataset/train' 
NP_PATH = 'new_dataset/NP_PATH'
for action in os.listdir(NP_PATH):
    actions.append(action)


# def draw_landmarks(image, results_hands, results_pose):
#     mp_drawing = mp.solutions.drawing_utils
#     mp_hands = mp.solutions.hands
#     mp_pose = mp.solutions.pose
#     if results_pose.pose_landmarks:
#         mp_drawing.draw_landmarks(
#             image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#     if results_hands.multi_hand_landmarks:
#         for hand_landmarks in results_hands.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#     return image
def draw_landmarks(image, results_hands, results_pose):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    # Configuración para líneas más delgadas
    landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    connection_drawing_spec = mp_drawing.DrawingSpec(thickness=1)

    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_drawing_spec,
            connection_drawing_spec=connection_drawing_spec)
    
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=landmark_drawing_spec,
                connection_drawing_spec=connection_drawing_spec)
    
    return image

def extract_keypoints(results_hands, results_pose):
    # # pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results_pose.pose_landmarks.landmark]).flatten() if results_pose.pose_landmarks else np.zeros(33*4)
    
    # # if results_hands.multi_hand_landmarks:
    # #     lh = np.array([[res.x, res.y, res.z] for res in results_hands.multi_hand_landmarks[0].landmark]).flatten()
    # #     if len(results_hands.multi_hand_landmarks) > 1:
    # #         rh = np.array([[res.x, res.y, res.z] for res in results_hands.multi_hand_landmarks[1].landmark]).flatten()
    # #     else:
    # #         rh = np.zeros(21*3)
    # # else:
    # #     lh = np.zeros(21*3)
    # #     rh = np.zeros(21*3)
    
    # # return np.concatenate([pose, lh, rh])
    # pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results_pose.pose_landmarks.landmark]).flatten() if results_pose.pose_landmarks else np.zeros(33*4)
    # lh = np.array([[res.x, res.y, res.z] for res in results_hands.left_hand_landmarks.landmark]).flatten() if results_hands.left_hand_landmarks else np.zeros(21*3)
    # rh = np.array([[res.x, res.y, res.z] for res in results_hands.right_hand_landmarks.landmark]).flatten() if results_hands.right_hand_landmarks else np.zeros(21*3)
    # return np.concatenate([pose, lh, rh])
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results_pose.pose_landmarks.landmark]).flatten() if results_pose.pose_landmarks else np.zeros(33*4)
    
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    
    if results_hands.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            hand = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            if idx == 0:
                lh = hand
            elif idx == 1:
                rh = hand
    
    return np.concatenate([pose, lh, rh])


def prob_viz(res, actions, image, color):
    output_image = image.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_image, (0, 60+num*40), (int(prob*100), 90+num*40), color, -1)
        cv2.putText(output_image, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_image

# Define un solo color para todas las acciones (por ejemplo, azul)
action_color = (255, 0, 0)  # BGR format (Blue)




# Captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(image_rgb)
    results_pose = pose.process(image_rgb)
    
    # Dibujar los landmarks en la imagen
    image_with_landmarks = draw_landmarks(image, results_hands, results_pose)
    
    # Verificar si se detecta una mano
    if results_hands.multi_hand_landmarks:
        # Extraer keypoints
        keypoints = extract_keypoints(results_hands, results_pose)
        
        # Añadir keypoints a la secuencia y mantener solo los últimos 33
        sequence.append(keypoints)
        sequence = sequence[-33:]
        current_time = time.time()
        if len(sequence) == 33 and current_time - last_prediction_time >= prediction_interval:
            # Realizar predicción
            input_data = np.expand_dims(sequence, axis=0)
            res = model.predict(input_data)[0]
            predicted_action = actions[np.argmax(res)]
            print(predicted_action)
            predictions.append(np.argmax(res))
            sequence.clear()  # Limpiar la secuencia cuando no se detecta mano
            
            # Lógica de visualización
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if predicted_action != sentence[-1]:
                            sentence.append(predicted_action)
                    else:
                        sentence.append(predicted_action)
            
            
            # Visualizar probabilidades
            image_with_landmarks = prob_viz(res, actions, image_with_landmarks, action_color)
            last_prediction_time = current_time
        # Mostrar la predicción en la imagen
        cv2.rectangle(image_with_landmarks, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image_with_landmarks, ' '.join(sentence), (3,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        # No se detecta mano
        sequence.clear()  # Limpiar la secuencia cuando no se detecta mano
        cv2.putText(image_with_landmarks, 'No hand detected', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Mostrar la imagen procesada
    cv2.imshow('Real-time Prediction', image_with_landmarks)
    
    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()